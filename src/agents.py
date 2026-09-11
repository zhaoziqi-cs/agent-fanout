"""
主 Agent + 并行 Subagent 编排（function calling 版）

采用 Orchestrator-Workers 拓扑，worker 数量在运行时动态确定：

  1. 主 agent 是一个 ReAct 循环，挂载两个工具：
     - web_search：单次联网搜索
     - dispatch_subagents：派发多个 subagent 并行调研
     由 LLM 根据问题自行路由，拓扑不固定。
  2. dispatch_subagents 一次派发 N 个 subagent，经 ThreadPoolExecutor
     并行执行，整批 wall-clock ≈ max(单个 subagent 耗时) 而非 sum。
  3. 每个 subagent 同为 ReAct 循环（仅挂载 web_search），
     其 trace 全程写入 State，供调用方按节点查看。

并发上限：单次派发最多 6 个 subagent；主 agent 10 步、subagent 8 步。
"""

import time, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop, Tool
from tavily_search import tavily_search, format_search_result


# ══════════════════════════════════════════════════════════════════════════
# 共享状态
# ══════════════════════════════════════════════════════════════════════════

class State:
    """一次调研的全部中间产物。

    整个编排过程共享同一个 State 实例：主 agent、各 subagent 与工具都读写它，
    而非通过函数参数层层透传。状态变更一律由方法表达
    （add_dispatch → record_subagent → add_parallel_stats）。

    注意这是「一次调研一份」，不是模块级单例：serve.py 的 /query 是并发端点，
    模块级全局会让两次调研互相污染。run_research() 每次新建实例并经闭包传给工具，
    因此 ReActLoop 对 State 无感知。
    """

    def __init__(self):
        self.subagents: dict[str, dict] = {}   # sid -> {subtopic, trace, duration, final_answer}
        self.dispatches: list[dict] = []       # 每次派发 {subtopics, subagent_ids}
        self.parallel_stats: list[dict] = []   # 每次派发的并行/串行耗时统计

    def add_dispatch(self, subtopics: list, subagent_ids: list) -> dict:
        """记录一次派发。使用真实的 subagent id，
        以保证拓扑节点与后续 subagent_step 事件的 id 一致。"""
        rec = {"subtopics": list(subtopics), "subagent_ids": list(subagent_ids)}
        self.dispatches.append(rec)
        return rec

    def record_subagent(self, sid: str, subtopic: str, result: dict) -> None:
        """记录某个 subagent 的执行结果。"""
        self.subagents[sid] = {"subtopic": subtopic,
                               "trace": result["trace"],
                               "duration": result["duration"],
                               "final_answer": result["final_answer"]}

    def add_parallel_stats(self, n: int, wall: float, serial_sum: float) -> dict:
        """记录一批 subagent 的并行 / 串行耗时对比。
        serial_sum 为本批各 subagent 耗时之和，作为串行基线的估算值。"""
        rec = {"n_subagents": n, "wall_clock": wall, "serial_sum": serial_sum,
               "speedup": round(serial_sum / wall, 2) if wall else 0}
        self.parallel_stats.append(rec)
        return rec


MAIN_SYSTEM = """你是AI产品竞品分析主分析师。你有 2 个工具：
- web_search：联网搜索一次，仅用于单一事实可一次答出的问题
- dispatch_subagents：复杂问题拆分为多个子课题并行调研，参数是一组子课题

【关键决策原则】
- 用户询问某AI产品的竞品情况时，先从不同角度设计分析路线，如：产品功能、技术能力、目标用户、商业模式、市场表现、竞争优势与劣势、发展趋势。
  若涉及2个及以上角度，必须使用dispatch_subagents并行调研，把各侧面拆成子课题并行处理，不要自己串行 web_search 多次。
  示例："DeepSeek、Claude Code、Cursor 竞品分析：产品功能、技术能力、目标用户"
       → dispatch_subagents(subtopics=["DeepSeek、Claude Code、Cursor 的产品功能对比",
                                       "三者的技术能力对比（代码生成、上下文、工具调用）",
                                       "三者的目标用户与定价策略对比"])
  子课题必须覆盖问题里点名的每一个侧面，一个都不能漏；每个子课题要能独立搜出结果。
- 只有单一事实问题（如"DeepSeek V3 的参数量是多少"）才直接 web_search
- 拿到子调研结果后，综合成结构化报告

报告要求：分维度组织，每个要点带来源，末尾给结论与不确定性说明。"""


# ══════════════════════════════════════════════════════════════════════════
# 工具定义
# ══════════════════════════════════════════════════════════════════════════

def make_web_search_tool() -> Tool:
    """主 agent 和 subagent 共用的联网搜索工具（描述统一，避免两边不一致）。"""
    return Tool(
        name="web_search",
        description="联网搜索一次，参数=查询词。返回若干条带来源的摘要。",
        fn=lambda query: format_search_result(tavily_search(query)),
        parameters={"type": "object",
                    "properties": {"query": {"type": "string",
                                             "description": "查询词，一句话"}},
                    "required": ["query"]})


def _dispatch_subagents(subtopics, state: State,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。

    派发 N 个 subagent 并行执行（ThreadPoolExecutor），收齐后返回汇总文本。
    serial=True 时退化为串行执行，用作并行 / 串行基线对比。
    state 由 run_research() 经闭包传入。
    """
    # 单次派发上限 6 个
    topics = [str(t).strip() for t in (subtopics or []) if str(t).strip()][:6]
    if not topics:
        return "subtopics 为空：需要给出一组子课题字符串"

    # 构造 (sid, subagent, topic) 三元组
    defs = []
    for topic in topics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools=[make_web_search_tool()],
            max_steps=8, model_tag="deepseek-chat(子)")
        defs.append((sid, sub, topic))

    # 记录派发，使用真实 subagent id
    dispatch_info = state.add_dispatch(topics, [sid for sid, _, _ in defs])
    if on_dispatch:
        on_dispatch(dispatch_info)

    def _run_one(sid, sub, topic):
        cb = None
        if on_subagent_step:
            cb = lambda step, sid=sid: on_subagent_step(sid, step)
        return sid, topic, sub.run(topic, on_step=cb)

    def _collect(done_iter):
        """统一的结果收集：写入 State 并触发回调。
        串行与并行仅 done_iter 的来源不同，收集逻辑共用一份。"""
        results = {}
        for sid, topic, res in done_iter:
            results[sid] = (topic, res)
            state.record_subagent(sid, topic, res)
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
        return results

    t0 = time.time()
    # 执行：serial=True 串行，否则并行
    if serial:
        # 串行基线，逐个执行
        results = _collect(_run_one(sid, sub, topic) for sid, sub, topic in defs)
    else:
        # 并行执行
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = [pool.submit(_run_one, sid, sub, topic) for sid, sub, topic in defs]
            results = _collect(fut.result() for fut in as_completed(futs))

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    stats = state.add_parallel_stats(len(defs), wall, serial_sum)

    # 汇总文本作为 Observation 回灌主 agent；单个子结果截断以控制上下文长度。
    # 按 defs 顺序拼接，避免并行执行导致子课题顺序随机。
    parts = [f"【子课题: {topic}】(用时{results[sid][1]['duration']}s)\n"
             f"{results[sid][1]['final_answer'][:500]}"
             for sid, _, topic in defs if sid in results]
    return (f"并行调研完成：{len(defs)} 个子调研员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def make_dispatch_tool(state: State, on_subagent_step: Callable = None,
                       on_subagent_done: Callable = None,
                       on_dispatch: Callable = None, serial: bool = False) -> Tool:
    """构造 dispatch_subagents 工具。state 和回调经闭包注入，
    所以工具签名里只有模型该填的参数（subtopics）。"""
    return Tool(
        name="dispatch_subagents",
        description=("派发多个子调研员并行调研。参数是子课题列表，"
                     "每个元素是一个独立调研侧面（如「XX产品功能」「主要技术能力」）。"
                     "适合多侧面问题；单一事实问题请直接用 web_search。"),
        fn=lambda subtopics: _dispatch_subagents(
            subtopics, state, on_subagent_step=on_subagent_step,
            on_subagent_done=on_subagent_done, on_dispatch=on_dispatch,
            serial=serial),
        parameters={"type": "object",
                    "properties": {"subtopics": {
                        "type": "array", "items": {"type": "string"},
                        "description": "子课题列表，每个元素是一个独立调研侧面"}},
                    "required": ["subtopics"]})


def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False) -> dict:
    """执行一次调研。返回 {final_answer, main_trace, subagents, parallel_stats, dispatches}。
    serial=True 时 subagent 串行执行，用作并行基线。"""
    state = State()          # 每次调研独立一份

    main = ReActLoop(
        agent_name="main",
        tools=[make_web_search_tool(),
               make_dispatch_tool(state,
                                  on_subagent_step=on_subagent_step,
                                  on_subagent_done=on_subagent_done,
                                  on_dispatch=on_dispatch,
                                  serial=serial)],
        max_steps=10,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,   # 主 agent 的派发引导
    )
    result = main.run(question, on_step=on_main_step)
    return {"final_answer": result["final_answer"],
            "main_trace": result["trace"],
            "subagents": state.subagents,
            "parallel_stats": state.parallel_stats,
            "dispatches": state.dispatches}


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    q = "DeepSeek、Claude Code、Cursor 竞品分析：产品功能、技术能力、目标用户"
    r = run_research(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n报告头:\n{r['final_answer'][:200]}")
