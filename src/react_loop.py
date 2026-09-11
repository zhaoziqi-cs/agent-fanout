"""
通用 ReAct 循环引擎（function calling 版）

主 agent 与 subagent 共用同一引擎，差异仅在于注入的工具集合。

每轮循环：模型返回 tool_calls 即「行动」，工具执行结果以 role="tool"
消息回灌即「观察」；模型不再返回 tool_calls 即收敛，输出最终答案。

消息协议：assistant(带 tool_calls) 与 tool(带 tool_call_id) 必须成对出现。
每个 tool_call_id 都要有一条对应的 tool 消息，缺失会导致下一轮请求 400。

每一步的 Action/Observation 记录进 trace，并通过 on_step 回调实时暴露，
供上层做流式推送。

共享状态不由引擎持有：工具通过闭包捕获所需的状态对象
（见 agents.py 的 dispatch 工具），ReActLoop 对此无感知，保持通用。

依赖：仅 llm_client + 工具函数，无外部库
"""

import time, json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from llm_client import llm_chat_tools


@dataclass
class Tool:
    """一个可被模型调用的工具。

    fn：实际执行的函数。形参名必须与 parameters 中的 properties 严格对齐——
        引擎以 fn(**args) 调用，参数缺失会抛 TypeError，该异常将作为观察结果回灌。
    parameters：标准 JSON Schema，例如
        {"type": "object",
         "properties": {"query": {"type": "string", "description": "查询词"}},
         "required": ["query"]}
    description 与 parameters 会直接进入提交给模型的提示，描述越精确，
    工具误选与参数错误的概率越低。
    """
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict

    def schema(self) -> dict:
        """转成 chat.completions 需要的 tools 格式。"""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


REACT_SYSTEM = """你是AI技术分析助手，需要外部信息时调用工具。

规则：
- 拿到足够信息后直接给出最终答案，不要再调用工具
- 每次工具调用都要有明确目的，不要重复同一个查询
- 引用信息时带上来源"""


class ReActLoop:
    """通用 ReAct 循环（function calling）。主 agent / subagent 各自实例化一个。"""

    def __init__(self, agent_name: str, tools: list,
                 max_steps: int = 10, model_tag: str = "deepseek-chat",
                 system_prompt: Optional[str] = None):
        """
        tools: list[Tool]
        model_tag: 写入 trace 的标识，用于区分同一份 trace 内的不同 agent；
                   实际调用的模型由 llm_client.DEEPSEEK_MODEL 决定。
        system_prompt: 系统提示。None 时使用默认 REACT_SYSTEM。
        """
        self.agent_name = agent_name
        self.tools = list(tools)
        # 按名索引，避免每次 tool_call 都线性扫描工具列表
        self.tools_by_name = {t.name: t for t in self.tools}
        self.max_steps = max_steps
        self.model_tag = model_tag
        self.system = system_prompt or REACT_SYSTEM
        self.trace: list[dict] = []   # 本轮执行的 trace

    def run(self, question: str, on_step: Callable = None) -> dict:
        """
        执行 ReAct 循环。
        on_step(step_dict): 每步回调，用于流式推送；None 时不回调。
        返回 {final_answer, trace, duration}。
        """
        self.trace = []
        t0 = time.time()
        system = self.system
        schemas = [t.schema() for t in self.tools]
        # 对话历史：user 提问 + assistant(带 tool_calls) + tool(观察结果) 交替
        messages: list[dict] = [{"role": "user", "content": question}]
        step_idx = 0
        final_answer = ""

        while step_idx < self.max_steps:
            msg = llm_chat_tools(system, messages, schemas,
                                 temperature=0.0, max_tokens=768)
            tool_calls = msg.tool_calls or []
            thought = (msg.content or "").strip()

            # 无工具调用：模型已给出最终答案，循环收敛
            if not tool_calls:
                final_answer = thought
                step = {"idx": step_idx, "agent": self.agent_name,
                        "thought": thought, "action": "Final Answer",
                        "action_input": final_answer, "observation": None,
                        "final": True}
                self.trace.append(step)
                if on_step: on_step(step)     # final：单次回调
                return {"final_answer": final_answer, "trace": self.trace,
                        "duration": round(time.time() - t0, 2)}

            # 有工具调用：assistant 消息原样回填，维持 tool_call_id 配对
            messages.append(msg)

            # 逐个执行工具，每个 tool_call_id 都必须回一条 tool 消息。
            # 模型一次返回多个 tool_call 时，即一条 assistant 消息后跟多条 tool 消息。
            for tc in tool_calls:
                if step_idx >= self.max_steps:   # 步数用尽，剩下的 tool_call 不再执行
                    break
                step = {"idx": step_idx, "agent": self.agent_name,
                        "thought": thought, "action": tc.function.name,
                        "action_input": tc.function.arguments or "",
                        "observation": None, "final": False}

                # 工具执行前先回调一次（observation 为 None），
                # 调用方无需等待工具返回即可收到该决策
                if on_step: on_step(step)

                # 执行工具。dispatch_subagents 等工具耗时可能较长
                observation = self._exec_tool(tc)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": observation})

                # 执行完成后以同一 idx 再次回调，携带真实 observation，
                # 调用方据此原地更新该步骤而非追加新条目
                step["observation"] = observation
                step["done"] = True
                self.trace.append(step)
                if on_step: on_step(step)
                step_idx += 1
        else:
            # 跑满 max_steps 仍在调用工具：返回兜底答案，避免空结果
            tail = self.trace[-1].get("observation", "") if self.trace else ""
            final_answer = "（已达最大步数）" + (tail or "")
            step = {"idx": self.max_steps, "agent": self.agent_name,
                    "thought": "达到步数上限", "action": "Final Answer",
                    "action_input": final_answer, "observation": None, "final": True}
            self.trace.append(step)
            if on_step: on_step(step)

        return {"final_answer": final_answer, "trace": self.trace,
                "duration": round(time.time() - t0, 2)}

    def _exec_tool(self, tool_call) -> str:
        """执行单个 tool_call，始终返回可回灌给模型的文本，不向外抛异常。

        以下四类失败一律作为一条观察结果交回模型，而非中断循环：
            1. 参数不是合法 JSON
            2. 参数不是 JSON 对象
            3. 工具不存在
            4. 工具执行抛出异常

        将错误交回模型而非向外抛出，是为了保留其自我纠错的机会：
        错误信息本身对模型就是有效的反馈信号。
        """
        name = tool_call.function.name
        raw = tool_call.function.arguments or ""
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            return f"参数解析失败：{e}，收到的是 {raw!r}，请重新用合法 JSON 调用"
        if not isinstance(args, dict):
            return f"参数格式错误：期望 JSON 对象，收到 {raw!r}"

        tool = self.tools_by_name.get(name)
        if tool is None:
            return f"工具 '{name}' 不存在，可选: {list(self.tools_by_name)}"
        try:
            result = tool.fn(**args)
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"
        # 协议要求 role="tool" 的 content 必须是字符串
        return result if isinstance(result, str) else str(result)


# 冒烟测试：单工具 ReAct 全流程
if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    from tavily_search import tavily_search, format_search_result

    web_search = Tool(
        name="web_search", description="联网搜索一次，参数=查询词",
        fn=lambda query: format_search_result(tavily_search(query)),
        parameters={"type": "object",
                    "properties": {"query": {"type": "string", "description": "查询词"}},
                    "required": ["query"]})

    loop = ReActLoop("smoke", tools=[web_search], max_steps=4)
    r = loop.run("DEEPSEEK v4 pro的参数有多少？")
    print(f"\n答案: {r['final_answer'][:120]}")
    print(f"trace {len(r['trace'])} 步:")
    for s in r["trace"]:
        print(f"  [{s['idx']}] {s['action']}({s['action_input'][:40]}) → {(s.get('observation') or '')[:50]}")
