"""LLM 客户端

DeepSeek deepseek-chat，经 OpenAI 兼容接口调用，内置指数退避重试。
依赖：pip install openai"""
import os, time, logging
from openai import OpenAI
logger = logging.getLogger(__name__)
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
_client = None
def get_client():
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key: raise EnvironmentError("请设置 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _client
def llm_chat(system, user, *, temperature=0.0, max_tokens=1024, stop=None, retries=3):
    """单轮纯文本对话，返回字符串内容（不带工具的场景用）。"""
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=temperature, max_tokens=max_tokens, stop=stop)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries-1: raise
            time.sleep(2**attempt); logger.warning(f"LLM 重试({attempt+1}): {str(e)[:80]}")


def llm_chat_tools(system, messages, tools=None, *, temperature=0.0,
                   max_tokens=1024, retries=3):
    """function calling 版：返回 assistant message 对象（含 .content / .tool_calls），
    而不是纯文本——ReAct 循环要靠 tool_calls 决定下一步行动。

    messages 是「system 之后」的对话历史（含 assistant 带 tool_calls 的消息、
    以及 role="tool" 的观察结果），由调用方维护。
    """
    kwargs = {"model": DEEPSEEK_MODEL,
              "messages": [{"role": "system", "content": system}, *messages],
              "temperature": temperature, "max_tokens": max_tokens}
    if tools:                       # 空列表时不能传，否则 API 报错
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(**kwargs)
            return resp.choices[0].message
        except Exception as e:
            if attempt == retries-1: raise
            time.sleep(2**attempt); logger.warning(f"LLM 重试({attempt+1}): {str(e)[:80]}")
