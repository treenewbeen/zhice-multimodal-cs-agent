#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层对话记忆与指代消解 (Dialogue Memory + Coreference Resolution)
================================================================
面向「多轮对话」的模块。

职责：
  1) 会话记忆——按 session_id 存储多轮 (问题, 答案) 历史；
  2) 指代消解——把含指代/省略的追问（如"那个呢""再详细点""它怎么拆"）结合历史
     改写为**可独立检索的完整问题**，再交给检索/作答，从而保持多轮连贯。

设计：进程内字典存储（生产可平滑替换为 Redis）；改写失败时回退原问（非破坏性）。
改写复用 model_client.qwen_call（与仲裁同一 Qwen 端点）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import parse_json_loose, log
from model_client import qwen_call

# session_id -> [(question, answer_ret), ...]
_SESSIONS: dict = {}


def history(session_id: str):
    """取某会话的历史轮次列表。"""
    return _SESSIONS.get(session_id, [])


def append(session_id: str, question: str, answer: str):
    """追加一轮到会话记忆。"""
    _SESSIONS.setdefault(session_id, []).append((question, answer))


def resolve(question: str, session_id: str) -> str:
    """
    指代消解：有历史则把追问改写为独立查询；无历史或改写失败则返回原问。
    仅取最近 3 轮作为上下文，控制 token 与延迟。
    """
    hist = _SESSIONS.get(session_id) or []
    if not hist:
        return question
    ctx = "\n".join(f"用户: {q}\n客服: {a[:120]}" for q, a in hist[-3:])
    sys_p = (
        "你是多轮对话改写器。根据对话历史，把用户最新一句可能含指代或省略的追问，"
        "改写成一个不依赖上下文、可独立检索的完整问题；保持与原问相同语言，不要新增信息。"
        '只输出JSON: {"q": "改写后的独立问题"}'
    )
    user_p = f"对话历史:\n{ctx}\n\n最新追问: {question}"
    try:
        out = parse_json_loose(qwen_call(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            temperature=0.0, max_tokens=200, tag="coref"))
        q2 = (out.get("q") or "").strip()
        if q2:
            return q2
    except Exception as e:
        log(f"[memory] 指代消解失败(回退原问): {type(e).__name__}")
    return question
