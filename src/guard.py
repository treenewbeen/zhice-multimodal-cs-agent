#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三重幻觉抑制 · 第三重：事后接地校验 (Post-hoc Groundedness Verification)
======================================================================
面向「幻觉抑制」的模块。

本智能体的三重抗幻觉机制：
  ① 证据边界约束 —— 生成时严格「仅依据题源章节、禁止节外信息」（见 agent.answer_manual / COMPOSE 提示词）；
  ② 仲裁弃权    —— 检索仲裁"都不是"则弃权降级，不强答（见 agent.adjudicate）；
  ③ 事后接地校验 —— 本模块：逐句核验答案是否被证据支持，输出未接地句与接地率。

设计：仅产出**元信息**（不改写答案），故对主链零回归；证据为空或调用失败时返回 None（不阻断）。
校验复用 model_client.qwen_call。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import parse_json_loose, log
from model_client import qwen_call


def verify_grounding(answer: str, evidence: str):
    """
    逐句接地校验。
    参数：
        answer   : 待校验答案（可含 <PIC> 占位，将被忽略）
        evidence : 该答案所依据的题源章节文本
    返回：
        {"grounded_rate": float, "ungrounded": [句, ...]}  或  None（证据缺失/调用失败）
    """
    ans = re.sub(r"<PIC>", "", answer or "").strip()
    if not ans or not evidence:
        return None
    sys_p = (
        "你是事实接地校验器。判断'答案'中的每个关键陈述是否能在'证据'中找到支持。"
        '只输出JSON: {"ungrounded": ["未被证据支持的句子", ...]}；若全部有据支持则输出空数组。'
    )
    user_p = f"证据:\n{evidence[:3500]}\n\n答案:\n{ans[:1500]}"
    try:
        out = parse_json_loose(qwen_call(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            temperature=0.0, max_tokens=400, tag="ground"))
        ung = [s for s in (out.get("ungrounded") or []) if isinstance(s, str) and s.strip()]
        n_sent = max(1, len(re.split(r"[。.!?！？]+", ans)))   # 粗估句数
        rate = round(max(0.0, 1.0 - len(ung) / n_sent), 3)
        return {"grounded_rate": rate, "ungrounded": ung[:5]}
    except Exception as e:
        log(f"[guard] 接地校验失败(忽略): {type(e).__name__}")
        return None
