#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态感知融合层 (Multimodal Perception Fusion)
================================================
面向「结合问题与图片识别用户意图」的模块。

职责：把用户在客服对话中上传的图片（Base64）经视觉模型理解为一句「产品部件 /
报错码 / 损坏状态 / 型号」的文字事实，并与问题文本做**跨模态融合**，生成统一的
检索查询表征，交给下游路由与检索使用。

设计：非破坏性——无图片、或未配置视觉后端（.env 的 VISION_*）时，原样返回问题文本，
不影响纯文本主链路。视觉调用复用 runtime.vision_call（OpenAI 兼容多模态端点 + 磁盘缓存）。
"""
import base64
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import vision_call, VISION_MODEL, log

# 引导视觉模型只输出对客服检索有用的关键信息，避免冗长描述。
_VISION_PROMPT = (
    "这是用户在客服对话中上传的图片。请用一句话描述图中与产品相关的关键信息："
    "产品部件 / 屏幕上的报错码或提示 / 损坏或异常状态 / 型号或标签，供客服检索定位手册章节。"
)


def _decode_to_tmpfile(b64: str) -> str:
    """把 'data:image/...;base64,xxx' 或纯 Base64 解码到临时 PNG 文件，返回路径。"""
    if b64.strip().startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def perceive(question: str, images=None):
    """
    多模态感知融合主入口。

    参数：
        question : 用户问题文本（必填）
        images   : Base64 图片字符串列表（0-3 张，可空）
    返回：
        (fused_query, image_facts)
        - fused_query : 融合了图片事实的检索查询（无图时即原问题）
        - image_facts : 每张图的关键信息描述列表（用于溯源/调试）
    """
    facts = []
    # 仅当确有图片且视觉后端已配置时才做视觉理解；否则直接走纯文本（非破坏性）。
    if images and VISION_MODEL:
        for i, b64 in enumerate(images[:3]):
            tmp = None
            try:
                tmp = _decode_to_tmpfile(b64)
                desc = (vision_call(_VISION_PROMPT, tmp, tag=f"perceive-{i}") or "").strip()
                if desc:
                    facts.append(desc)
            except Exception as e:  # 单图失败不影响整体，记录后跳过
                log(f"[perceive] 图{i} 理解失败(忽略): {type(e).__name__}")
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
    fused = question if not facts else f"{question} [图片信息: {'; '.join(facts)}]"
    return fused, facts
