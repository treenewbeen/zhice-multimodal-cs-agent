#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知册 · 多模态客服智能体 —— 线上部署 RESTful API 服务（赛方接口规范实现）。
================================================================================
本文件是「线上部署的服务」入口，部署到自有服务器后供在线访问与集成调用。
实现《接口定义说明》：核心端点 POST /chat，支持 JSON 文本 + Base64 图片，
Bearer 鉴权，返回标准响应信封 {code,msg,data:{answer,session_id,timestamp,...}}。

复用 ../src 下的核心智能体（路由→检索→仲裁→二因子忠实作答→硬校验）与三个增强模块
（perception 多模态感知 / memory 对话记忆 / guard 事后接地校验），本层只做协议适配与编排。

启动（开发）：
    cd api && uvicorn app:app --host 0.0.0.0 --port 8000
启动（生产，多进程）：
    cd api && gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 app:app
"""
import os
import base64
import binascii
import re
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

# 将核心智能体所在的 src/ 加入模块搜索路径（本服务复用其全部推理逻辑）。
PKG_ROOT = Path(__file__).resolve().parent.parent          # 源码包根目录
sys.path.insert(0, str(PKG_ROOT / "src"))

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import Assets, answer_question_ex     # 核心智能体
from runtime import ENV, image_stem_map, IMAGE_DIR  # .env / 插图清单 / 插图目录
import perception                                  # 多模态感知融合
import memory                                      # 对话记忆 + 指代消解
import guard                                       # 事后接地校验

app = FastAPI(
    title="知册 多模态客服智能体 API",
    version="1.0",
    description="多模态客服智能体的线上服务：POST /chat 进行多模态对话交互。",
)

# 鉴权令牌：优先读进程环境变量；若用源码包根目录 .env 部署，也能通过 runtime.ENV 读取。
# 设置后 /chat 强制校验 Bearer，用于鉴权。
KAFU_API_TOKEN = os.environ.get("KAFU_API_TOKEN", ENV.get("KAFU_API_TOKEN", ""))

# 静态托管知识库插图，使答案中的 image_id 可通过 URL 在线查看（图文验证）。
if IMAGE_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")

# 标准答案格式： "正文(含<PIC>)" 或 "正文",["img1","img2",...]
_RET_PAT = re.compile(r'^"(.*)"(?:,(\[.*\]))?$', re.S)
_IMG_DATA_PAT = re.compile(r"^data:image/(png|jpg|jpeg|webp);base64,(.+)$", re.I | re.S)
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_STEMS = None  # 延迟加载的 {image_id(stem): 文件名}


def _stems():
    """惰性获取插图 id→文件名 映射（用于拼接图片 URL）。"""
    global _STEMS
    if _STEMS is None:
        _STEMS = image_stem_map()
    return _STEMS


def _split_ret(ret: str):
    """把标准格式 ret 拆为 (正文, [image_id,...])。多轮答案直接原样返回正文。"""
    import json
    m = _RET_PAT.match(ret or "")
    if not m:
        return ret or "", []
    text = m.group(1)
    ids = json.loads(m.group(2)) if m.group(2) else []
    return text, ids


def _validate_image_payloads(images: Optional[List[str]]) -> None:
    """Validate contest image payload constraints before multimodal parsing."""
    for idx, item in enumerate(images or []):
        if not isinstance(item, str):
            raise HTTPException(status_code=400, detail=f"images[{idx}] must be a base64 data URL string")
        m = _IMG_DATA_PAT.match(item.strip())
        if not m:
            raise HTTPException(
                status_code=400,
                detail=f"images[{idx}] must use data:image/{{png/jpg/jpeg/webp}};base64,... format",
            )
        try:
            raw = base64.b64decode(m.group(2), validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail=f"images[{idx}] is not valid base64")
        if len(raw) > _MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail=f"images[{idx}] exceeds 5MB")


# --------------------------------------------------------------------------
# 请求 / 响应数据模型（《请求体 Request Body》）
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str                              # 核心必填：用户问题字符串（长度 > 1）
    images: Optional[List[str]] = []           # 可选：Base64 图片列表，0-3 张，每张 ≤5MB
    session_id: Optional[str] = None           # 可选：会话 ID，用于多轮对话；不传则自动生成
    stream: Optional[bool] = False             # 可选：是否流式（本实现同步返回完整答案）


class VerifyRequest(BaseModel):
    answer: str
    evidence: str


# --------------------------------------------------------------------------
# 端点
# --------------------------------------------------------------------------

@app.on_event("startup")
def _warmup():
    """进程启动即预载知识库（章节库 / 产品目录 / 插图清单），避免首请求冷启动延迟。"""
    Assets.get()
    _stems()


@app.get("/health")
def health():
    """健康检查：负载均衡 / 探活用。"""
    return {"ok": True, "service": "zhice-multimodal-cs-agent", "version": "1.0"}


@app.post("/chat")
def chat(
    req: ChatRequest,
    request: Request,
    authorization: str = Header(default=""),       # Bearer 鉴权
    x_request_id: str = Header(default=""),         # 可选：请求追踪 id
    x_client_type: str = Header(default=""),        # 可选：调用方终端标识
):
    """
    核心端点：多模态对话交互。

    流程：鉴权 → 取/建 session → 指代消解(多轮) → 多模态感知融合(图) →
          核心智能体推理 → 事后接地校验 → 写回会话记忆 → 标准响应信封返回。
    各增强环节均 try/except 兜底：无图片 / 无历史 / 无视觉后端时主链不受影响。
    """
    # 1) 鉴权（仅当部署设置了 KAFU_API_TOKEN 时强制）
    if KAFU_API_TOKEN and authorization.strip() != f"Bearer {KAFU_API_TOKEN}":
        raise HTTPException(status_code=401, detail="invalid or missing Bearer token")
    # 2) 入参校验（question 必填、长度 > 1；images 至多 3 张）
    if not req.question or len(req.question.strip()) <= 1:
        raise HTTPException(status_code=400, detail="question length must be greater than 1")
    if req.images and len(req.images) > 3:
        raise HTTPException(status_code=400, detail="at most 3 images are allowed")
    _validate_image_payloads(req.images)

    sid = req.session_id or f"kf_{uuid.uuid4().hex[:12]}"

    # 3) 指代消解：结合历史把追问改写为可独立检索的问题（无历史/失败则原问）
    try:
        resolved = memory.resolve(req.question, sid)
    except Exception:
        resolved = req.question

    # 4) 多模态感知融合：用户上传图 → 关键事实 → 融合进检索查询（无图/无视觉后端则原样）
    try:
        fused, image_facts = perception.perceive(resolved, req.images)
    except Exception:
        fused, image_facts = resolved, []

    # 5) 核心推理（路由→检索→仲裁→二因子忠实作答→硬校验）
    ret, meta = answer_question_ex(fused, qid=None)

    # 6) 事后接地校验（第三重抗幻觉）：仅手册类有 evidence 时执行；仅作元信息，不改答案
    grounding = None
    try:
        if meta.get("evidence"):
            text_only, _ = _split_ret(ret)
            grounding = guard.verify_grounding(text_only, meta["evidence"])
    except Exception:
        grounding = None

    # 7) 写回会话记忆
    try:
        memory.append(sid, req.question, ret)
    except Exception:
        pass

    # 8) 组装响应：正文 + 图片 id + 可在线查看的图片 URL
    text, image_ids = _split_ret(ret)
    stems = _stems()
    base = str(request.base_url).rstrip("/")
    image_urls = [f"{base}/images/{stems[i]}" for i in image_ids if i in stems]

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "answer": text,                    # 答案正文（<PIC> 标记图片插入位置）
            "image_ids": image_ids,            # 图片 id 列表（与 <PIC> 顺序一一对应）
            "image_urls": image_urls,          # 图片在线 URL（可直接访问查看）
            "session_id": sid,
            "timestamp": int(time.time()),
            "meta": {k: v for k, v in meta.items() if k != "evidence"},  # 路由/章节/置信/策略，便于追溯
            "grounding": grounding,            # 接地校验：接地率与未接地句（抗幻觉证据）
            "x_request_id": x_request_id or None,
        },
    }


@app.post("/verify")
def verify(req: VerifyRequest):
    """事后接地校验独立端点：给定答案与证据，返回接地率与未接地句（演示第三重抗幻觉）。"""
    return {"code": 0, "msg": "success", "data": guard.verify_grounding(req.answer, req.evidence)}
