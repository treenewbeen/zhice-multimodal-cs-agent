#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知册 · 多模态客服智能体 — 核心库（纯标准库实现，无第三方重型依赖）。
================================================================================
本模块是被 agent / perception / memory / guard / api 等上层调用的**运行时核心库**，
封装了智能体推理所需的全部底层能力。所有外部模型调用（文本生成 / 向量嵌入 / 视觉 /
裁判）均用标准库 urllib 实现并带磁盘缓存，故部署无需 numpy/faiss/torch 等。

────────────────────────── 函数索引（按职责分组） ──────────────────────────
· 环境与缓存
    load_env            读取包根 .env（密钥/网关）
    _cache_path         缓存键 → 磁盘路径（前两位分桶）
· 外部模型调用（OpenAI 兼容，urllib + 磁盘缓存 + 退避重试）
    llm_call            文本生成（路由/作答用）
    embed_call          向量嵌入（章节检索用）
    vision_call         多模态视觉（解析用户图）
    cosine              余弦相似度（纯 Python）
· JSON / 文本工具
    parse_json_loose    宽松解析模型 JSON（多重兜底）
    _repair_json / _repair_inner_quotes   非法转义 / 内部裸引号修复
    lang_of             语言判别（zh/en）
    read_jsonl / write_jsonl              JSONL 读写
    sanitize_body       正文规整为提交格式
· 知识与题目
    load_questions      读 400 题
    load_policies       读客服政策库
    image_stem_map      插图清单（存在性校验 / URL）
    detect_count_constraint  计数约束识别
    block_prior         题号先验（诊断用，默认关）
· 客服（CS）作答与校验
    raw_turns           多轮拆分
    gen_cs_turns / gen_cs_answer          多轮 / 单轮 CS 生成
    validate_cs / validate_cs_turns       CS 校验闸
· 标准格式
    banned_hit          禁用语/幻觉腔过滤
    build_ret           装配标准 ret 字符串

知识库构建（从原始手册建库）的 stage_* 阶段已归档到 old/tools/（构建期，部署不需要）。

运行入口（详见包根 README.md）：
    python src/agent.py one "问题"        # 单题
    python src/agent.py batch --workers 24 # 全量 400 题
    cd api && uvicorn app:app               # 启动线上服务
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
csv.field_size_limit(10 ** 9)

ROOT = Path(__file__).resolve().parent.parent
# ---- 干净三层结构：src/(代码) · knowledge_base/(数据+插图，扁平) · result/(产物) · cache/(运行缓存) ----
KB_DIR = ROOT / "knowledge_base"      # 知识库根：所有数据文件扁平存放
RESULT_DIR = ROOT / "result"          # 批跑产物
DATA = KB_DIR                         # 数据 = 知识库扁平目录
CACHE = ROOT / "cache"                # LLM/嵌入磁盘缓存（运行时生成，可安全删除）
REPORTS = ROOT / "cache"              # 回测/校验报告
for d in (DATA, CACHE, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

MANUAL_DIR = KB_DIR                   # （构建期）手册原文目录
IMAGE_DIR = KB_DIR / "images"         # 2608 张说明书插图（image_id = 文件名 stem）
EN_COMBINED = KB_DIR / "汇总英文手册.txt"   # （构建期，包内不附）
QUESTION_FILE = KB_DIR / "question_public.csv"
POLICY_FILE = KB_DIR / "customer_policy_v2.jsonl"
SEED_FILE = KB_DIR / "catalog_seed.json"
STRESS_PLAN = KB_DIR / "STRESS_SAMPLE_PLAN.csv"

MANUALS_OUT = KB_DIR / "manuals.jsonl"
CATALOG_OUT = KB_DIR / "catalog_locked.json"
SPEC_OUT = KB_DIR / "question_spec.jsonl"
CHUNKS_OUT = KB_DIR / "chunks.jsonl"
EVIDENCE_OUT = KB_DIR / "evidence_pack.jsonl"
ANSWERS_OUT = KB_DIR / "answers.jsonl"          # 累积式: 每id一行,重跑覆盖
JUDGE_OUT = REPORTS / "judge_report.csv"
VERIFY_OUT = REPORTS / "verifier_report.csv"
SUBMIT_OUT = RESULT_DIR / "answers_submit.csv"
FULL_OUT = RESULT_DIR / "answers_full.csv"

# 冒烟用的 10 题代表样本（覆盖多产品/中英/多轮）。
SAMPLE10 = [1, 46, 67, 89, 124, 145, 244, 322, 417, 434]

# ----------------------------------------------------------------------------
# env / LLM client
# ----------------------------------------------------------------------------

def load_env():
    """读取**包根目录** .env 为 dict（不读系统环境变量）。
    形如 KEY=VALUE，忽略空行与 # 注释行；供 LLM/嵌入/裁判/视觉各端点取用密钥与网关地址。
    源码提交不含 .env，部署时单独配置。"""
    # 逐行解析 KEY=VALUE，跳过空行与注释；
    # 返回 dict 供各端点取密钥/网关。
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env

ENV = load_env()
# 文本生成端点/密钥/模型（从 .env 读取；默认 deepseek，可切 Claude/opus）。
TEXT_BASE = ENV.get("TEXT_API_BASE_URL", "https://api.deepseek.com").rstrip("/")
TEXT_KEY = ENV.get("TEXT_API_KEY", "")
TEXT_MODEL = ENV.get("TEXT_MODEL", "deepseek-chat")
# 视觉端点/密钥/模型（多模态感知用，可选配置）。
VISION_BASE = ENV.get("VISION_API_BASE_URL", "").rstrip("/")
VISION_KEY = ENV.get("VISION_API_KEY", "")
VISION_MODEL = ENV.get("VISION_MODEL", "")
# 向量嵌入端点/密钥/模型（章节检索用）。
EMBED_BASE = ENV.get("EMBEDDING_API_BASE_URL", "").rstrip("/")
EMBED_KEY = ENV.get("EMBEDDING_API_KEY", "")
EMBED_MODEL = ENV.get("EMBEDDING_MODEL", "")
EMBED_OUT = CACHE / "chunk_embeddings.jsonl"

_print_lock = threading.Lock()

def log(*a):
    """线程安全打印（加锁 + flush），多线程批跑时日志不串行错乱。"""
    with _print_lock:
        print(*a, flush=True)

class LLMError(Exception):
    pass

def _cache_path(key: str) -> Path:
    """由缓存键 sha1 计算磁盘缓存文件路径：按键前 2 位分桶建子目录，避免单目录文件过多。"""
    # 以键前两位分桶，避免单目录文件过多导致变慢。
    d = CACHE / key[:2]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"

def llm_call(messages, temperature=0.2, max_tokens=2200, tag="", force_json=True, cache_salt=""):
    """OpenAI 兼容**文本生成**调用（纯 urllib 实现，无第三方 SDK）。
    机制：
      · 按 (模型,消息,温度,max_tokens,salt) 计算 sha1 做磁盘缓存——命中即零成本零延迟；
      · 网络/限流(429/5xx)异常 4 次重试 + 指数退避；
      · force_json=True 时要求并解析严格 JSON。
    参数: messages OpenAI 格式消息; temperature 采样温度; tag/cache_salt 区分缓存。
    返回: 模型输出文本。"""
    # 实现：先查磁盘缓存（命中直接返回，省钱省时）；
    # 未命中再发 HTTP，对 429/5xx 退避重试；
    # 成功后写回缓存。force_json 时请求体加 response_format。
    # 命中缓存直接返回（省钱省时）；
    # 未命中 POST；网络/限流异常退避后重试 4 次；
    # force_json 时请求并解析严格 JSON。
    payload = {
        "model": TEXT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}
    key = hashlib.sha1(
        json.dumps([TEXT_MODEL, messages, temperature, max_tokens, cache_salt], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cp = _cache_path(key)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))["content"]
        except Exception:
            pass
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                TEXT_BASE + "/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {TEXT_KEY}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=150) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            content = out["choices"][0]["message"]["content"]
            cp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
            return content
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            # response_format不支持时退化
            if e.code == 400 and force_json and "response_format" in err_body:
                payload.pop("response_format", None)
                body = json.dumps(payload).encode("utf-8")
                force_json = False
                continue
            last_err = f"HTTP {e.code} {err_body}"
            if e.code in (429, 500, 502, 503):
                time.sleep(2 ** attempt * 2)
                continue
            break
        except Exception as e:  # 网络/超时
            last_err = repr(e)
            time.sleep(2 ** attempt * 2)
    raise LLMError(f"[{tag}] LLM调用失败: {last_err}")

def embed_call(texts, tag=""):
    """文本**向量嵌入**调用（DashScope 兼容端点，支持批量）。
    逐条文本按内容做磁盘缓存（避免重复计费）；返回与输入等长的向量列表，
    供章节检索 section_affinity 与产品路由兜底使用。"""
    # 实现：逐条文本独立缓存，已缓存的不重复请求；
    # 未命中的批量送嵌入端点，回填后按原顺序返回。
    # 已缓存条目不重复请求；
    # 未命中的批量送嵌入端点、回填后按原顺序返回。
    vecs = [None] * len(texts)
    miss_idx = []
    for i, t in enumerate(texts):
        key = hashlib.sha1(f"emb|{EMBED_MODEL}|{t}".encode("utf-8")).hexdigest()
        cp = _cache_path(key)
        if cp.exists():
            try:
                vecs[i] = json.loads(cp.read_text(encoding="utf-8"))["v"]
                continue
            except Exception:
                pass
        miss_idx.append(i)
    for s in range(0, len(miss_idx), 10):
        batch = miss_idx[s:s + 10]
        payload = {"model": EMBED_MODEL, "input": [texts[i][:6000] for i in batch]}
        body = json.dumps(payload).encode("utf-8")
        last = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    EMBED_BASE + "/embeddings", data=body,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {EMBED_KEY}"},
                    method="POST")
                with urllib.request.urlopen(req, timeout=90) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                for j, d in zip(batch, out["data"]):
                    v = [round(x, 5) for x in d["embedding"]]
                    vecs[j] = v
                    key = hashlib.sha1(f"emb|{EMBED_MODEL}|{texts[j]}".encode("utf-8")).hexdigest()
                    _cache_path(key).write_text(json.dumps({"v": v}), encoding="utf-8")
                last = None
                break
            except Exception as e:
                last = repr(e)
                time.sleep(2 ** attempt)
        if last:
            raise LLMError(f"[{tag}] embedding失败: {last}")
    return vecs

def cosine(a, b):
    """两向量**余弦相似度**（纯 Python 点积/模长实现，避免引入 numpy/faiss）。"""
    # 点积除以两模长之积；零向量返回 0 防除零。
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / na / nb

def vision_call(text_prompt: str, img_path: str, tag="", max_tokens=700):
    """**多模态视觉**调用：一张图片 + 文本提示 → 视觉模型理解文字。
    OpenAI 兼容多模态端点（图片走 data:image;base64），带磁盘缓存。
    供 perception 多模态感知模块解析用户上传图（部件/报错码/损坏状态）。"""
    # 实现：图片读为 base64 内联进 OpenAI 多模态消息；
    # 同样走磁盘缓存；失败重试。
    # 图片读为 base64 内联进多模态消息；
    # 同样磁盘缓存；失败重试。
    import base64
    key = hashlib.sha1(f"vision|{VISION_MODEL}|{text_prompt}|{img_path}".encode("utf-8")).hexdigest()
    cp = _cache_path(key)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))["content"]
        except Exception:
            pass
    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    ext = "png" if img_path.lower().endswith("png") else "jpeg"
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                VISION_BASE + "/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {VISION_KEY}"},
                method="POST")
            with urllib.request.urlopen(req, timeout=150) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            content = out["choices"][0]["message"].get("content") or ""
            if content.strip():
                cp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
                return content
            last_err = "empty content"
            payload["max_tokens"] = min(1400, payload["max_tokens"] + 400)
            body = json.dumps(payload).encode("utf-8")
        except Exception as e:
            last_err = repr(e)
            time.sleep(2 ** attempt)
    raise LLMError(f"[{tag}] vision失败: {last_err}")

def _repair_json(t: str) -> str:
    """修复模型 JSON 输出的常见**非法转义**（手册含反斜杠时模型常输出无效转义序列），提升解析成功率。"""
    # 对常见无效转义做保守替换，尽量不动合法部分。
    # 修复非法转义(手册内容含反斜杠时模型常输出 \x 这类无效转义)
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", t)

def _repair_inner_quotes(t: str) -> str:
    """结构化扫描，重新转义 JSON 字符串内部**未转义的裸 ASCII 双引号**。
    模型常对中文术语用未转义的双引号导致 JSON 解析崩溃；本函数逐字符跟踪字符串状态、
    仅对字符串内部的裸引号补转义，纯 additive 兜底（不破坏已正确的部分）。"""
    # 逐字符跟踪是否在字符串内，仅给内部裸引号补反斜杠。
    out, n, in_str, i = [], len(t), False, 0
    while i < n:
        ch = t[i]
        if ch == "\\" and in_str and i + 1 < n:
            out.append(ch); out.append(t[i + 1]); i += 2; continue
        if ch == '"':
            if not in_str:
                p = i - 1
                while p >= 0 and t[p] in " \t\r\n":
                    p -= 1
                if p < 0 or t[p] in "{[,:":
                    in_str = True; out.append(ch)
                else:
                    out.append('\\"')
            else:
                nx = i + 1
                while nx < n and t[nx] in " \t\r\n":
                    nx += 1
                if nx >= n or t[nx] in ",}]:":
                    in_str = False; out.append(ch)
                else:
                    out.append('\\"')
        else:
            out.append(ch)
        i += 1
    return "".join(out)

def parse_json_loose(text: str):
    """**宽松解析**模型返回的 JSON（多重兜底，最大化鲁棒性）。
    依次尝试：①strict 解析 → ②非法转义修复 → ③内部裸引号重转义 → ④花括号子串提取，
    从噪声输出中尽量恢复结构化结果。"""
    # 逐级兜底：strict → 修非法转义 → 修内部裸引号 → 提取首个花括号子串；
    # 全失败抛异常由调用方处理。
    # strict 失败后逐级修复；
    # 末位用栈匹配提取首个完整花括号子串再解析。
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    subs = [t]
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        subs.append(t[i:j + 1])
    # 原行为优先(base + _repair_json, strict=True); 其余仅作兜底, 不影响已能解析的输入
    for base in subs:
        for cand in (base, _repair_json(base), _repair_inner_quotes(base), _repair_inner_quotes(_repair_json(base))):
            for strict in (True, False):
                try:
                    return json.loads(cand, strict=strict)
                except Exception:
                    pass
    raise ValueError("LLM输出不是JSON: " + t[:200])

# ----------------------------------------------------------------------------
# 共用工具
# ----------------------------------------------------------------------------

CJK_RE = re.compile(r"[一-鿿]")

def lang_of(text: str) -> str:
    """按 CJK 字符占比粗判文本语言，返回 'zh' 或 'en'（路由与答案语言一致性校验用）。"""
    # 统计 CJK 字符占比，过阈值判中文，否则英文。
    zh = len(CJK_RE.findall(text))
    return "zh" if zh >= max(2, len(text) * 0.05) else "en"

def read_jsonl(p: Path):
    """逐行读取 JSONL 文件为 dict 迭代器（跳过空行）。"""
    # 逐行 json.loads，遇空行跳过，惰性产出。
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

def write_jsonl(p: Path, rows):
    """把 dict 列表写为 JSONL 文件（UTF-8，每行一条 JSON）。"""
    # 每条 json.dumps(ensure_ascii=False) 一行写出。
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def image_stem_map():
    """扫描插图目录，返回 {文件名 stem: 完整文件名}。
    用途：①答案 image_id 的**存在性硬校验**；②线上服务把 image_id 拼成可访问的图片 URL。"""
    # 遍历插图目录，建立 文件名stem→文件名 映射，O(1) 校验图是否存在。
    # 遍历插图目录建立 stem→文件名 映射，供 O(1) 存在性校验。
    m = {}
    for f in IMAGE_DIR.iterdir():
        if f.is_file():
            m[f.stem] = f.name
    return m

# 禁用短语集：拒答腔 / 元话术 / 检索腔 / 目录页码残留——
# 答案命中其一即判失败、触发重试或降级（抗幻觉关键闸）。
BANNED_PHRASES = [
    "请提供订单号", "请您提供订单号或具体商品信息，我们会帮您核实处理",
    "无法回答", "没有足够明确的依据", "说明书中没有", "建议结合具体型号向售后",
    "provide your order number", "cannot answer", "no information available",
    "根据检索", "根据提供的资料", "检索到的", "evidence", "chunk_id",
    "based on the provided", "the manual does not contain",
    "not included in the provided", "provided content", "this excerpt",
    "not available in this excerpt", "the excerpt", "未包含在", "本摘录",
    "提供的内容中", "provided manual content", "in the given content",
    "manual does not", "manual doesn't", "not covered in the manual",
    "no dedicated", "说明书未", "说明书中未", "手册未", "手册中未", "手册没有",
    "are not covered", "is not covered", "not described in",
]
# 目录页码残留正则（如 ‘..... 12’），供 banned_hit 检测。
TOC_RESIDUE_RE = re.compile(r"\.{2,}\s*\d{1,3}\b|…{2,}\s*\d{1,3}\b")

def banned_hit(ans: str):
    """命中**禁用短语**则返回该短语（否则空字符串）。
    禁用集涵盖：拒答腔/元话术（无法回答/说明书未提及）、检索腔（根据检索/提供的内容/excerpt）、
    目录页码残留等。生成结果命中即判失败、触发重试或降级——抗幻觉与不当措辞的关键闸。"""
    # 遍历禁用短语集，命中即返回该短语；
    # 另检目录页码残留正则。
    # 遍历禁用短语集，命中即返回该短语并触发重试/降级。
    low = ans.lower()
    for b in BANNED_PHRASES:
        if b.lower() in low:
            return b
    if ans.lstrip().startswith("#"):
        return "markdown#"
    if TOC_RESIDUE_RE.search(ans):
        return "toc_residue"
    return None

# ----------------------------------------------------------------------------
# Stage: catalog
# ----------------------------------------------------------------------------

OCR_FIXES = [
    (re.compile(r"\$\\boxed\{[^}]*\}\$"), " "),
    (re.compile(r"\\textcircled\{([^}]{1,4})\}"), r"(\1)"),
    (re.compile(r"\\circ\b"), "°"),
    (re.compile(r"\\(?:twoheadrightarrow|blacktriangleright|rightarrow)"), "→"),
    (re.compile(r"\\(?:blacktriangleleft|leftarrow|leftrightarrow)"), ""),
    (re.compile(r"\^\{([^}]*)\}"), r"\1"),
    (re.compile(r"_\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\(?:mathsf|textperth|overbrace|phantom|left|right|begin|end)\b\{?"), ""),
    (re.compile(r"\$"), ""),
    (re.compile(r"\bEJU\d{4,6}\b"), ""),
    (re.compile(r"\bCAUTloN\b", re.I), "CAUTION"),
    (re.compile(r"\bNOTlCE\b", re.I), "NOTICE"),
    (re.compile(r"[ \t]{3,}"), "  "),
]

# ----------------------------------------------------------------------------
# Stage: spec (题目规格+路由)
# ----------------------------------------------------------------------------

ZH_BLOCKS = [
    (64, 69, "zh_吹风机"), (70, 85, "zh_空调"), (86, 88, "zh_蒸汽清洁机"), (89, 91, "zh_人体工学椅"),
    (92, 103, "zh_洗碗机"), (104, 112, "zh_空气净化器"), (113, 122, "zh_健身单车"), (123, 130, "zh_电钻"),
    (131, 144, "zh_健身追踪器"), (145, 152, "zh_冰箱"), (153, 172, "zh_发电机"), (173, 180, "zh_摩托艇"),
    (181, 185, "zh_水泵"), (186, 194, "zh_可编程温控器"), (195, 199, "zh_VR头显"), (200, 205, "zh_功能键盘"),
    (206, 207, "zh_儿童电动摩托车"), (208, 215, "zh_蓝牙激光鼠标"), (216, 227, "zh_烤箱"), (228, 234, "zh_相机"),
]
EN_BLOCKS = [
    (241, 241, "en_airfryer"), (242, 264, "en_boat"), (265, 270, "en_coffee_machine"), (271, 279, "en_boat"),
    (280, 295, "en_camera"), (296, 302, "en_earphones"), (303, 310, "en_ereader"), (311, 316, "en_fax"),
    (317, 321, "en_grill"), (322, 350, "en_jetski"), (351, 357, "en_landline_phone"), (358, 365, "en_lawn_mower"),
    (366, 373, "en_microwave"), (374, 386, "en_motherboard"), (387, 400, "en_pressure_cooker_air_fryer"),
    (401, 412, "en_vacuum"), (413, 414, "en_security_camera"), (415, 426, "en_snowmobile"),
    (427, 433, "en_television"), (434, 436, "en_electric_toothbrush"),
]

def block_prior(qid: int):
    """由题号返回先验 (类型, 产品)——仅供老题诊断的题号先验提示，默认关闭（新问题不可用）。"""
    # 按题号区间返回先验类型/产品；仅诊断用，默认不启用。
    if 1 <= qid <= 58:
        return "cs", None
    for lo, hi, key in ZH_BLOCKS + EN_BLOCKS:
        if lo <= qid <= hi:
            return "manual", key
    return "manual", None

def load_questions():
    """读取 question_public.csv，返回题目列表（id / raw 原文 / language 等字段）。"""
    # 按 id 排序，附带语言等字段，供构建与回测。
    rows = []
    with QUESTION_FILE.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        for r in rd:
            if not r or not r[0].strip():
                continue
            qid = int(r[0])
            qraw = r[1] if len(r) > 1 else ""
            subs = re.findall(r'"([^"]+)"', qraw)
            subs = [s.strip() for s in subs if s.strip()]
            if not subs:
                subs = [qraw.strip()]
            rows.append({"id": qid, "raw": qraw, "subs": subs})
    return rows

COUNT_PATTERNS = [
    (re.compile(r"前([一二三四五六七八九十两\d]+)\s*(条|个|步|项)"), "first"),
    (re.compile(r"最后([一二三四五六七八九十两\d]+)\s*(条|个|步|项|步骤)"), "last"),
    (re.compile(r"first\s+(one|two|three|four|five|six|\d+)\s+steps?", re.I), "first"),
    (re.compile(r"last\s+(one|two|three|four|five|six|\d+)\s+steps?", re.I), "last"),
]
NUM_MAP = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
           "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

def detect_count_constraint(text: str):
    """识别题面的**计数约束**（如"前 3 步 / 最后 2 条"），返回 {kind:first|last, n}，
    供作答时严格控量、不多不少；无约束返回 None。"""
    # 用正则匹配'前/后 N 条/步'句式，归一为 {kind,n}。
    # 正则匹配‘前/后 N 条/步’句式，归一为 {kind,n}。
    for pat, kind in COUNT_PATTERNS:
        m = pat.search(text)
        if m:
            v = m.group(1).lower()
            n = NUM_MAP.get(v) or (int(v) if v.isdigit() else None)
            if n:
                return {"kind": kind, "n": n}
    return None

# ----------------------------------------------------------------------------
# Stage: evidence (切块 + BM25 + 证据包)
# ----------------------------------------------------------------------------

TOC_LINE_RE = re.compile(r"\.{3,}\s*\d{1,3}\s*$|…{2,}\s*\d{1,3}\s*$")
NON_ANSWERABLE_PAT = [
    "table of contents", "目录", "no part of this manual", "版权所有", "copyright",
    "all rights reserved",
]

WORD_RE = re.compile(r"[a-zA-Z0-9]+")

# ----------------------------------------------------------------------------
# Stage: gen (生成)
# ----------------------------------------------------------------------------

# 客服话术样例（few-shot）：注入生成提示以对齐口吻与时效表述。
CS_EXAMPLES = (
    '样例1(多子问题逐一作答): 问:"请问你们的商品能送到乡镇吗？""需要额外加运费吗？多久能到？" '
    '答:"您好，我们的商品支持送到大部分乡镇哦，具体能否送达取决于您的收货地址，您可以提供详细地址，我帮您查询。'
    '送到乡镇一般不需要额外加运费，和市区运费一致；物流时效会比市区稍慢，正常情况下下单后48小时发货，'
    '乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。"\n'
    '样例2: 问:"物流一直显示待揽收，什么原因？" '
    '答:"您好，物流显示待揽收，大概率是商品已打包完成，等待快递员上门取件哦，一般24小时内会完成揽收；'
    '若超过24小时仍未揽收，您可以联系我们客服，我们会催促快递方尽快上门。"'
)

def load_policies():
    """读取**售前售后客服政策库**，拼为文本（每条：意图 / 要点 / 参考话术）。
    CS 单轮/多轮作答时作为依据，保证退换货/物流/发票/投诉等口径统一。"""
    # 把每条政策的意图/要点/话术拼成可读文本块。
    # 把每条政策的意图/要点/参考话术拼成可读文本块。
    pols = read_jsonl(POLICY_FILE)
    lines = []
    for p in pols:
        pts = " / ".join(p.get("answer_points", []))
        lines.append(f"[{p.get('intent','')}] 要点: {pts}\n  参考话术: {p.get('answer_template','')}")
    return "\n".join(lines)

# 多轮话术样例镜像: 多轮题(题面含多段引号)逐轮作答
MULTI_TURN_IDS = {1, 2, 3, 4, 6, 17, 19, 20, 21, 22, 24, 26, 33, 34, 40, 41}

def raw_turns(spec):
    """题面的原始引号段即对话轮次（与 LLM 语义拆分无关）。
    ≥2 段引号 ⇒ 多轮题（镜像多轮话术样例）；否则回退到 LLM 拆分的子问题列表。"""
    # 正则抽题面所有引号段；≥2 段视为多轮，否则回退子问题。
    segs = [s.strip() for s in re.findall(r'"([^"]+)"', spec.get("raw", "")) if s.strip()]
    return segs if len(segs) >= 2 else spec["sub_questions"]

def gen_cs_turns(spec, policies_text, attempt=0):
    """**多轮**客服题逐轮作答（镜像多轮话术样例）。
    轮次 = 题面原始引号段；每段只答该轮、结合前轮**消解指代**
    （如第二轮"多久能收到"指发票 1-3 个工作日，而非商品物流时效）；
    您好口吻、给时效数字/责任归属、不配图。返回各轮回复列表（条数 == 轮数）。"""
    # 拼系统提示（含两轮客服话术样例做 few-shot）+ 用户多轮问题 + 政策库；
    # 要求模型输出严格 JSON 的逐轮答案数组；
    # 解析后逐轮规整空白。
    # few-shot 注入两轮客服话术样例引导口吻；
    # 强约束输出严格 JSON 的逐轮答案数组；
    # 解析后逐轮规整空白。
    subs = raw_turns(spec)
    # 多轮系统提示设计要点：
    #   ①每段只答对应轮、不串轮；②强制结合前轮消解指代/省略（多轮连贯的关键）；
    #   ③"您好"口吻 + 时效数字 + 责任归属；④注入两轮客服话术样例做 few-shot。
    sys_p = (
        "你是电商平台金牌客服, 正在进行多轮对话。用户的提问分为多轮, 你需要对每一轮分别给出一段独立回复:\n"
        "1. 第k段回复只回答第k轮的问题(该轮内若有多个小问都要答全), 不要把后面轮次的内容提前;\n"
        "1b. 关键: 后续轮次中的代词和省略对象必须结合前面轮次理解。例如前一轮问'能开发票吗', 后一轮问'多久能收到呢'指的是发票多久收到(电子发票1-3个工作日), 绝不是商品物流时效;\n"
        "2. 每段回复都以完整口吻独立成段, 风格自然: 您好开头(第一轮), 亲切自然, 适当用'哦/呢', "
        "给出明确的政策结论、时效数字、责任归属;\n"
        "3. 每段60-200字; 禁止用'请提供订单号'代替实质回答; 不配图; 不用markdown。\n"
        '客服话术样例(两轮): 轮1问"能送到乡镇吗?" 轮1答"您好，我们的商品支持送到大部分乡镇哦，具体能否送达，取决于您的收货地址，您可以告诉我详细的收货地址，我帮您查询。" '
        '轮2问"需要额外加运费吗？多久能到？" 轮2答"送到乡镇一般不需要额外加运费，和市区运费一致；物流时效会比市区稍慢，正常情况下，下单后48小时发货，乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。"\n'
        '输出严格JSON: {"answers": ["第1轮回复", "第2轮回复", ...]} (条数必须等于轮数)'
    )
    user_p = (
        f"用户的{len(subs)}轮提问:\n" + "\n".join(f"第{i+1}轮: {s}" for i, s in enumerate(subs)) +
        f"\n\n政策库:\n{policies_text}"
    )
    out = parse_json_loose(llm_call(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        temperature=0.4, tag=f"csmt-{spec['id']}", cache_salt=f"csmt-v4-a{attempt}"))
    turns = [re.sub(r"\s+", " ", str(t)).strip() for t in (out.get("answers") or [])]
    return turns

def validate_cs_turns(spec, turns):
    """校验多轮 CS 答案：轮数匹配、每轮长度门(30-400)、禁用语(banned_hit)、不得含图。
    返回错误列表（空 = 通过；非空触发重试）。"""
    # 逐轮检查长度/禁用语/是否误带图，并校验轮数匹配。
    # 校验轮数匹配，并逐轮检查长度/禁用语/误带图。
    errs = []
    expect = len(raw_turns(spec))
    if len(turns) != expect:
        errs.append(f"turn_count:{len(turns)}!={expect}")
    for i, t in enumerate(turns):
        if len(t) < 30:
            errs.append(f"turn{i+1}_too_short")
        if len(t) > 400:
            errs.append(f"turn{i+1}_too_long")
        b = banned_hit(t)
        if b:
            errs.append(f"turn{i+1}_banned:{b}")
        if "<PIC" in t:
            errs.append(f"turn{i+1}_with_image")
    return errs

# 增强提示：回答不够深入时追加，要求补全流程/材料/时效/例外。
ENRICH_ZH = ("\n增强要求: 上一版回答不够深入。请在保持准确的前提下更完整: 补充具体流程步骤、所需材料/凭证、"
             "时效数字、注意事项与例外情况, 让每个子问题都有实质细节。")
# 手册题增强提示：要求覆盖全部步骤/参数/注意并适当配图。
ENRICH_MAN_ZH = ("\n增强要求: 上一版回答不够深入。请覆盖该主题的全部相关步骤/参数/数值/注意事项(含NOTE与警告), "
                 "并在步骤/部件/指示灯类内容紧邻处有图时配1-3张图。")
ENRICH_MAN_EN = ("\nENRICHMENT: the previous version was judged lacking depth. Cover ALL relevant steps/parameters/values/cautions "
                 "(including NOTE and WARNING details) for the topic, and include 1-3 adjacent images for steps/parts/indicator content.")

def gen_cs_answer(spec, policies_text, attempt=0, enrich=False, deep=False):
    """**单轮**客服题作答（依据政策库）。
    首句直答（能/不能/分情况 + 结论）再展开，逐一回答每个子问题、给明确时效/责任/流程；
    您好口吻、不配图、120-350 字（deep 时 180-400）。返回 (答案, coverage 覆盖度标注)。"""
    # 系统提示强调'首句直答 + 逐子问题作答 + 客服口吻 + 不配图'；
    # deep 时放宽长度上限；按 enrich/deep 区分缓存 salt；
    # 返回答案与覆盖度标注。
    # 首句直答 + 逐子问题作答 + 客服口吻 + 不配图；
    # deep 放宽长度上限；缓存 salt 区分增强档。
    subs = spec["sub_questions"]
    # 单轮系统提示设计要点：
    #   ①首句直答(能/不能/分情况+结论)杜绝含糊；②逐子问题给时效/责任/流程；
    #   ③禁止用"请提供订单号"代替实质回答；④客服口吻、不配图；⑤政策未覆盖时按电商通行惯例合理承诺、不编造极端承诺。
    sys_p = (
        "你是电商平台金牌客服。根据政策库回答用户问题。要求:\n"
        "1. 第一句必须直接回答用户核心问题(能/不能/分情况+结论), 再展开; 每个子问题都逐一、具体地回答, "
        "给出明确的时效数字/责任归属/流程步骤/所需材料, 不要含糊;\n"
        "2. 口吻自然: 您好开头, 亲切自然, 适当用'哦/呢', 不机械;\n"
        "3. 禁止用'请提供订单号'之类的话代替实质回答(可以在结尾补充一句协助核实, 但主体必须先给政策答案);\n"
        "4. 长度120-350字; 不配图; 不用markdown; 不分条编号也可以, 语句连贯即可;\n"
        "5. 政策库没有完全覆盖时, 按电商通行惯例给出合理、对用户友好的承诺, 但不要编造极端承诺(如无条件终身包退)。\n"
        '输出严格JSON: {"answer": "...", "coverage": [{"sub": "子问题", "answered_by": "答案中对应的关键句"}]}'
    )
    user_p = (
        f"用户问题(共{len(subs)}个子问题):\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(subs)) +
        f"\n\n政策库:\n{policies_text}\n\n客服话术样例:\n{CS_EXAMPLES}" +
        (ENRICH_ZH if (enrich or deep) else "")
    )
    if deep:
        sys_p = sys_p.replace("长度120-350字", "长度180-400字")
    salt = "cs-v3-" if deep else ("cs-e-" if enrich else "cs-")
    out = parse_json_loose(llm_call(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        temperature=0.4, tag=f"cs-{spec['id']}", cache_salt=f"{salt}a{attempt}"))
    ans = re.sub(r"\s+", " ", str(out.get("answer", ""))).strip()
    cov = out.get("coverage", [])
    return ans, cov

def validate_cs(spec, ans, cov):
    """校验单轮 CS 答案：长度门(80-600)、禁用语、语言一致、子问题覆盖度、不得含图。
    返回错误列表（空 = 通过）。"""
    # 逐项检查并累计错误：长度、禁用语、语言、覆盖度、是否误带图。
    # 长度/禁用语/语言/覆盖度/误带图 逐项累计错误。
    errs = []
    if len(ans) < 80:
        errs.append("too_short")
    if len(ans) > 600:
        errs.append("too_long")
    b = banned_hit(ans)
    if b:
        errs.append(f"banned:{b}")
    if lang_of(ans) != spec["language"]:
        errs.append("lang_mismatch")
    if len(spec["sub_questions"]) > 1 and isinstance(cov, list) and len(cov) < len(spec["sub_questions"]):
        errs.append("coverage_incomplete")
    if "<PIC" in ans:
        errs.append("cs_with_image")
    return errs

GOLD_SYS_ZH = (
    "你是产品说明书问答助手。从给定说明书内容中定位最能回答问题的章节, 以'贴原文摘编'方式输出答案, "
    "对齐以下满分范例的风格:\n"
    '范例问题:"DCB107或DCB112型号电钻指示灯闪烁代表什么含义?" '
    '范例答案:"DCB107、DCB112 电池组充电中 <PIC> 电池组已充满 <PIC> 过热/过冷延迟 <PIC>" (配3张图)\n'
    '范例问题:"我想更换健身追踪器的表带，有其他尺寸可选吗?" '
    '范例答案:"表带尺寸 表带尺寸如下所示。注意：单独销售的配件表带可能略有差异。<PIC> 环境条件 <PIC>" (配2张图)\n'
    "规则:\n"
    "1. 直接以相关章节内容作答: 保留原文措辞、步骤编号、参数数值; 可保留小节标题(纯文本, 不加#号); 与问题无关的句子删掉;\n"
    "2. 不要添加原文没有的开场白、过渡解释或总结; 不要写'根据说明书/手册中提到'之类的话;\n"
    "3. 图片是答案的核心组成: 你摘编的内容中出现的<PIC id=\"xxx\">, 按原文位置插入<PIC>(不带id), id按顺序放入image_ids, 最多5张; 文字短而图多是被鼓励的;\n"
    "4. 问题的每个小问都要覆盖; 计数要求(前N条/最后N步)必须严格满足数量与顺序;\n"
    "4b. 若问题指定了特定工况/型号/模式(如冷机vs热机、手动vs电动、某型号), 只摘对应小节的内容, 严禁混入相邻工况的步骤;\n"
    "5. 答案为中文; 禁止说'无法回答/未提及'; 内容不完全匹配时用最接近的章节摘编, 必要时可补一句通用建议(不得编造参数);\n"
    "6. 若产品名称易被误解(如'吹风机'实为汽油吹叶机), 可在开头用原文词语自然带出产品类型。\n"
    '输出严格JSON: {"answer": "...", "image_ids": ["..."], "quotes": []}'
)
GOLD_SYS_EN = (
    "You are a product-manual QA assistant. Locate the section that best answers the question and compose the answer "
    "by faithful excerpting, matching this full-score example style:\n"
    'Example Q: "What do the charger indicator flashes mean?" '
    'Example A: "DCB107, DCB112 battery pack charging <PIC> fully charged <PIC> hot/cold pack delay <PIC>" (3 images)\n'
    "Rules:\n"
    "1. Answer directly with the relevant section content: keep original wording, step numbers, parameter values; "
    "plain-text section titles allowed (no # symbols); drop sentences unrelated to the question;\n"
    "2. Do not add introductions, transitions or conclusions that are not in the manual; never write 'according to the manual';\n"
    "3. Images are a core part of the answer: for <PIC id=\"xxx\"> appearing inside the content you excerpt, insert <PIC> (without id) "
    "at the original positions and put ids in order into image_ids, max 5; short text with more images is encouraged;\n"
    "4. Cover every sub-question; count constraints (first/last N steps) must match exactly in number and order;\n"
    "4b. If the question targets a specific condition/model/mode (cold vs warm start, manual vs electric, a specific model), "
    "excerpt only that sub-section; never mix in steps from the adjacent condition;\n"
    "5. Answer in English; never say 'cannot answer / not mentioned'; if coverage is partial, excerpt the closest section "
    "and add one sentence of safe general guidance if needed (never invent specs);\n"
    '6. Output strict JSON: {"answer": "...", "image_ids": ["..."], "quotes": []}'
)

# ----------------------------------------------------------------------------
# Stage: judge (裁判模拟)
# ----------------------------------------------------------------------------

RUBRIC = (
    "1分,质量差: 回答未回应问题, 结构混乱或缺失, 图片无关或无帮助。\n"
    "2分,质量一般: 回答部分回应问题, 但不完整; 结构较碎, 图文结合较差或仅部分有帮助。\n"
    "3分,质量中等: 回答基本完整, 但缺乏深度; 结构清晰, 图片有一定帮助但未充分提升理解。\n"
    "4分,质量良好: 回答清晰, 较为全面; 结构严谨连贯, 组织合理, 图片有助于理解文本。\n"
    "5分,质量优秀: 回答详细, 有深度; 结构严谨连贯, 图片与文本完美互补, 显著提升理解效果。"
)

# ----------------------------------------------------------------------------
# Stage: imgcheck (MiMo 视觉验图)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Stage: verify (硬规则) / export
# ----------------------------------------------------------------------------

def sanitize_body(ans: str) -> str:
    """规整答案正文为标准提交格式：合并连续空白、去除真实换行、转义必要字符。"""
    # 合并连续空白、去真实换行、转义提交格式所需字符。
    # 合并连续空白、去真实换行、转义提交格式所需字符。
    s = ans.strip()
    s = s.replace("**", "")
    s = re.sub(r"\bEJU\d{4,6}\b", "", s)  # 雅马哈手册章节代码残留
    s = re.sub(r"\$[^$\n]{0,60}\$", "", s)  # OCR LaTeX 残留 ($^{...}$ / $\boxed{}$ 等)
    s = re.sub(r"^#{1,3}\s*", "", s, flags=re.M)
    s = s.replace('"', "'")
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def build_ret(a):
    """把 {answer, image_ids[, turn_answers]} 装配为**标准格式 ret 字符串**。
    单轮：「正文(含<PIC>)」+ 可选图片 id 数组；多轮：各轮引号段按多轮格式镜像拼接(逗号+换行分隔)。
    正文经 sanitize_body 规整，保证可被评分系统正确解析。"""
    # 先 sanitize_body 规整正文；
    # 有图则追加 JSON 图片数组；多轮则各轮引号段镜像拼接。
    # sanitize 正文后，按单轮(加图数组)/多轮(引号段镜像)拼标准格式。
    turns = a.get("turn_answers")
    if turns:
        ret = ",\n".join(f'"{sanitize_body(t)}"' for t in turns)
        if a["image_ids"]:
            ret += f',{json.dumps(a["image_ids"], ensure_ascii=False)}'
        return ret
    body = sanitize_body(a["answer"])
    if a["image_ids"]:
        return f'"{body}",{json.dumps(a["image_ids"], ensure_ascii=False)}'
    return f'"{body}"'

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
