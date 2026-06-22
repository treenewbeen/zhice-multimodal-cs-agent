#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知册 · 离线质量评测
================================================================================
基于**本地真值数据**与**当前 src/ 模块**，
对作品做多维量化验证，产出可写入《验证报告》的指标与数据表格。

验证维度：
  - RAG 检索准确率：当前检索器选中的章节是否命中题→节对齐图谱 / 人验真值
  - 多模态理解准确度：返回图片 id 与真值图片的 P/R/F1、图文计数一致率、图片真实存在率
  - 幻觉抑制·接地率：guard 逐句核验答案是否被证据支持（抽样）
  - 对话连贯性：多轮镜像格式通过率 + Qwen 连贯性抽检
  - 本地综合质量分：用 Qwen 按 1-5 质量标准复评批跑结果并归一化
  - 工程稳定性：硬校验通过率、人验黄金行复现率、线上 /chat 接口性能与可用性

用法：
  python validate.py                 # 全量
  python validate.py --quick         # 小样本快速跑通（自检用）
  python validate.py --no-live       # 跳过线上接口压测
  可选：--workers N --ground-n N --judge-every K --live-n N --base URL

产物：result/validate_run.jsonl、result/validation_metrics.json、result/validation_metrics.md
注意：接地率与 Qwen 评分会调用 LLM 网关（走 cache/ 缓存）；请确认 .env 网关余额。
"""
import os
# ★ 关键：本机若开着 Mihomo/Clash 代理（HTTP_PROXY=127.0.0.1:7897），urllib 会走代理；
#   一旦代理退出端口就死，所有请求失败。这里在任何网络调用前清掉代理变量，强制直连
#   （文本/嵌入/裁判网关与线上服务器均为国产域名，直连即可）。
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)

import sys
import re
import csv
import json
import time
import hashlib
import argparse
import statistics
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# 复用当前线上模块（不依赖 old/ 旧壳）
from runtime import KB_DIR, ENV, read_jsonl, image_stem_map, MULTI_TURN_IDS, _cache_path  # noqa: E402
from agent import Assets, section_affinity, adjudicate, norm_slice  # 检索/仲裁/取证（零生成）  # noqa: E402
import guard      # verify_grounding                  # noqa: E402

RESULT = ROOT / "result"
BASELINE_SUBMISSION = RESULT / "baseline_submission.csv"   # 评测用基准提交（自行准备）
BASELINE_V20 = KB_DIR / "answers_submit_v20.csv"          # 回测对照基准
REFERENCE_SCORE = None   # 可选：填入线上评测参考分以校准离线口径；为 None 时仅报告本地指标

# 标准答案格式： "正文(含<PIC>)"  或  "正文",["img1","img2",...]
RET_PAT = re.compile(r'^"(.*)"(?:,(\[.*\]))?$', re.S)

# 5 档质量评分标准
RUBRIC = (
    "1分，质量差：回答未回应问题，结构混乱或缺失，图片无关或无帮助。\n"
    "2分，质量一般：回答部分回应问题，但不完整；结构较碎，图文结合较差或仅图部分有帮助。\n"
    "3分，质量中等：回答基本完整，但缺乏深度；结构清晰但图片仅一定帮助，未充分提升理解。\n"
    "4分，质量良好：回答清晰，较为全面；结构严谨连贯，组织合理，图片有助于理解文本。\n"
    "5分，质量优秀：回答详细，有深度；结构严谨连贯，图片与文本完美互补，显著提升理解效果。"
)
JUDGE_BASE = ENV.get("JUDGE_API_BASE_URL", "").rstrip("/")
JUDGE_KEY = ENV.get("JUDGE_API_KEY", "")
JUDGE_MODEL = ENV.get("JUDGE_MODEL", "qwen-plus")

NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 强制直连


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
def load_specs():
    """题目规格 {id: spec}，spec.raw 为原始问题串，spec.type ∈ {cs, manual}。"""
    return {s["id"]: s for s in read_jsonl(KB_DIR / "question_spec.jsonl")}


def load_slice_map():
    """题→节对齐图谱（真值）：{id: {manual_key, final_sec, pic_ids, tier, ...}}。"""
    return {r["id"]: r for r in read_jsonl(KB_DIR / "slice_map.jsonl")}


def load_golden():
    """人验对齐行集合（whitelist 40 + keep_list 70 = 110 行人工核对真值）。"""
    wl = set(json.loads((KB_DIR / "whitelist.json").read_text(encoding="utf-8")))
    kl = set(json.loads((KB_DIR / "keep_list.json").read_text(encoding="utf-8")))
    return wl | kl


def load_submission(path):
    """读标准格式提交 CSV 为 {id: {ret, answer, image_ids}}。"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ret = row["ret"]
            m = RET_PAT.match(ret)
            out[int(row["id"])] = {
                "ret": ret,
                "answer": m.group(1) if m else ret,
                "image_ids": json.loads(m.group(2)) if (m and m.group(2)) else [],
            }
    return out


def split_ret(ret):
    """标准格式 ret → (正文, [image_id,...])。"""
    m = RET_PAT.match(ret or "")
    if not m:
        return ret or "", []
    return m.group(1), (json.loads(m.group(2)) if m.group(2) else [])


def norm_text(s):
    """去空白后比较，避免格式差异干扰。"""
    return re.sub(r"\s+", "", s or "")


def pctl(xs, p):
    """简单分位数。"""
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round((len(xs) - 1) * p)))
    return xs[k]


# ----------------------------------------------------------------------------
# 本地 Qwen 裁判（移植自 g1_qwen_judge，带磁盘缓存与退避重试）
# ----------------------------------------------------------------------------
def qwen_call(messages, max_tokens=200, salt=""):
    """调用 Qwen 评测模型（OpenAI 兼容端点），命中 cache/ 则零成本复跑。"""
    key = hashlib.sha1(json.dumps([JUDGE_MODEL, messages, 0.0, max_tokens, salt],
                                  ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cp = _cache_path(key)
    if cp.exists():
        try:
            c = json.loads(cp.read_text(encoding="utf-8"))["content"]
            if c.strip():
                return c
        except Exception:
            pass
    body = json.dumps({"model": JUDGE_MODEL, "messages": messages,
                       "temperature": 0.0, "max_tokens": max_tokens}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                JUDGE_BASE + "/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {JUDGE_KEY}"},
                method="POST")
            with NO_PROXY_OPENER.open(req, timeout=120) as r:
                content = json.loads(r.read().decode())["choices"][0]["message"].get("content") or ""
            if content.strip():
                cp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
                return content
        except Exception:
            time.sleep(2 ** attempt)
    return ""


def judge_one(question, ret, salt=""):
    """对单条问答按 1-5 质量标准打分；失败返回 0（按缺测排除）。"""
    sys_p = ("你是客服问答评分员。首要标准是回答是否切实回应并解决了用户的问题；在此基础上参考：\n"
             + RUBRIC +
             "\n说明：只要回答直接回应了问题、信息具体准确、无明显缺漏，即应给4分及以上；"
             "5分不要求长篇幅，简洁而信息完整、图文恰当即可。<PIC>表示展示插图，其后列表为图片id。"
             '\n只输出JSON：{"score": 1-5}')
    txt = qwen_call([{"role": "system", "content": sys_p},
                     {"role": "user", "content": f"问题：{question}\n\n回答：{ret}"}], salt="qj" + salt)
    m = re.search(r'"score"\s*:\s*([1-5])', txt) or re.search(r"\b([1-5])\b", txt)
    return int(m.group(1)) if m else 0


# ----------------------------------------------------------------------------
# 检索采集（零生成）：答案/图片取自基准提交，题型/产品取自 question_spec，
# 仅用 section_affinity + adjudicate 复算"检索选中的章节 + 证据"——与 answer_manual
# 完全同一检索路径（传真实 qid，命中基准提交时写下的缓存，近乎零成本零费用）。
# ----------------------------------------------------------------------------
def retrieve_meta(spec, assets):
    """只做检索+仲裁，得到 {type, manual, sec, conf, evidence}，不调用生成模型。"""
    t = spec.get("type")
    mk = spec.get("product_key") or spec.get("llm_product")
    if t != "manual" or not mk or mk not in assets.secs_by:
        return {"type": t, "manual": mk}
    secs = assets.secs_by[mk]
    aff = section_affinity(spec["raw"], mk, assets)                 # 混合检索打分（嵌入已缓存）
    order = sorted(range(len(secs)), key=lambda j: -aff[j])
    top1 = order[0]
    margin = aff[order[0]] - (aff[order[1]] if len(order) > 1 else 0.0)
    pick, verdict, _ = adjudicate(spec["raw"], mk, aff, assets, spec["language"], spec["id"])  # 仲裁（已缓存）
    conf = "high" if (pick is not None and pick == top1 and margin >= 0.03) else ("mid" if pick is not None else "low")
    pidx = pick if pick is not None else top1
    sec = secs[pidx]
    # 证据范围与 answer_manual 实际生成一致：high=命中节；mid=锚节±1窗；low/弃权=宽证据
    if conf == "high":
        ev = norm_slice(sec["text"])
    elif conf == "mid":
        k = sec["sec_idx"]
        ev = norm_slice(" ".join(secs[j]["text"] for j in (k - 1, k, k + 1) if 0 <= j < len(secs)))
    else:
        ev = (norm_slice(" ".join(s["text"] for s in secs)) if sum(s["plain_len"] for s in secs) <= 14000
              else norm_slice(" ".join(secs[j]["text"] for j in order[:3])))
    return {"type": "manual", "manual": mk, "sec": sec["sec_idx"], "conf": conf,
            "verdict": verdict, "evidence": ev[:3500]}


def collect(specs, qids, sub_csv, assets, workers):
    """组装每题验证记录：answer/image_ids 取自基准提交，meta 由 retrieve_meta 复算。"""
    def work(qid):
        spec = specs[qid]
        a = sub_csv.get(qid, {"ret": "", "answer": "", "image_ids": []})
        try:
            meta = retrieve_meta(spec, assets)
        except Exception as e:
            meta = {"type": spec.get("type"), "error": repr(e)}
        return qid, {"ret": a["ret"], "answer": a["answer"], "image_ids": a["image_ids"], "meta": meta}

    out = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(work, q) for q in qids]
        done = 0
        for fut in as_completed(futs):
            qid, r = fut.result()
            out[qid] = r
            done += 1
            if done % 50 == 0:
                print(f"  [retrieve] {done}/{len(qids)}", flush=True)
    return out


# ----------------------------------------------------------------------------
# 各项指标
# ----------------------------------------------------------------------------
def metric_retrieval(run, specs, smap, golden):
    """RAG 检索准确率：检索器选中章节 vs 题→节真值（总体/产品/人验子集/分层）。"""
    rows = []
    for qid, r in run.items():
        if specs.get(qid, {}).get("type") != "manual":
            continue
        gm = smap.get(qid)
        if not gm:
            continue
        meta = r.get("meta", {})
        mk_ok = (meta.get("manual") == gm.get("manual_key"))
        sec_ok = mk_ok and (meta.get("sec") == gm.get("final_sec"))
        tier = (gm.get("tier") or "?").split("-")[0].split("+")[0]   # 归并 T1/T2/T3
        rows.append((qid, tier, mk_ok, sec_ok, qid in golden))
    n = max(1, len(rows))
    by_tier = {}
    for t in sorted(set(x[1] for x in rows)):
        sub = [x for x in rows if x[1] == t]
        by_tier[t] = {"n": len(sub), "sec_acc": round(sum(x[3] for x in sub) / len(sub), 4)}
    gold = [x for x in rows if x[4]]
    return {
        "n_manual": len(rows),
        "product_locate_acc": round(sum(x[2] for x in rows) / n, 4),   # manual_key 命中（产品定位）
        "section_hit_rate": round(sum(x[3] for x in rows) / n, 4),     # 章节命中（vs 对齐图谱，含一致性含义）
        "human_verified_acc": round(sum(x[3] for x in gold) / max(1, len(gold)), 4),  # 110 人验真值上的准确率
        "human_verified_n": len(gold),
        "by_tier": by_tier,
    }


def metric_image(run, specs, smap):
    """多模态/图文：image_id vs 真值 pic_ids 的 P/R/F1；图文计数一致率；图片真实存在率。"""
    stems = image_stem_map()
    tp = fp = fn = 0
    consistent = total = 0
    exists_ok = total_pics = 0
    n_with_gt = 0
    for qid, r in run.items():
        ans = r.get("answer", "")
        ids = r.get("image_ids", [])
        total += 1
        if ans.count("<PIC>") == len(ids):
            consistent += 1
        for i in ids:
            total_pics += 1
            if i in stems:
                exists_ok += 1
        if specs.get(qid, {}).get("type") == "manual":
            gm = smap.get(qid)
            if gm is not None:
                gt = set(gm.get("pic_ids", []))
                pred = set(ids)
                if gt or pred:
                    n_with_gt += 1
                    tp += len(pred & gt)
                    fp += len(pred - gt)
                    fn += len(gt - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "image_precision": round(prec, 4),
        "image_recall": round(rec, 4),
        "image_f1": round(f1, 4),
        "n_eval": n_with_gt,
        "pic_count_consistency": round(consistent / max(1, total), 4),  # <PIC>数==图数
        "image_exists_rate": round(exists_ok / max(1, total_pics), 4),  # 引用图片真实存在
        "total_pics": total_pics,
    }


def metric_grounding(run, specs, n, workers):
    """幻觉抑制·接地率：guard 逐句核验 manual 答案是否被证据支持（抽样 n 题，并发）。"""
    cands = [(qid, r) for qid, r in run.items()
             if specs.get(qid, {}).get("type") == "manual"
             and r.get("meta", {}).get("evidence") and r.get("answer")]
    cands.sort()
    sample = cands[:n] if n else cands

    def work(it):
        qid, r = it
        try:
            g = guard.verify_grounding(r["answer"], r["meta"]["evidence"])
            return float(g.get("grounded_rate", 0.0)) if g else None
        except Exception:
            return None

    rates = []
    with ThreadPoolExecutor(workers) as ex:
        for gr in ex.map(work, sample):
            if gr is not None:
                rates.append(gr)
    full = sum(1 for x in rates if x >= 0.999)
    return {
        "n": len(rates),
        "mean_grounded_rate": round(statistics.mean(rates), 4) if rates else None,
        "fully_grounded_pct": round(full / max(1, len(rates)), 4),
        "min": round(min(rates), 4) if rates else None,
    }


def metric_conf_dist(run, specs):
    """检索置信分布（high/mid/low），manual 题。"""
    c = Counter(r.get("meta", {}).get("conf") for q, r in run.items()
                if specs.get(q, {}).get("type") == "manual" and r.get("meta", {}).get("conf"))
    return dict(sorted(c.items()))


def metric_coherence(run, specs, judge_n):
    """对话连贯性：多轮题镜像格式通过率 + Qwen 连贯性抽检。"""
    mt = [q for q in MULTI_TURN_IDS if q in run]
    mirror_ok = 0
    for q in mt:
        r = run[q]
        pol = r.get("meta", {}).get("policy", "")
        if '",\n"' in r.get("ret", "") or pol.endswith("multiturn"):
            mirror_ok += 1
    # 连贯性抽检：让 Qwen 判断多轮答案是否逐问承接、无串答
    coh = []
    for q in mt[:judge_n]:
        spec = specs.get(q, {})
        subs = spec.get("sub_questions") or []
        if not subs:
            continue
        prompt_q = "（多轮追问）" + " || ".join(subs)
        s = judge_one(prompt_q, run[q].get("ret", "")[:3500], salt=f"coh{q}")
        if s:
            coh.append(s)
    return {
        "multiturn_n": len(mt),
        "mirror_pass_rate": round(mirror_ok / max(1, len(mt)), 4),
        "coherence_mean_1to5": round(statistics.mean(coh), 3) if coh else None,
        "coherence_n": len(coh),
    }


def metric_judge(sub, specs, qids, every, workers):
    """本地综合质量分：用 Qwen 按 1-5 质量标准给提交逐题打分并归一化。
    仅评 qids 范围内的题（quick 模式只评样本；全量模式评全部 400）。"""
    items = [(q, sub[q]) for q in sorted(qids) if q in sub]
    if every > 1:
        items = items[::every]

    def work(it):
        qid, a = it
        q = specs.get(qid, {}).get("raw", "")
        return qid, judge_one(q, a["ret"][:3500], salt=str(qid))

    scores = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(work, it) for it in items]
        done = 0
        for fut in as_completed(futs):
            qid, s = fut.result()
            scores[qid] = s
            done += 1
            if done % 50 == 0:
                print(f"  [judge] {done}/{len(items)}", flush=True)
    valid = [v for v in scores.values() if v > 0]
    mean = sum(valid) / max(1, len(valid))
    return {
        "n_scored": len(valid),
        "mean_1to5": round(mean, 3),
        "normalized": round(mean / 5.0, 4),
        "reference_score": REFERENCE_SCORE,
        "abs_gap": (None if REFERENCE_SCORE is None else round(abs(mean / 5.0 - REFERENCE_SCORE), 4)),
        "dist_1to5": dict(sorted(Counter(valid).items())),
    }


def metric_hardcheck(sub, qids):
    """硬校验通过率：格式正确 / <PIC>数==图数 / 图片真实存在 / 单轮无杂换行。"""
    stems = image_stem_map()
    bad = Counter()
    for qid in qids:
        a = sub.get(qid)
        if not a or a["ret"] in ('""', ""):
            bad["empty"] += 1
            continue
        if not RET_PAT.match(a["ret"]):
            bad["format"] += 1
            continue
        if a["answer"].count("<PIC>") != len(a["image_ids"]):
            bad["pic_count"] += 1
        if any(i not in stems for i in a["image_ids"]):
            bad["img_missing"] += 1
        if "\n" in a["ret"] and '",\n"' not in a["ret"]:
            bad["stray_newline"] += 1
    n = len(qids)
    passed = n - sum(bad.values())
    return {"n": n, "pass": passed, "pass_rate": round(passed / max(1, n), 4), "defects": dict(bad)}


def metric_golden_repro(sub, v20, golden):
    """人验黄金行复现率：110 行人验答案 vs 基准 v20 逐字一致比例。"""
    hit = sum(1 for q in golden if q in sub and q in v20
              and norm_text(sub[q]["answer"]) == norm_text(v20[q]["answer"]))
    return {"hit": hit, "total": len(golden), "rate": round(hit / max(1, len(golden)), 4)}


def metric_interface(specs, qids, base, token, n):
    """线上 /chat 接口性能与可用性：成功率、延迟分位、/health 探活。"""
    sample = qids[:n]
    lat, ok, err = [], 0, 0
    for qid in sample:
        q = specs.get(qid, {}).get("raw", "")
        body = json.dumps({"question": q}).encode()
        t0 = time.time()
        try:
            req = urllib.request.Request(base + "/chat", data=body,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {token}"}, method="POST")
            with NO_PROXY_OPENER.open(req, timeout=40) as r:
                d = json.loads(r.read().decode())
            lat.append(time.time() - t0)
            if d.get("code") == 0 and d.get("data", {}).get("answer"):
                ok += 1
        except Exception:
            err += 1
    health_ok = 0
    for _ in range(20):
        try:
            with NO_PROXY_OPENER.open(base + "/health", timeout=10) as r:
                if json.loads(r.read().decode()).get("ok"):
                    health_ok += 1
        except Exception:
            pass
    return {
        "requests": len(sample), "success": ok, "errors": err,
        "success_rate": round(ok / max(1, len(sample)), 4),
        "latency_p50_s": round(pctl(lat, 0.5), 2),
        "latency_p90_s": round(pctl(lat, 0.9), 2),
        "latency_p99_s": round(pctl(lat, 0.99), 2),
        "health_pings": f"{health_ok}/20",
    }


# ----------------------------------------------------------------------------
# 报告渲染
# ----------------------------------------------------------------------------
def render_md(M):
    """把指标字典渲染为 markdown（含数据表格）。"""
    r, im, gd = M["retrieval"], M["image"], M["grounding"]
    co, ju, hc = M["coherence"], M["judge"], M["hardcheck"]
    gr, itf = M["golden"], M.get("interface")
    L = []
    L.append("# 验证报告补充 · 离线自建量化验证\n")
    L.append(f"> 生成时间：{M['ts']}　|　批跑题数：{M['n_questions']}\n")
    L.append("本报告基于本地真值数据（题→节对齐图谱、110 行人验对齐、"
             "基准提交、插图清单）与当前线上模块，独立量化各项能力。\n")

    L.append("## 一、核心指标总览\n")
    L.append("| 维度 | 指标 | 结果 |")
    L.append("|---|---|---|")
    L.append(f"| RAG 检索 | 产品定位准确率 | **{r['product_locate_acc']:.1%}** |")
    L.append(f"| RAG 检索 | 章节命中率（vs 对齐图谱） | **{r['section_hit_rate']:.1%}** |")
    L.append(f"| RAG 检索 | 人验真值检索准确率（{r['human_verified_n']} 行） | **{r['human_verified_acc']:.1%}** |")
    L.append(f"| 多模态 | 图片 F1 / 精确率 / 召回率 | **{im['image_f1']:.1%}** / {im['image_precision']:.1%} / {im['image_recall']:.1%} |")
    L.append(f"| 多模态 | 图文计数一致率 | **{im['pic_count_consistency']:.1%}** |")
    L.append(f"| 多模态 | 引用图片真实存在率 | **{im['image_exists_rate']:.1%}** |")
    if gd['mean_grounded_rate'] is not None:
        L.append(f"| 幻觉抑制 | 平均接地率（抽样 {gd['n']}） | **{gd['mean_grounded_rate']:.1%}** |")
        L.append(f"| 幻觉抑制 | 完全接地答案占比 | **{gd['fully_grounded_pct']:.1%}** |")
    L.append(f"| 对话连贯 | 多轮镜像格式通过率（{co['multiturn_n']}） | **{co['mirror_pass_rate']:.1%}** |")
    if co['coherence_mean_1to5'] is not None:
        L.append(f"| 对话连贯 | Qwen 连贯性评分（1-5，抽样 {co['coherence_n']}） | **{co['coherence_mean_1to5']}** |")
    _gap = "" if ju['abs_gap'] is None else f"（参考分 {ju['reference_score']}，差 {ju['abs_gap']:.3f}）"
    L.append(f"| 综合质量 | 本地 Qwen 归一化分（n={ju['n_scored']}） | **{ju['normalized']:.4f}**{_gap} |")
    L.append(f"| 工程稳定 | 硬校验通过率（{hc['n']} 题） | **{hc['pass_rate']:.1%}** |")
    if itf:
        L.append(f"| 接口稳定 | 线上成功率 / p50 / p90 延迟 | **{itf['success_rate']:.1%}** / {itf['latency_p50_s']}s / {itf['latency_p90_s']}s |")
    L.append("")

    L.append("## 二、RAG 检索准确率（分层）\n")
    L.append("| 层级 tier | 题数 | 章节命中率 |")
    L.append("|---|---|---|")
    for t, v in r["by_tier"].items():
        L.append(f"| {t} | {v['n']} | {v['sec_acc']:.1%} |")
    L.append(f"\n检索置信分布（manual，仲裁后）：**{M.get('conf_dist', {})}**\n")
    L.append(f"说明：章节命中率以题→节对齐图谱（单调 DP + 仲裁构建）为参照，含检索可复现性含义；"
             f"**人验真值检索准确率 {r['human_verified_acc']:.1%}**（{r['human_verified_n']} 行人工核对）为独立准确率口径。\n")

    L.append("## 三、综合质量分（本地 Qwen 复现）\n")
    L.append(f"对基准提交逐题按 1-5 质量标准复评：归一化质量分 **{ju['normalized']:.4f}**"
             + ("。\n" if ju['abs_gap'] is None else f"，与参考分 {ju['reference_score']} 差 {ju['abs_gap']:.3f}（用于校准离线评测口径）。\n"))
    L.append("| 分值 | 1 | 2 | 3 | 4 | 5 |")
    L.append("|---|---|---|---|---|---|")
    d = ju["dist_1to5"]
    L.append("| 题数 | " + " | ".join(str(d.get(i, 0)) for i in range(1, 6)) + " |")
    L.append("")

    L.append("## 四、硬校验与稳定性\n")
    L.append(f"- 硬校验通过率：**{hc['pass_rate']:.1%}**（{hc['pass']}/{hc['n']}）；缺陷：{hc['defects'] or '无'}")
    L.append(f"- 人验真值检索准确率：**{r['human_verified_acc']:.1%}**（{r['human_verified_n']} 行人工核对，检索独立准确率口径）")
    if itf:
        L.append(f"- 线上接口：成功率 **{itf['success_rate']:.1%}**（{itf['success']}/{itf['requests']}），"
                 f"延迟 p50 {itf['latency_p50_s']}s / p90 {itf['latency_p90_s']}s / p99 {itf['latency_p99_s']}s，"
                 f"健康探活 {itf['health_pings']}")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="小样本快速跑通（自检）")
    ap.add_argument("--no-live", action="store_true", help="跳过线上接口压测")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ground-n", type=int, default=80, help="接地率抽样题数")
    ap.add_argument("--judge-every", type=int, default=1, help="综合分评测抽样步长(>1 省费)")
    ap.add_argument("--live-n", type=int, default=40, help="线上压测请求数")
    ap.add_argument("--base", default="http://your-server-host:8000")
    ap.add_argument("--token", default="<KAFU_API_TOKEN>")
    args = ap.parse_args()

    RESULT.mkdir(exist_ok=True)
    specs = load_specs()
    smap = load_slice_map()
    golden = load_golden()
    qids = sorted(specs.keys())
    if args.quick:
        # 各类型取少量，快速验证流程
        manual = [q for q in qids if specs[q].get("type") == "manual"][:8]
        cs = [q for q in qids if specs[q].get("type") == "cs"][:4]
        mt = [q for q in MULTI_TURN_IDS if q in specs][:4]
        qids = sorted(set(manual + cs + mt))
        args.ground_n = min(args.ground_n, 4)
        args.live_n = min(args.live_n, 3)
        print(f"[quick] 仅跑 {len(qids)} 题自检")

    sub_base = load_submission(BASELINE_SUBMISSION)   # 基准提交（答案+图片来源）
    v20 = load_submission(BASELINE_V20) if BASELINE_V20.exists() else {}

    print(f"[1/3] 检索采集 {len(qids)} 题（零生成，命中缓存）...", flush=True)
    assets = Assets.get()
    run = collect(specs, qids, sub_base, assets, args.workers)
    with open(RESULT / "validate_run.jsonl", "w", encoding="utf-8") as f:
        for qid in sorted(run):
            f.write(json.dumps({"id": qid, **run[qid]}, ensure_ascii=False) + "\n")
    errs = [q for q, r in run.items() if r.get("meta", {}).get("error")]
    if errs:
        print(f"  ⚠ 检索异常 {len(errs)} 题: {errs[:8]}")

    print("[2/3] 计算指标 ...", flush=True)
    M = {
        "ts": time.strftime("%Y-%m-%d %H:%M"),
        "n_questions": len(qids),
        "retrieval": metric_retrieval(run, specs, smap, golden),
        "conf_dist": metric_conf_dist(run, specs),
        "image": metric_image(run, specs, smap),
        "grounding": metric_grounding(run, specs, args.ground_n, args.workers),
        "coherence": metric_coherence(run, specs, judge_n=6 if not args.quick else 2),
        "judge": metric_judge(sub_base, specs, qids, args.judge_every, args.workers),
        "hardcheck": metric_hardcheck(sub_base, sorted(sub_base.keys())),
        "golden": metric_golden_repro(sub_base, v20, golden),
    }
    if not args.no_live:
        print("  线上接口压测 ...", flush=True)
        M["interface"] = metric_interface(specs, qids, args.base, args.token, args.live_n)

    print("[3/3] 写出报告 ...", flush=True)
    (RESULT / "validation_metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_md(M)
    (RESULT / "validation_metrics.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\n✓ 完成：result/validation_metrics.json / .md / validate_run.jsonl")


if __name__ == "__main__":
    main()
