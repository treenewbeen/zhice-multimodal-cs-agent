#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared model client used by agent adjudication and verification helpers.

The online agent uses this small runtime client for Qwen-style judgement calls:
retrieval adjudication, follow-up rewriting, and grounding checks. Offline CSV
scoring tools live in ``old/offline_tools``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

from runtime import ENV, _cache_path

JUDGE_BASE = ENV.get("JUDGE_API_BASE_URL", "").rstrip("/")
JUDGE_KEY = ENV.get("JUDGE_API_KEY", "")
JUDGE_MODEL = ENV.get("JUDGE_MODEL", "qwen3.5-27b")
JUDGE_THINKING = os.getenv("JUDGE_THINKING", "0") == "1"


def _sse_collect(resp):
    """Collect OpenAI-compatible SSE delta content into a single string."""
    content = []
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            c = delta.get("content")
            if c:
                content.append(c)
        except Exception:
            continue
    return "".join(content)


def qwen_call(messages, temperature=0.0, max_tokens=300, tag="", cache_salt="", think=None):
    """Call the Qwen-compatible judge model with disk cache and retry backoff."""
    if think is None:
        think = JUDGE_THINKING
    key = hashlib.sha1(
        json.dumps(
            [JUDGE_MODEL, messages, temperature, max_tokens, cache_salt, bool(think)],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cp = _cache_path(key)
    if cp.exists():
        try:
            c = json.loads(cp.read_text(encoding="utf-8"))["content"]
            if c.strip():
                return c
        except Exception:
            pass

    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": bool(think),
    }
    if think:
        payload["stream"] = True
        payload["max_tokens"] = max(max_tokens, 1200)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                JUDGE_BASE + "/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {JUDGE_KEY}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                content = (
                    _sse_collect(r)
                    if think
                    else (json.loads(r.read().decode("utf-8"))["choices"][0]["message"].get("content") or "")
                )
            if content.strip():
                cp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
                return content
            last = "empty"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} {e.read().decode('utf-8', errors='replace')[:200]}"
            if e.code in (429, 500, 503):
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(last)
        except Exception as e:
            last = repr(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"[{tag}] qwen失败: {last}")
