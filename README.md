# 知册 · 多模态客服智能体

> 一个**完整可运行**的端到端多模态客服智能体：基于图文产品说明书构建多模态知识库，
> 对用户问题（售前售后政策类 / 说明书操作类）做 **路由 → 检索 → 仲裁 → 忠实作答 → 硬校验**，
> 输出答案正文 + 相关插图 id。
>
> 核心库在 `src/`（纯标准库实现，无 numpy/faiss/torch 等重依赖），线上服务入口为 `api/app.py`（`POST /chat`）。

---

## 一、30 秒上手

```bash
# 1) 依赖（核心库仅标准库；以下为 API 服务所需）
pip install -r requirements.txt

# 2) 配置密钥：复制模板，填入你的网关/密钥
cp .env.example .env        # 然后编辑 .env

# 3a) 命令行单题（打印路由/检索/策略元信息 + 答案）
python src/agent.py one "如何安装空气净化器的脚轮？"

# 3b) 启动 RESTful API 服务（/chat 端点）
cd api && uvicorn app:app --host 0.0.0.0 --port 8000
#   POST http://localhost:8000/chat   body: {"question":"物流一直显示待揽收，是什么原因？"}
```

> Python 3.10+。首次调用会请求外部 LLM/嵌入接口（按 `.env`），结果写入 `cache/`，重复调用命中缓存。
> 知识库数据不随仓库分发，运行前需自行准备（见 [§七](#七知识库) 与 `knowledge_base/README.md`）。

---

## 二、目录结构

```
zhice-multimodal-cs-agent/
├── README.md  requirements.txt  .env.example
├── api/                          线上部署服务（RESTful，部署到自有服务器）
│   ├── app.py                    /chat（Bearer + Base64 图 + session）+ /verify + /health + 图片在线 URL
│   ├── 接口说明文档.md           完整接口定义（请求/响应/示例/错误码）
│   ├── 部署说明.md               服务器部署步骤（uvicorn/gunicorn/Docker）
│   └── requirements.txt  Dockerfile  .dockerignore
├── src/                          核心源码
│   ├── agent.py                  智能体主体：路由 → 检索 → 仲裁 → 忠实作答 → 硬校验
│   ├── runtime.py                运行时核心库：.env / LLM / embedding / 视觉 / 缓存 / 答案格式
│   ├── model_client.py           仲裁、指代消解、接地校验共用的轻量模型客户端
│   ├── perception.py             多模态感知融合：用户上传图 → 关键事实 → 融合检索查询
│   ├── memory.py                 分层对话记忆 + 指代消解：多轮 session + 追问改写
│   └── guard.py                  事后接地校验：逐句核验答案是否被证据支持
├── validate.py                   离线质量评测脚本（检索准确率 / 多模态 / 接地率 / 连贯性）
└── knowledge_base/               知识库目录（数据不含在仓库内，见该目录 README.md）
```

> 运行时还会生成 `cache/`（LLM/嵌入磁盘缓存，可安全删除）、`result/`（批跑产物）。

---

## 三、系统架构

```
用户请求 (question[, images, session_id])
        │
        ▼  api/app.py /chat   Bearer 鉴权 · 多模态入参 · 会话记忆 · 标准响应信封
        ▼
┌─ 1. 意图路由  route() ─────────────────────────────────────────┐
│   customer_service（政策/物流/退换/发票/投诉）vs manual（查手册）│
│   + 产品定位（LLM 选 + 向量兜底）+ 子问题拆分 + 计数约束识别       │
└──────────────┬─────────────────────────────────────────────────┘
       manual ◄┴► customer_service
        │                       │
        ▼                       ▼
┌─ 2. 混合检索 ────────────┐   ┌─ CS 政策作答 answer_cs() ───────┐
│  section_affinity()      │   │  gen_cs_answer / gen_cs_turns   │
│  0.72·向量 + 0.28·词面    │   │  多轮逐问镜像；客服口吻；无图    │
│  − 目录0.15 − 引言0.08    │   └─────────────────────────────────┘
└──────────┬───────────────┘
           ▼  3. 检索仲裁 adjudicate()：模型看 top-k 选题源节；"都不是"则弃权 → 置信 high/mid/low
           ▼
┌─ 4. 忠实二因子作答  answer_manual() ───────────────────────────┐
│   high：命中节逐字 / 微节标签直出；  mid：锚 ±1 窗成文；          │
│   low ：宽证据（整册≤14K 或 top3 节）成文。                       │
│   一律「仅依据题源章节、首句直答、覆盖相关步骤/参数/注意、图内联」  │
└──────────┬─────────────────────────────────────────────────────┘
           ▼  5. 幻觉抑制/硬校验：<PIC>数==图数 · 图须真实 · 语言一致 · 禁用语 · 多轮镜像；不达标→去图保文/兜底
           ▼
       答案  "正文(含<PIC>)"[, ["img1","img2",...]]
```

---

## 四、核心设计

1. **证据接地·二因子忠实生成（兼任幻觉抑制）** — `agent.answer_manual`
   答案严格接地到检索命中的**单一题源章节**，按章节长度分档作答（微节逐字 / 常规成文 / 低置信宽证据）。
   "只依据题源节、不引入节外信息" 是天然的幻觉抑制，也是答案质量的主线。

2. **混合检索 + 模型仲裁弃权门** — `agent.section_affinity` / `adjudicate`
   稠密向量(0.72) + 词面 TF-IDF(0.28) − 目录/引言陷阱降权；top-k 交模型仲裁，"都不是"则弃权降级到宽证据。

3. **单调 DP「一节多题」对齐** — 产物 `knowledge_base/slice_map.jsonl`
   同一手册的多道题在文档中按章节顺序单调出现，用动态规划做全局最优题→节指派，以预构建图谱形式提供。

4. **多约束硬校验 + 可解释溯源** — `agent.answer_question_ex` / `api/app.py`
   `<PIC>` 计数、图存在性、语言一致、多轮镜像、禁用语逐项把关；每次回答返回章节、置信度、策略与接地校验信息，便于追溯答案依据。

辅助能力：多模态感知融合、产品本体、分层会话记忆、指代消解、事后接地校验、跨语言图文对齐，均在代码或知识库中落地。

---

## 五、运行方式

| 目的 | 命令 |
|---|---|
| 单题（含路由/检索/策略元信息） | `python src/agent.py one "你的问题"` |
| 批量跑数据集 | `python src/agent.py batch --workers 24` → `result/answers_submit_api.csv` |
| 启动 API 服务 | `cd api && uvicorn app:app --host 0.0.0.0 --port 8000` |
| 离线质量评测 | `python validate.py`（需准备知识库数据） |

**生成后端可切换**（环境变量）：`GEN_BACKEND=deepseek`（默认，走 `.env` 的 OpenAI 兼容网关）｜ `anthropic`（Anthropic SDK，需 `pip install anthropic` + `GEN_MODEL_ANTHROPIC`）。
`KB_CACHE=1` 启用已验证答案直答（加速）；`USE_ID_PRIOR=1` 启用题号先验（仅诊断用）。

### `/chat` 调用示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <KAFU_API_TOKEN>" \
  -d '{"question":"如何给空调遥控器安装电池？","session_id":"demo_1"}'
```
```jsonc
{"code":0,"msg":"success",
 "data":{"answer":"安装电池 使用遥控器前，请先安装电池…<PIC>…",
         "session_id":"demo_1","timestamp":1741008000,
         "meta":{"type":"manual","product_key":"zh_空调","conf":"high","policy":"compose_high"}}}
```

---

## 六、多模态说明

- **手册侧多模态（已落地）**：知识库为图文手册，每个 `<PIC>` 占位与一个真实插图 `image_id`（=`knowledge_base/images/` 下文件名 stem）绑定；答案按"图内联紧邻其说明步骤"产出，并经硬校验保证图文计数一致、图真实存在。
- **用户侧多模态（接口已就位）**：`/chat` 接收 `images`（Base64，0-3 张）。当前核心检索以问题文本为主键；如需将用户上传图纳入意图理解，可在 `.env` 配置 `VISION_*`，由 `perception.perceive` 把图转为关键事实拼接到问题文本（架构已预留该位）。

---

## 七、知识库

知识库数据放在 `knowledge_base/`，**因体积与版权不随仓库分发**（见 `knowledge_base/README.md`）。代码所需的文件与作用：

| 文件 | 作用 |
|---|---|
| `sections_raw.jsonl` | **检索主库**：手册按 `#` 标题切出的章节（含 `<PIC id>` 占位、is_toc 标记） |
| `slice_map.jsonl` | 题→节 对齐图谱（单调 DP + 仲裁） |
| `catalog_locked.json` | 产品目录：产品名 / 别名 / 语言（路由用） |
| `manuals.jsonl` | 手册全文（清洗后） |
| `customer_policy_v2.jsonl` | 售前售后客服政策库（CS 作答用） |
| `image_descriptions.jsonl` | 插图文字描述（跨语图文绑定 / 多模态检索） |
| `images/*.png` | 说明书插图，文件名 stem 即 `image_id` |
| `question_spec.jsonl` | 问题规格（类型 / 产品 / 子问题 / 计数约束） |

---

## 八、部署与路径

- 代码以 `ROOT = src/..`（即**仓库根目录**）为基准定位数据：`ROOT/knowledge_base/...`、`ROOT/result/...`、`ROOT/cache/`、`ROOT/.env`。
  **请保持目录结构不变**；在任意路径放置后从根目录运行即可，无需改代码。
- `.env` 放在**根目录**（与 `README.md` 同级），由 `runtime.load_env()` 直接读取（不读系统环境变量）。
- 隔离环境复现：`pip install -r requirements.txt` → 填 `.env` → 准备 `knowledge_base/` 数据 → `python src/agent.py one ...` 或 `cd api && uvicorn app:app ...`。
- `cache/` 为运行时自动生成的磁盘缓存，可随时删除（仅影响重复调用的速度与费用）。
- 完整部署（含 Docker / 反向代理 / HTTPS）见 `api/部署说明.md`。

---

## 九、核心模块

| 文件 | 职责 |
|---|---|
| `src/agent.py` | 智能体主体：`Assets` 知识库单例、`section_affinity` 混合检索、`adjudicate` 仲裁、`answer_manual` 二因子作答、`answer_cs`、`answer_question_ex` 入口+硬校验、CLI `one`/`batch` |
| `src/runtime.py` | 运行时核心库：`.env` 读取、`llm_call`/`embed_call`/`vision_call`、磁盘缓存、客服作答、格式装配、图片清单 |
| `src/model_client.py` | 仲裁、指代消解、事后接地校验共用的 Qwen/OpenAI 兼容轻量客户端 |
| `src/perception.py` | 多模态感知融合：`perceive(question, images)` —— 视觉模型把用户上传图理解为关键事实，融合进检索查询 |
| `src/memory.py` | 分层对话记忆 + 指代消解：`resolve(question, session)` 把追问改写为独立查询；session 多轮存取 |
| `src/guard.py` | 事后接地校验（第三重抗幻觉）：`verify_grounding(answer, evidence)` 逐句核验、返回接地率与未接地句 |

---

## 十、许可证与数据

- 本仓库为作者实现的源代码。知识库中的产品说明书与插图为第三方材料，不包含在本仓库内，请在获得相应授权后自行准备并放入 `knowledge_base/`。
