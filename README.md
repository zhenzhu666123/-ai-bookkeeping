# 家庭记账系统（AI Native）

> 帮我爸妈做的一份**能说话就能记账**的家庭记账系统。

## 📖 项目背景

爸妈的记账方式很传统：要么用纸笔记（记两天就懒得记了），要么翻微信支付宝账单（一堆消费混在一起，看不懂到底花哪了）。

我做了一个**能说话就能记账**的系统：
- 打开网页，输入 "午饭 15"，回车，就记上了
- 问 "上次猪肉多少钱"，AI 从历史账里翻出来
- 每个月末，饼图告诉我钱花哪儿了

## 🚀 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 配置（可选：不配置也能跑，只是 AI 走规则兜底）
cp .env.example .env
# 编辑 .env 填入你自己的 LLM_API_KEY

# 3. 启动
python main.py
# 或者
uvicorn main:app --reload

# 4. 打开浏览器
# 手机: http://你的IP:8000
# 电脑: http://localhost:8000
```

首次启动会自动建表 + 灌入 60 条示例数据，可以直接看到效果。

### 支持的模型

代码走的是 **OpenAI 兼容协议**，只要模型支持标准 `/v1/chat/completions` 接口就能用：

| 模型 | base_url | model | 备注 |
|---|---|---|---|
| **SenseNova** | `https://token.sensenova.cn/v1` | `sensenova-6.8-flash-lite` | 推理模型，响应慢（8-50s） |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` | 便宜，中文好 |
| **通义千问** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` | 阿里 |
| **智谱 GLM** | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` | 有免费额度 |
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4o-mini` | 需翻墙 |
| **Ollama（本地）** | `http://localhost:11434/v1` | `qwen2.5:7b` | 离线跑 |

**改配置一行切换**：只改 `LLM_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL` 三个字段。

## 🎯 核心功能

### 1. 自然语言记账

爸妈不需要学复杂操作，一句话就能记：

| 你说 | 系统理解 |
|---|---|
| "午饭 15" | 支出 15 元 / 吃饭 / 今天 |
| "打车 20 块" | 支出 20 元 / 交通 / 今天 |
| "工资 8000 到账" | 收入 8000 元 / 工资 / 今天 |
| "昨天买菜 88" | 支出 88 元 / 购物 / 昨天 |
| "交电费 150" | 支出 150 元 / 水电 / 今天 |

背后是**意图路由**：把自然语言解析成结构化 JSON，然后入库。

### 2. RAG 智能检索

问历史账目：

- "上次猪肉多少钱"
- "上周打车花了多少"
- "这个月水电交了多少"

背后是**混合检索**：向量语义 + 关键词兜底，双路互补。

### 3. 月度统计

饼图/柱状图看钱花哪儿了：
- 总收入 / 总支出 / 结余
- 按类别分布（吃饭、交通、购物...）
- 按收支类型分开统计

## 🏗 技术栈

| 层 | 技术 | 选型理由 |
|---|---|---|
| Web 框架 | FastAPI | 类型标注 + 异步 + 自动生成 OpenAPI 文档 |
| ORM | SQLAlchemy 2.0 | `DB_URL` 一行切换 SQLite/MySQL |
| 数据校验 | Pydantic v2 | `field_validator` 做业务约束 |
| 配置 | pydantic-settings | 三级优先级（环境变量 > .env > 默认值）|
| HTTP | httpx | 直接调 OpenAI 兼容 API，透明可控 |
| 测试 | pytest | 无 LLM 依赖也能跑通核心逻辑 |
| 前端 | 原生 HTML+JS | 零构建，手机友好，爸妈直接打开 |

## 🧠 技术亮点

### 亮点 1：AI 挂了也能记账（规则兜底）

大多数 AI 项目遇到 LLM API 超时/Key 用完就直接报错。
我做了**五层保护**：

```
Layer 1: Prompt 里明确"必须返回严格 JSON"
Layer 2: response_format={"type":"json_object"} 强制结构化
Layer 3: json.loads 严格解析，失败立刻重试
Layer 4: Pydantic 校验字段，失败重试（最多 3 次）
Layer 5: 全部失败 → 走正则规则兜底（覆盖 80% 场景）
```

即使 LLM API 完全挂了，"午饭 15"这种常见场景规则能命中，**记账功能依然可用**。

### 亮点 2：RAG 双路检索（不是黑箱）

传统 RAG 是"把问题 embedding → 找相似向量 → 塞回 prompt"。
我的做法是**向量 + 关键词双路互补**：

1. 先跑向量检索（语义相近）
2. 分数都低于阈值 → 判定向量不靠谱，走关键词
3. 分数够高 + 关键词也命中 → 混合排序（0.7 向量 + 0.3 关键词）

**为什么关键词兜底？** 家庭记账场景，"猪肉"两个字几乎一定会出现在原文里，字面匹配比语义相似更准。

### 亮点 3：透明可解释

每次 AI 记账，API 返回：
```json
{
  "intent": {
    "source": "llm",       // 或 "fallback"
    "confidence": 0.85,    // 置信度
    "raw_llm_output": "..."
  },
  "stored": true
}
```

用户能看到：
- 这次是 LLM 解析的还是规则兜底的？
- 置信度多少？
- 具体记了什么？

**出问题一眼定位**，不是黑箱。

### 亮点 4：三级配置（12 要素应用）

```
优先级：环境变量 > .env 文件 > 代码默认值
```

- 本地开发：改 `.env`
- 团队共享：`.env` 不提交 git，每人填自己的 API Key
- 生产部署：环境变量注入，代码零修改
- 新人上手：默认值兜底，`git clone` → `pip install` → `python main.py` 就能跑

**换数据库 / 换模型 / 改超时都是改一行配置**，代码不动。

## 📁 项目结构

```
ai_bookkeeping/
├── main.py                    # FastAPI 入口 + 静态资源
├── config.py                  # 三级配置（环境变量/.env/默认值）
├── database.py                # DB_URL 切换 SQLite/MySQL
├── requirements.txt           # 依赖
├── .env.example               # 环境变量示例
│
├── models/                    # 数据模型
│   ├── db.py                  # SQLAlchemy ORM（表结构）
│   └── schemas.py             # Pydantic（请求/响应）
│
├── services/                  # 业务逻辑
│   └── bookkeeping.py         # CRUD + 统计 + RAG 编排
│
├── ai/                        # AI 相关
│   ├── llm.py                 # OpenAI 兼容 API 客户端
│   ├── intent.py              # 意图解析（5 层 JSON 正确性）
│   ├── fallback.py            # 规则兜底（正则+关键词）
│   ├── vector_store.py        # 内存向量库 + 余弦相似度
│   └── rag.py                 # 混合检索编排
│
├── routes/                    # FastAPI 路由
│   ├── bookkeeping.py         # 记账 CRUD + 自然语言
│   ├── stats.py               # 月度统计
│   └── rag.py                 # RAG 搜索
│
├── static/                    # 前端
│   └── index.html             # 手机友好的记账页面
│
├── tests/                     # pytest 测试
│   ├── test_api.py            # API 端点（不依赖 LLM）
│   ├── test_fallback.py       # 规则兜底测试
│   └── test_vector_store.py   # 向量库数学测试
│
└── data/
    └── seed_data.py           # 首次启动灌入示例数据
```

## 🧪 运行测试

```bash
pytest tests/ -v
```

测试覆盖：
- 规则兜底：金额/类别/收支类型/日期识别
- 向量库：余弦相似度数学正确性
- API：CRUD、参数校验、自然语言记账（走规则兜底）、月度统计、RAG 关键词检索

**测试不需要配置 LLM API Key**，验证的是核心业务逻辑。

## 🚧 面试问答（3 个高频问题）

### Q1: AI 生成的结果怎么保证是正确的 JSON？

**5 层保障**，任何一层失败自动降级：

1. **Prompt 层**：system prompt 里明确"必须返回严格 JSON，不要 markdown"，加 few-shot 示例
2. **response_format 层**：调 API 时传 `response_format={"type":"json_object"}`，服务端保证返回合法 JSON
3. **解析层**：`json.loads` 严格解析，兼容 markdown 包裹（```json ... ```），失败立即重试
4. **Schema 层**：Pydantic 严格校验字段类型、必填、枚举值，失败重试（`settings.llm_max_retries` 次）
5. **兜底层**：LLM 全部失败 → 走正则规则（`ai/fallback.py`）覆盖 80% 常见场景

**代码位置**：`ai/intent.py`

### Q2: RAG 怎么确保检索到对的知识？

**4 层保障**：

1. **同源 Embedding**：问题和文档用同一个 embedding 模型，避免向量空间不对齐
2. **余弦相似度**：只看方向忽略量纲，归一化后点积 = 余弦值
3. **Top-K + 阈值**：只返回 top-k，分数低于阈值不返回（避免 prompt 塞爆）
4. **关键词兜底**：向量分数都低时自动降级到关键词命中（`ai/rag.py` 里 0.7 向量 + 0.3 关键词混合）

**为什么关键词兜底有效**：家庭记账场景，"猪肉"两个字几乎一定出现在原文里，字面匹配比语义相似更准。

**代码位置**：`ai/rag.py` + `ai/vector_store.py`

### Q3: 简历很多地方都和环境变量挂钩，是为啥？

**遵循 12 要素应用原则**，三级配置优先级：

```
环境变量 > .env 文件 > 代码默认值
```

- **本地开发**：`.env` 里填自己的 API Key，不提交 git
- **团队共享**：每人 `.env` 各自填，代码共享
- **生产部署**：CI/CD 用环境变量注入，代码零修改
- **切换模型**：改 `LLM_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL` 三个字段就行
- **切换数据库**：改 `DB_URL` 一行（SQLite → MySQL）
- **新人上手**：默认值兜底，`git clone` → `pip install` → `python main.py` 零配置就能跑

**代码位置**：`config.py` 用 `pydantic-settings` 自动加载

## 📄 License

MIT
