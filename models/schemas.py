# -*- coding: utf-8 -*-
"""
Pydantic 模型：API 请求/响应的类型约束

分三类：
- Create：新建账目（LLM 解析结果也符合这个结构）
- Update：局部更新
- Response：返回给前端的完整账目

field_validator 做业务校验：金额必须正数、类别不能空、类型只能是 income/expense
"""
from __future__ import annotations

from datetime import date as _Date, datetime as _DateTime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================
# 记账条目（核心模型）
# ============================================================

class BookkeepingEntryCreate(BaseModel):
    """新建账目请求。

    也用作 LLM 意图解析的目标 schema：LLM 返回的 JSON 必须能塞进这个模型。
    """

    date: _Date = Field(default_factory=_Date.today, description="交易日期，默认今天")
    amount: float = Field(gt=0, description="金额必须大于 0")
    type: Literal["income", "expense"] = Field(description="收入或支出")
    category: str = Field(min_length=1, max_length=30, description="类别，例如：吃饭")
    note: Optional[str] = Field(default=None, max_length=200, description="可选备注")
    source: Literal["manual", "nl"] = Field(default="manual", description="来源标记")

    @field_validator("category")
    @classmethod
    def _strip_category(cls, v: str) -> str:
        """去掉首尾空格，避免 ' 吃饭 ' 和 '吃饭' 存成两条类别。"""
        stripped = v.strip()
        if not stripped:
            raise ValueError("类别不能为空字符串")
        return stripped

    @field_validator("amount")
    @classmethod
    def _round_amount(cls, v: float) -> float:
        """四舍五入到分。LLM 有时返回 15.000000001 这种浮点误差。"""
        return round(v, 2)


class BookkeepingEntryUpdate(BaseModel):
    """局部更新。所有字段可选，None 表示不改。"""

    amount: Optional[float] = Field(default=None, gt=0)
    category: Optional[str] = Field(default=None, min_length=1, max_length=30)
    note: Optional[str] = Field(default=None, max_length=200)
    date: Optional[_Date] = None
    type: Optional[Literal["income", "expense"]] = None


class BookkeepingEntryResponse(BaseModel):
    """返回给前端的完整账目。"""

    id: int
    date: _Date
    type: str
    amount: float
    category: str
    note: Optional[str]
    source: str
    created_at: _DateTime

    model_config = ConfigDict(from_attributes=True)  # 支持 ORM 对象直接转


# ============================================================
# AI 意图解析结果
# ============================================================

class ParsedIntent(BaseModel):
    """LLM 解析自然语言后的结构化意图。

    三层保障 JSON 正确性：
    1. Prompt 里明确"必须返回严格 JSON，不要 markdown"
    2. response_format={"type": "json_object"} 强制结构化输出
    3. Pydantic 校验 + 重试 + 规则兜底
    """

    action: Literal["record", "query", "delete", "unknown"] = "unknown"
    amount: Optional[float] = None
    type: Optional[Literal["income", "expense"]] = None
    category: Optional[str] = None
    date: Optional[_Date] = None
    note: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_record_fields(self) -> "ParsedIntent":
        """record 动作必须有 amount 和 type，避免残缺数据入库。"""
        if self.action == "record":
            if self.amount is None or self.type is None:
                raise ValueError("record 动作必须包含 amount 和 type")
        return self


class IntentResult(BaseModel):
    """意图解析包装：成功/失败 + 数据来源（llm/fallback）。"""

    success: bool
    data: Optional[ParsedIntent] = None
    source: Literal["llm", "fallback", "error"] = "error"
    error: Optional[str] = None
    raw_llm_output: Optional[str] = None  # 调试用，记录 LLM 原始输出


# ============================================================
# 查询和统计
# ============================================================

class EntryListResponse(BaseModel):
    """列表接口返回。"""

    total: int
    entries: list[BookkeepingEntryResponse]


class MonthlyStats(BaseModel):
    """月度收支统计。前端图表用这个数据画饼图/柱状图。"""

    year: int
    month: int
    total_income: float
    total_expense: float
    balance: float
    entry_count: int
    by_category: dict[str, float] = Field(default_factory=dict)
    by_type_category: dict[str, dict[str, float]] = Field(default_factory=dict)


class NaturalLanguageEntryRequest(BaseModel):
    """自然语言记账请求。"""

    text: str = Field(min_length=1, max_length=500, description="用户输入的自然语言")


class NaturalLanguageEntryResponse(BaseModel):
    """自然语言记账返回：原始解析结果 + 可选入库结果。"""

    intent: IntentResult
    entry: Optional[BookkeepingEntryResponse] = None
    stored: bool = False


# ============================================================
# 批量自然语言记账
# ============================================================

class BatchNaturalLanguageRequest(BaseModel):
    """批量记账请求：一段话里可能包含多笔账。

    用户场景：晚上复盘一天说"早饭 15 中午外卖 30 打车 20 晚饭 45"
    系统应该识别成 4 条独立账目。
    """

    text: str = Field(min_length=1, max_length=2000, description="一段可能包含多笔账的文本")


class BatchNaturalLanguageResponse(BaseModel):
    """批量记账返回。"""

    parsed_count: int          # AI 识别出几笔
    stored_count: int          # 实际入库几笔
    entries: list[BookkeepingEntryResponse]   # 入库成功的条目
    source: Literal["llm", "fallback", "error"] = "error"
    error: Optional[str] = None


# ============================================================
# 截图识别记账
# ============================================================

class ImageExtractResult(BaseModel):
    """视觉模型从图片里提取到的字段。"""

    amount: float = Field(gt=0, description="金额")
    type: Literal["income", "expense"] = Field(description="收支类型")
    category: str = Field(min_length=1, description="分类")
    note: Optional[str] = None
    date_hint: Optional[str] = Field(default=None, description="图片里显示的日期，YYYY-MM-DD")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "vision"


# ============================================================
# CSV 批量导入
# ============================================================

class CsvImportEntry(BaseModel):
    """CSV 解析出的一条账目。"""

    date: _Date
    amount: float
    type: str
    category: str
    note: Optional[str] = None
    row_index: int  # 原文件里的行号


class CsvImportError(BaseModel):
    """CSV 单行解析错误。"""

    row_index: int
    reason: str


class CsvImportResponse(BaseModel):
    """CSV 导入返回。"""

    total_rows: int                    # 文件里的总数据行数（不含表头）
    parsed_count: int                  # 成功解析的行数
    imported_count: int                # 成功入库的行数
    errors: list[CsvImportError]       # 解析失败的行及原因
    entries: list[BookkeepingEntryResponse]  # 入库成功的账目


class RAGSearchRequest(BaseModel):
    """RAG 价格搜索请求。"""

    query: str = Field(min_length=1, max_length=200, description="查询语句，如：上次猪肉多少钱")
    top_k: int = Field(default=5, ge=1, le=20)


class RAGResultItem(BaseModel):
    """RAG 检索结果条目。"""

    entry_id: int
    date: str
    category: str
    amount: float
    note: Optional[str]
    score: float  # 余弦相似度或关键词命中率


class DateFilterInfo(BaseModel):
    """查询中解析出的时间过滤器，让用户看到"你到底在过滤什么"。"""

    date_eq: Optional[str] = None   # 单日（ISO 字符串）
    start: Optional[str] = None     # 区间起点
    end: Optional[str] = None       # 区间终点
    raw_text: str = ""              # 用户原文里匹配到的日期片段


class RAGSummary(BaseModel):
    """按时间过滤后的聚合汇总。

    例如"10月17号花了多少钱"→ 返回当天的总支出、总收入、笔数。
    """

    date_range: str = ""            # 人话描述："2026-10-17" / "2026-10"
    total_income: float = 0.0
    total_expense: float = 0.0
    balance: float = 0.0
    entry_count: int = 0


class RAGSearchResponse(BaseModel):
    """RAG 检索返回。"""

    query: str
    mode: Literal["vector", "keyword", "hybrid", "empty", "filtered"]
    results: list[RAGResultItem]
    total_entries_scanned: int      # 扫描总条数（含未过滤）
    filter: Optional[DateFilterInfo] = None     # 应用的时间过滤器
    summary: Optional[RAGSummary] = None        # 过滤后的聚合汇总
