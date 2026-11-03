# -*- coding: utf-8 -*-
"""
记账业务逻辑层。

职责：
1. CRUD：创建/查询/更新/删除账目
2. 统计：月度收支、类别分布
3. 自然语言记账：编排"意图解析 → 入库 → 索引"
4. RAG 搜索：编排"混合检索 → 格式化返回"

不写在这里的东西：
- HTTP 请求/响应（那是 routes/）
- LLM 调用细节（那是 ai/）
- 数据库 CRUD 语法糖（那是 models/）

好处：
- 单元测试只测这一层，不启动 FastAPI
- 未来换 MCP 服务端 / CLI / 微信机器人，直接复用
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Literal, Optional

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from ai.batch_intent import batch_parse_intent
from ai.intent import parse_intent
from ai.query_filter import extract_date_filter
from ai.rag import hybrid_search, index_entries, index_entry, remove_entry
from models.db import BookkeepingEntryORM
from models.schemas import (
    BatchNaturalLanguageResponse,
    BookkeepingEntryCreate,
    BookkeepingEntryResponse,
    BookkeepingEntryUpdate,
    DateFilterInfo,
    EntryListResponse,
    IntentResult,
    MonthlyStats,
    NaturalLanguageEntryResponse,
    RAGResultItem,
    RAGSearchResponse,
    RAGSummary,
)

logger = logging.getLogger(__name__)


# ============================================================
# 基础 CRUD
# ============================================================

def create_entry(db: Session, entry: BookkeepingEntryCreate) -> BookkeepingEntryResponse:
    """创建一条账目，同时异步建立向量索引。"""
    orm = BookkeepingEntryORM(
        date=entry.date,
        type=entry.type,
        amount=entry.amount,
        category=entry.category,
        note=entry.note,
        source=entry.source,
    )
    db.add(orm)
    db.flush()  # 拿到 id 但不开启新事务
    db.commit()
    db.refresh(orm)

    # 建立向量索引（失败不影响主流程）
    _index_after_write(orm)

    return _orm_to_response(orm)


def list_entries(
    db: Session,
    *,
    month_start: Optional[date] = None,
    month_end: Optional[date] = None,
    category: Optional[str] = None,
    entry_type: Optional[Literal["income", "expense"]] = None,
    limit: int = 100,
    offset: int = 0,
) -> EntryListResponse:
    """按条件查询账目列表。"""
    filters = []
    if month_start:
        filters.append(BookkeepingEntryORM.date >= month_start)
    if month_end:
        filters.append(BookkeepingEntryORM.date < month_end)
    if category:
        filters.append(BookkeepingEntryORM.category == category)
    if entry_type:
        filters.append(BookkeepingEntryORM.type == entry_type)

    query = select(BookkeepingEntryORM)
    if filters:
        query = query.where(and_(*filters))
    query = query.order_by(BookkeepingEntryORM.date.desc(), BookkeepingEntryORM.id.desc())
    query = query.offset(offset).limit(limit)

    entries = list(db.scalars(query).all())
    return EntryListResponse(
        total=len(entries),
        entries=[_orm_to_response(e) for e in entries],
    )


def get_entry(db: Session, entry_id: int) -> Optional[BookkeepingEntryResponse]:
    """按 ID 查询单条。"""
    orm = db.get(BookkeepingEntryORM, entry_id)
    return _orm_to_response(orm) if orm else None


def update_entry(
    db: Session, entry_id: int, update: BookkeepingEntryUpdate
) -> Optional[BookkeepingEntryResponse]:
    """局部更新。None 字段不改。"""
    orm = db.get(BookkeepingEntryORM, entry_id)
    if not orm:
        return None

    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(orm, field, value)

    db.commit()
    db.refresh(orm)
    return _orm_to_response(orm)


def delete_entry(db: Session, entry_id: int) -> bool:
    """删除。返回是否成功。"""
    orm = db.get(BookkeepingEntryORM, entry_id)
    if not orm:
        return False
    db.delete(orm)
    db.commit()
    # 删除向量索引
    remove_entry(entry_id)
    return True


# ============================================================
# 自然语言记账（AI 编排层）
# ============================================================

def record_from_natural_language(
    db: Session, text: str
) -> NaturalLanguageEntryResponse:
    """自然语言记账入口。

    流程：
    1. 调 ai.intent.parse_intent 解析成 IntentResult
    2. 失败 → 返回错误响应，不入库
    3. 成功 → 转成 BookkeepingEntryCreate，入库
    4. 入库后建向量索引，方便后续 RAG 搜索
    """
    intent_result: IntentResult = parse_intent(text)

    if not intent_result.success or intent_result.data is None:
        return NaturalLanguageEntryResponse(
            intent=intent_result,
            entry=None,
            stored=False,
        )

    data = intent_result.data
    if data.action != "record":
        # 不是记账动作（可能是 query/delete），返回意图让上层决定
        return NaturalLanguageEntryResponse(
            intent=intent_result,
            entry=None,
            stored=False,
        )

    entry_create = BookkeepingEntryCreate(
        date=data.date or date.today(),
        amount=data.amount or 0.0,
        type=data.type or "expense",
        category=data.category or "其他支出",
        note=data.note,
        source="nl",  # 标记是 AI 解析的
    )

    orm = create_entry(db, entry_create)
    return NaturalLanguageEntryResponse(
        intent=intent_result,
        entry=orm,
        stored=True,
    )


def record_batch_from_natural_language(
    db: Session, text: str
) -> BatchNaturalLanguageResponse:
    """批量自然语言记账：一段文本 → 多笔账。

    场景：用户晚上复盘一天
    输入：'早饭 15 中午外卖 30 打车 22 晚饭 45'
    返回：4 笔账都入库，同时告诉用户 AI 识别了几笔、实际存了几笔。
    """
    intents, source, error = batch_parse_intent(text)

    if error:
        return BatchNaturalLanguageResponse(
            parsed_count=0,
            stored_count=0,
            entries=[],
            source="error",
            error=error,
        )

    stored_entries: list[BookkeepingEntryResponse] = []
    for intent in intents:
        if intent.action != "record" or not intent.amount or not intent.type:
            continue
        entry_create = BookkeepingEntryCreate(
            date=intent.date or date.today(),
            amount=intent.amount,
            type=intent.type,
            category=intent.category or "其他支出",
            note=intent.note,
            source="nl",
        )
        orm = create_entry(db, entry_create)
        stored_entries.append(orm)

    return BatchNaturalLanguageResponse(
        parsed_count=len(intents),
        stored_count=len(stored_entries),
        entries=stored_entries,
        source=source,
    )


# ============================================================
# 月度统计
# ============================================================

def monthly_statistics(db: Session, year: int, month: int) -> MonthlyStats:
    """月度收支统计。前端饼图/柱状图数据源。"""
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)

    entries = list(
        db.scalars(
            select(BookkeepingEntryORM).where(
                and_(BookkeepingEntryORM.date >= start, BookkeepingEntryORM.date < end)
            )
        ).all()
    )

    total_income = sum(e.amount for e in entries if e.type == "income")
    total_expense = sum(e.amount for e in entries if e.type == "expense")

    # 按类别汇总
    by_category: Counter[str] = Counter()
    for e in entries:
        by_category[e.category] += e.amount

    # 按 type + category 汇总（前端画"收入分布"和"支出分布"两张饼图）
    by_type_category: dict[str, defaultdict[float]] = {"income": defaultdict(float), "expense": defaultdict(float)}
    for e in entries:
        by_type_category[e.type][e.category] += e.amount

    return MonthlyStats(
        year=year,
        month=month,
        total_income=round(total_income, 2),
        total_expense=round(total_expense, 2),
        balance=round(total_income - total_expense, 2),
        entry_count=len(entries),
        by_category={k: round(v, 2) for k, v in by_category.items()},
        by_type_category={k: {kk: round(vv, 2) for kk, vv in v.items()}
                          for k, v in by_type_category.items()},
    )


def last_n_days(db: Session, n: int = 7) -> EntryListResponse:
    """最近 n 天账目，前端首页展示用。"""
    today = date.today()
    start = today - timedelta(days=n)
    return list_entries(db, month_start=start, limit=50)


# ============================================================
# RAG 搜索
# ============================================================

async def search_by_query(
    db: Session, query: str, top_k: int = 5
) -> RAGSearchResponse:
    """RAG 搜索：带日期过滤的混合检索。

    流程：
    1. 从查询文本抽时间过滤器（"10月17号"、"昨天"、"上周"）
    2. 全表转 dict，应用过滤缩小候选
    3. 在过滤后的候选集上跑混合检索
    4. 计算过滤后的收支汇总（总收入/支出/笔数），前端直接展示

    返回结构：
    - mode: 实际命中的检索策略（vector / keyword / hybrid / empty / filtered）
    - results: top-k 结果，每条带 entry_id / date / category / amount / note / score
    - filter: 用到的时间过滤器（用户能看到"系统理解到的日期"）
    - summary: 过滤后的汇总（当有过滤器时才填）
    - total_entries_scanned: 未过滤的总账目数（透明度）
    """
    all_entries = db.scalars(select(BookkeepingEntryORM)).all()
    total_scanned = len(all_entries)

    if not all_entries:
        return RAGSearchResponse(
            query=query,
            mode="empty",
            results=[],
            total_entries_scanned=0,
        )

    # 抽时间过滤器
    date_filter = extract_date_filter(query)

    # ORM 转 dict 供 rag.hybrid_search 使用
    all_as_dicts = [
        {
            "id": e.id,
            "date": e.date.isoformat() if e.date else "",
            "type": e.type,
            "amount": e.amount,
            "category": e.category,
            "note": e.note,
        }
        for e in all_entries
    ]

    # 应用日期过滤
    if date_filter and date_filter.has:
        entries_as_dicts = _apply_date_filter(all_as_dicts, date_filter)
    else:
        entries_as_dicts = all_as_dicts

    if not entries_as_dicts:
        # 日期过滤后为空：还是返回结构，让用户知道"这一天没有账"
        return RAGSearchResponse(
            query=query,
            mode="filtered" if date_filter else "empty",
            results=[],
            total_entries_scanned=total_scanned,
            filter=_filter_info(date_filter) if date_filter else None,
            summary=_summary_of([]) if date_filter else None,
        )

    mode, raw_results = await hybrid_search(query, entries_as_dicts, k=top_k)

    results = [
        RAGResultItem(
            entry_id=r["entry_id"],
            date=r.get("date", ""),
            category=r.get("category", ""),
            amount=float(r.get("amount", 0)),
            note=r.get("note"),
            score=r["score"],
        )
        for r in raw_results
    ]

    # 有过滤器时，把 mode 改成 "filtered"，让用户知道系统在过滤后检索
    response_mode: Literal["vector", "keyword", "hybrid", "empty", "filtered"]
    if date_filter and date_filter.has:
        response_mode = "filtered"
    else:
        response_mode = mode

    return RAGSearchResponse(
        query=query,
        mode=response_mode,
        results=results,
        total_entries_scanned=total_scanned,
        filter=_filter_info(date_filter) if date_filter else None,
        summary=_summary_of(entries_as_dicts) if date_filter else None,
    )


def _apply_date_filter(
    entries: list[dict], date_filter
) -> list[dict]:
    """把日期过滤器应用到 entries 列表，返回命中的子集。"""
    from datetime import date as _d

    filtered = []
    for e in entries:
        raw = e.get("date")
        if not raw:
            continue
        try:
            d = _d.fromisoformat(str(raw))
        except ValueError:
            continue
        if date_filter.matches(d):
            filtered.append(e)
    return filtered


def _filter_info(date_filter) -> Optional[DateFilterInfo]:
    """把内部 DateFilter 转成 API 响应结构。"""
    if date_filter is None:
        return None
    return DateFilterInfo(
        date_eq=date_filter.date_eq.isoformat() if date_filter.date_eq else None,
        start=date_filter.start.isoformat() if date_filter.start else None,
        end=date_filter.end.isoformat() if date_filter.end else None,
        raw_text=date_filter.raw_text,
    )


def _summary_of(entries: list[dict]) -> Optional[RAGSummary]:
    """把 entries 聚合成收支汇总。"""
    if not entries:
        return RAGSummary(date_range="", entry_count=0)

    dates = [str(e.get("date", "")) for e in entries if e.get("date")]
    if dates:
        date_range = f"{min(dates)} ~ {max(dates)}" if len(dates) > 1 else dates[0]
    else:
        date_range = ""

    total_income = sum(e["amount"] for e in entries if e.get("type") == "income")
    total_expense = sum(e["amount"] for e in entries if e.get("type") == "expense")
    return RAGSummary(
        date_range=date_range,
        total_income=round(total_income, 2),
        total_expense=round(total_expense, 2),
        balance=round(total_income - total_expense, 2),
        entry_count=len(entries),
    )


# ============================================================
# 索引重建（管理工具）
# ============================================================

def rebuild_index(db: Session) -> int:
    """清空向量库并重建。用于：
    - 换 embedding 模型后
    - 索引损坏后
    - 手动触发
    """
    from ai.rag import clear_index

    clear_index()
    all_entries = db.scalars(select(BookkeepingEntryORM)).all()
    if not all_entries:
        return 0

    entries_as_dicts = [
        {
            "id": e.id,
            "date": e.date.isoformat() if e.date else "",
            "type": e.type,
            "amount": e.amount,
            "category": e.category,
            "note": e.note,
        }
        for e in all_entries
    ]
    # 同步调用（rebuild 通常在后台任务里跑）
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(index_entries(entries_as_dicts))


# ============================================================
# 内部工具
# ============================================================

def _orm_to_response(orm: BookkeepingEntryORM) -> BookkeepingEntryResponse:
    """ORM → Pydantic Response。"""
    return BookkeepingEntryResponse.model_validate(orm)


def _index_after_write(orm: BookkeepingEntryORM) -> None:
    """写入账目后建立向量索引（尽力而为，失败打日志不阻塞）。"""
    import asyncio

    entry_dict = {
        "id": orm.id,
        "date": orm.date.isoformat() if orm.date else "",
        "type": orm.type,
        "amount": orm.amount,
        "category": orm.category,
        "note": orm.note,
    }

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已经在 async 环境里，创建 task 但不阻塞
            asyncio.create_task(index_entry(entry_dict))
        else:
            loop.run_until_complete(index_entry(entry_dict))
    except RuntimeError:
        # 没有事件循环，直接跳过
        logger.debug("无事件循环，跳过异步索引")
