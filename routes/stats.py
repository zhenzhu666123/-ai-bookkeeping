# -*- coding: utf-8 -*-
"""
统计 API 路由。

前端图表数据源：
- 饼图 → by_type_category.expense（支出分布）
- 柱状图 → 按月 total_expense 对比
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ai.diagnosis import diagnose_spending
from database import get_db
from models.schemas import MonthlyStats
from services.bookkeeping import monthly_statistics

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/monthly", response_model=MonthlyStats)
def monthly(year: int, month: int,
            db: Session = Depends(get_db)) -> MonthlyStats:
    """指定年月的收支统计。

    参数：?year=2026&month=3 → 2026 年 3 月
    返回：总收入、总支出、结余、按类别分布
    """
    if not (1 <= month <= 12):
        raise ValueError(f"month 必须是 1-12，收到 {month}")
    return monthly_statistics(db, year=year, month=month)


@router.get("/diagnose")
def diagnose(
    days: int = Query(default=30, ge=7, le=365,
                      description="分析的时间跨度（天）。query 里没写时间时用这个"),
    query: Optional[str] = Query(default=None, max_length=200,
                                 description="自然语言问题。传入则从问题里抽时间/类目，做定向分析"),
    category: Optional[str] = Query(default=None, max_length=30,
                                    description="强制聚焦的类目，优先级高于从 query 抽的"),
    db: Session = Depends(get_db),
) -> dict:
    """个性化消费诊断 / 定向账本分析。

    两种模式：
    - 通用诊断：GET /api/stats/diagnose?days=30
      → LLM 看 30 天账本，找隐形浪费（"外卖占 75%"、"打车 12 次"）
    - 定向分析：GET /api/stats/diagnose?query=最近一周花销最多的是什么
      → 从问题里抽时间（最近一周）+ 类目（无），让 LLM 直接回答

    返回：
    {
        "period": "2026-09-17 ~ 2026-10-17",
        "days": 30,
        "query": "用户问题原文" | null,
        "category": "聚焦的类目" | null,
        "summary": {"total_income", "total_expense", "balance", "entry_count", "avg_daily_expense"},
        "diagnosis": "LLM 生成的文本",
        "diagnosis_source": "llm" | "fallback",
        "top_categories": [{"category", "amount", "count"}, ...]
    }
    """
    return diagnose_spending(db, days=days, query=query, category=category)
