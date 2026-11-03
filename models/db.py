# -*- coding: utf-8 -*-
"""
SQLAlchemy ORM 模型：数据库里的表结构

只有一张核心表 bookkeeping_entries，但把字段设计得丰富一点：
- date：交易日期（不是 created_at，日期是账本身的一部分）
- type：income/expense 二分类
- amount：金额，一律存正数（type 区分收支）
- category：类别，前端图表按这个分组
- source：manual（手动）/ nl（AI 自然语言解析）
- note：可选备注
- created_at：审计字段，记录入库时间
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional


def _utcnow() -> datetime:
    """带时区的当前 UTC 时间（替代已废弃的 datetime.utcnow()）。"""
    return datetime.now(timezone.utc)

from sqlalchemy import DateTime, Date, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 ORM 模型的基类，metadata 挂在上面方便 init_db 一次性建表。"""


class BookkeepingEntryORM(Base):
    """账目主表。

    字段设计要点：
    - id 自增主键，跨设备同步时可作外部 ID
    - date 用 Date 类型（不含时分秒），跟记账语义对齐
    - amount 用 Float 够用（家庭记账精度到分，不用 Decimal）
    - type 用 String(10) 而不是 Enum，MySQL 兼容性好
    - category 用 String(30)，预留中英文扩展
    - source 记录这笔账是手动还是 AI 解析的，方便后面复盘 AI 准确率
    """

    __tablename__ = "bookkeeping_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    type: Mapped[str] = mapped_column(String(10), nullable=False)  # income / expense
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str] = mapped_column(String(30), index=True, nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(10), nullable=False, default="manual")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<BookkeepingEntry id={self.id} date={self.date} "
            f"type={self.type} amount={self.amount} category={self.category}>"
        )
