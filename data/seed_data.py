# -*- coding: utf-8 -*-
"""
示例数据：首次启动自动灌入，方便直接给爸妈演示。

数据特点：
1. 覆盖最近 3 个月，让月度统计有内容
2. 包含各种类别，饼图好看
3. 一部分是"nl"来源（AI 记账），一部分是"manual"（手动），
   让 source 字段的区分有实际意义

如果不想灌示例数据，把 config.py 里 seed_on_first_run 改成 False。
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import SessionLocal
from models.db import BookkeepingEntryORM


def _random_entries():
    """生成 60 条左右的随机但真实感的示例数据。"""
    today = date.today()
    random.seed(20260315)  # 固定种子，保证每次生成相同数据

    entries = []

    # 每月发一次工资（家庭记账的核心收入）
    for month_offset in range(3):
        salary_date = date(today.year - (month_offset // 3),
                           max(1, today.month - month_offset), 15)
        entries.append({
            "date": salary_date,
            "type": "income",
            "amount": 8500.00,
            "category": "工资",
            "note": "爸爸工资到账",
            "source": "manual",
        })

    # 每周买菜、买日用品
    for i in range(10):
        d = today - timedelta(days=i * 7 + random.randint(0, 3))
        if d > today:
            continue
        entries.append({
            "date": d,
            "type": "expense",
            "amount": round(random.uniform(60, 200), 2),
            "category": "购物",
            "note": f"第{i+1}次买菜",
            "source": random.choice(["nl", "manual"]),
        })

    # 日常吃饭（每天 1-2 笔）
    for i in range(30):
        d = today - timedelta(days=i)
        entries.append({
            "date": d,
            "type": "expense",
            "amount": round(random.uniform(15, 80), 2),
            "category": "吃饭",
            "note": random.choice([
                "午饭", "晚饭", "外卖", "食堂", "早饭",
                "和同事聚餐", "出去吃火锅", "买菜回来做饭",
            ]),
            "source": random.choice(["nl", "nl", "manual"]),  # 更多是 AI 记的
        })

    # 水电费（每月一次）
    for month_offset in range(2):
        d = date(today.year - (month_offset // 3),
                 max(1, today.month - month_offset), 5)
        entries.append({
            "date": d,
            "type": "expense",
            "amount": round(random.uniform(180, 320), 2),
            "category": "水电",
            "note": "水电燃气费",
            "source": "manual",
        })

    # 打车
    for i in range(6):
        d = today - timedelta(days=random.randint(1, 60))
        entries.append({
            "date": d,
            "type": "expense",
            "amount": round(random.uniform(15, 60), 2),
            "category": "交通",
            "note": random.choice(["打车去机场", "滴滴回家", "地铁", "加油"]),
            "source": random.choice(["nl", "manual"]),
        })

    # 红包（过年过节）
    entries.append({
        "date": today - timedelta(days=30),
        "type": "income",
        "amount": 200.00,
        "category": "奖金",
        "note": "过年红包",
        "source": "nl",
    })

    # 医药
    for i in range(3):
        d = today - timedelta(days=random.randint(20, 70))
        entries.append({
            "date": d,
            "type": "expense",
            "amount": round(random.uniform(30, 250), 2),
            "category": "医疗",
            "note": random.choice(["买感冒药", "医院挂号", "门诊"]),
            "source": "nl",
        })

    return entries


def seed_if_empty() -> int:
    """如果数据库为空，灌入示例数据。返回插入的条数。

    幂等：数据库不为空时什么都不做，方便重复启动。
    """
    db = SessionLocal()
    try:
        count = db.scalar(select(func.count()).select_from(BookkeepingEntryORM)) or 0
        if count > 0:
            print(f"[seed] 已有 {count} 条数据，跳过灌入")
            return 0

        entries = _random_entries()
        for e in entries:
            db.add(BookkeepingEntryORM(**e))

        db.commit()
        print(f"[seed] 灌入 {len(entries)} 条示例数据")
        return len(entries)
    finally:
        db.close()


if __name__ == "__main__":
    from database import init_db
    init_db()
    n = seed_if_empty()
    print(f"完成：{n} 条")
