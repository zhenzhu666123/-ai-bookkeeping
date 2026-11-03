# -*- coding: utf-8 -*-
"""
CSV/Excel 记账文件解析。

【支持的场景】
1. 用户从支付宝/微信/银行导出流水 CSV（有表头）
2. 用户手工整理 CSV（自定义列名，中文或英文都行）
3. 有 BOM 头、GBK 编码、UTF-8 编码都支持

【列名识别】
模糊匹配列名，允许这些别名：
- date: 日期、时间、date、时间日期、date_time、发生时间、创建日期
- amount: 金额、amt、amount、交易额、金额(元)、金额(人民币)、总金额、消费金额
- type: 类型、type、收支、收支类型、direction
- category: 分类、类别、category、category_name、大类、类目
- note: 备注、说明、note、comment、摘要、商户、交易描述、交易类型

【金额处理】
- ¥/￥/$ 前缀、, 千分位、括号（会计记账表）都支持
- 负数视为支出，正数默认支出（除非 type 列说 income）

【日期处理】
- 2026-10-19 / 2026/10/19 / 2026.10.19 / 20261019
- 2026-10-19 12:30:45（带时分秒）
- 10-19（省略年，用当前年）
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class ParsedEntry:
    """从 CSV 解析出的一条账目（还未入库）。"""
    date: Optional[date]
    amount: float
    type: str  # "income" | "expense"
    category: str
    note: Optional[str]
    row_index: int  # 原始行号（1-based），方便报错定位


@dataclass
class ParseError:
    """单行解析错误。"""
    row_index: int
    reason: str
    raw_row: dict


@dataclass
class ParseResult:
    """CSV 解析结果。"""
    entries: list[ParsedEntry]
    errors: list[ParseError]
    total_rows: int


# 列名别名映射（都是小写形式）
COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["日期", "时间", "date", "date_time", "日期时间", "发生时间", "交易日期", "创建日期"],
    "amount": ["金额", "amt", "amount", "交易额", "总金额", "消费金额", "金额(元)", "金额（元）", "money", "value"],
    "type": ["类型", "type", "收支", "收支类型", "direction", "in_out", "inout"],
    "category": ["分类", "类别", "category", "大类", "类目", "category_name"],
    "note": ["备注", "说明", "note", "comment", "摘要", "商户", "商户名", "交易描述", "交易类型", "交易方"],
}


def _normalize_header(h: str) -> str:
    """去掉 BOM、空格、换行。"""
    return h.replace("", "").strip()


def _match_column(headers: list[str], field: str) -> Optional[int]:
    """在表头里找匹配某字段的列索引。"""
    aliases = COLUMN_ALIASES[field]
    normalized = [_normalize_header(h).lower() for h in headers]
    for i, h in enumerate(headers):
        hn = normalized[i]
        for alias in aliases:
            if hn == alias.lower() or hn == alias:
                return i
    # 模糊匹配：header 里包含 alias
    for i, h in enumerate(headers):
        hn = normalized[i]
        for alias in aliases:
            if alias.lower() in hn or alias in hn:
                return i
    return None


def _parse_amount(raw: str) -> tuple[Optional[float], str]:
    """解析金额字符串，返回 (数值, 是否原为负数)。"""
    if raw is None:
        return None, "negative"
    s = str(raw).strip()
    if not s:
        return None, "positive"

    # 括号表法：(500) 表示 500 是负数（会计记账习惯）
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1]
    elif s.startswith("-") or s.startswith("－") or s.startswith("−"):
        is_negative = True
        s = s[1:]

    # 去掉货币符号和千分位
    s = re.sub(r"[¥￥$¥，,\s]", "", s)
    # 去掉尾部 "元" 单位
    s = s.replace("元", "")

    try:
        val = float(s)
    except ValueError:
        return None, "positive"

    if is_negative:
        return abs(val), "negative"
    return val, "positive"


def _parse_date(raw: str) -> Optional[date]:
    """解析多种日期格式。"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # 带时分秒：2026-10-19 12:30:45
    m = re.match(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            pass

    # 短格式：10-19（用当前年）
    m = re.match(r"^(\d{1,2})[-/.](\d{1,2})$", s)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        try:
            return date(date.today().year, mo, d)
        except ValueError:
            pass

    # 紧凑格式：20261019
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # 中文：10月19日
    m = re.match(r"(\d{1,2})月(\d{1,2})[日号]?", s)
    if m:
        try:
            return date(date.today().year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass

    return None


def _parse_type(raw: str, amount_negative: bool) -> str:
    """解析收支类型。"""
    if raw:
        s = str(raw).strip().lower()
        if any(k in s for k in ["income", "收入", "in", "1"]):
            return "income"
        if any(k in s for k in ["expense", "支出", "out", "0"]):
            return "expense"

    # 靠金额符号推断
    return "income" if amount_negative else "expense"


def _normalize_category(raw: Optional[str]) -> str:
    """把乱七八糟的分类名映射到标准类目。"""
    if not raw:
        return "其他"
    s = str(raw).strip()
    if not s:
        return "其他"

    # 标准类目直接返回
    standard = {"吃饭", "交通", "购物", "水电", "医疗", "教育", "人情", "其他", "工资", "奖金"}
    if s in standard:
        return s

    # 常见别名映射
    mapping = {
        # 吃饭相关
        "餐": "吃饭", "餐饮": "吃饭", "外卖": "吃饭", "food": "吃饭", "meal": "吃饭",
        "餐费": "吃饭", "早餐": "吃饭", "午餐": "吃饭", "晚餐": "吃饭",
        # 交通
        "出行": "交通", "打车": "交通", "taxi": "交通", "transport": "交通",
        "car": "交通", "bus": "交通", "地铁": "交通",
        # 购物
        "买": "购物", "超": "购物", "sho": "购物", "purchase": "购物",
        # 水电
        "水": "水电", "electric": "水电", "utility": "水电", "gas": "水电",
        # 医疗
        "医": "医疗", "药": "医疗", "hospital": "医疗", "medicine": "医疗",
        # 教育
        "学": "教育", "education": "教育",
        # 工资
        "薪水": "工资", "salary": "工资", "pay": "工资",
        "奖励": "奖金", "bonus": "奖金",
    }
    sl = s.lower()
    for k, v in mapping.items():
        if k in sl or k in s:
            return v
    return "其他"


def _encode_fallback(content: bytes) -> str:
    """尝试多种编码解码字节串。"""
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "big5"):
        try:
            return content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("utf-8", errors="replace")


def parse_csv_content(content: bytes) -> ParseResult:
    """解析 CSV 字节流为账目列表。

    Args:
        content: CSV 文件字节内容

    Returns:
        ParseResult 包含：
        - entries: 成功解析的账目列表
        - errors: 解析失败的行及原因
        - total_rows: 总数据行数（不含表头）
    """
    text = _encode_fallback(content)
    if not text.strip():
        return ParseResult(entries=[], errors=[], total_rows=0)

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 2:
        return ParseResult(entries=[], errors=[], total_rows=0)

    headers = [_normalize_header(h) for h in rows[0]]

    # 识别列索引
    idx_date = _match_column(headers, "date")
    idx_amount = _match_column(headers, "amount")
    idx_type = _match_column(headers, "type")
    idx_category = _match_column(headers, "category")
    idx_note = _match_column(headers, "note")

    if idx_amount is None:
        return ParseResult(
            entries=[],
            errors=[ParseError(row_index=0, reason=f"未找到金额列。表头：{headers}", raw_row={})],
            total_rows=len(rows) - 1,
        )

    entries: list[ParsedEntry] = []
    errors: list[ParseError] = []

    for i, row in enumerate(rows[1:], start=1):
        # 空行跳过
        if not row or all(not c.strip() for c in row):
            continue

        def safe_get(idx: Optional[int]) -> str:
            if idx is None or idx >= len(row):
                return ""
            return str(row[idx]).strip()

        raw_amount = safe_get(idx_amount)
        amount, neg_flag = _parse_amount(raw_amount)
        if amount is None:
            errors.append(ParseError(
                row_index=i + 1,  # +1 因为 1-based 且含表头
                reason=f"金额无法解析：'{raw_amount}'",
                raw_row=dict(zip(headers, row)),
            ))
            continue

        raw_date = safe_get(idx_date)
        dt = _parse_date(raw_date)

        raw_type = safe_get(idx_type)
        entry_type = _parse_type(raw_type, neg_flag == "negative")

        # 如果 type 列明确写了但金额是负数且被判为支出，仍相信 type
        raw_cat = safe_get(idx_category)
        category = _normalize_category(raw_cat)

        raw_note = safe_get(idx_note)
        note = raw_note if raw_note else None

        # 如果没日期，跳过（无法归档）
        if dt is None:
            errors.append(ParseError(
                row_index=i + 1,
                reason=f"日期无法解析：'{raw_date}'（允许：2026-10-19 / 2026/10/19 / 10-19）",
                raw_row=dict(zip(headers, row)),
            ))
            continue

        entries.append(ParsedEntry(
            date=dt,
            amount=round(amount, 2),
            type=entry_type,
            category=category,
            note=note,
            row_index=i + 1,
        ))

    return ParseResult(entries=entries, errors=errors, total_rows=len(rows) - 1)
