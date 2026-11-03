# -*- coding: utf-8 -*-
"""
批量意图解析：一段文本可能包含多笔账，拆成独立的 ParsedIntent 列表。

【场景】
用户晚上复盘一天会说："早饭 15 中午外卖 30 打车去公司 22 晚饭跟同事 45"
传统记账 App 只能一笔一笔录，很烦。这里让 LLM 一次拆完。

【实现】
用同一个 OpenAI 兼容协议、同一个 JSON 模式，
只是 prompt 改成让模型输出 entries 数组。

失败降级：LLM 挂了 → 走规则（对每个可能的句子单独跑一次 parse_intent）。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from typing import Optional

from pydantic import ValidationError

from ai.fallback import parse_intent_by_rules
from ai.intent import (
    CATEGORY_LIST,
    _llm_available,
    _validate_intent_dict,
    parse_json_safely,
    parse_intent,
)
from ai.llm import LLMError, chat_completion
from config import settings
from models.schemas import IntentResult, ParsedIntent

logger = logging.getLogger(__name__)


BATCH_SYSTEM_PROMPT = """你是家庭记账助手的**批量**意图解析器。
用户可能会一次性描述多笔账目（比如晚上复盘一天），
你要把它们拆成独立的 JSON 数组元素。

【当前日期】
今天是 {today}。用户说"今天"、"昨天"、"前天"时按此推算。

【输出格式】
必须返回一个 JSON 对象（不是数组），结构如下：
{{
  "entries": [
    {{"action":"record","amount":数字,"type":"expense|income","category":"中文类别","date":"YYYY-MM-DD 或 null","note":"备注或 null","confidence":0.0-1.0}},
    ...
  ]
}}

【字段规则】（跟单笔一样）
- action: record / query / delete / unknown
- type: 必须英文 expense 或 income（不要用中文）
- category 只能从：{category_list}
- 类别区分：吃饭=餐厅/外卖/食堂；购物=买菜/超市/水果；医疗=买药/看病
- 工资/奖金/理财 → income；其他大多 expense
- date 不确定就填 null，不要瞎猜

【拆分规则】
- 一句话里有多个"事情+金额"就算多笔
- 常见分隔符：空格、逗号、"和"、"然后"、顿号、换行
- 一个动词短语（"打车去公司"）+ 一个金额 = 一笔
- 没有金额的描述性内容忽略（比如"吃了个饭"，不入库）

【示例】
输入: "早饭 15 中午外卖 30 打车去公司 22"
输出: {{"entries":[{{"action":"record","amount":15,"type":"expense","category":"吃饭","date":null,"note":"早饭","confidence":0.9}},{{"action":"record","amount":30,"type":"expense","category":"吃饭","date":null,"note":"外卖","confidence":0.9}},{{"action":"record","amount":22,"type":"expense","category":"交通","date":null,"note":"打车去公司","confidence":0.9}}]}}

输入: "上午买了菜 88 下午加油 200"
输出: {{"entries":[{{"action":"record","amount":88,"type":"expense","category":"购物","date":null,"note":"买菜","confidence":0.9}},{{"action":"record","amount":200,"type":"expense","category":"交通","date":null,"note":"加油","confidence":0.9}}]}}

输入: "昨天打车 20 昨天买菜 88"
输出: {{"entries":[{{"action":"record","amount":20,"type":"expense","category":"交通","date":"{yesterday}","note":"打车","confidence":0.85}},{{"action":"record","amount":88,"type":"expense","category":"购物","date":"{yesterday}","note":"买菜","confidence":0.85}}]}}

输入: "工资 8000 到账"
输出: {{"entries":[{{"action":"record","amount":8000,"type":"income","category":"工资","date":null,"note":null,"confidence":0.85}}]}}

【严格 JSON 规则】
- 必须是合法 JSON 对象，不要 markdown 代码块
- 没有识别到账目时返回 {{"entries":[]}}
- 不要输出解释、寒暄、思考
"""


def _batch_parse_llm(text: str) -> list[dict]:
    """单次 LLM 调用，返回 entries 列表。失败抛异常。"""
    today_str = date.today().isoformat()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    system = BATCH_SYSTEM_PROMPT.format(
        category_list=CATEGORY_LIST,
        today=today_str,
        yesterday=yesterday_str,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]

    raw = chat_completion(
        messages=messages,
        temperature=0,
        json_mode=True,
        max_tokens=2000,  # 多笔的话内容会更多
    )

    logger.debug("Batch LLM raw output for %r: %s", text[:50], raw[:400])
    parsed = parse_json_safely(raw)

    entries = parsed.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"entries 必须是 list，实际是 {type(entries).__name__}")
    return entries


def batch_parse_intent(text: str) -> tuple[list[ParsedIntent], str, Optional[str]]:
    """批量解析入口。

    Args:
        text: 可能包含多笔账的文本

    Returns:
        (intents, source, error)
        - intents: 解析出的 ParsedIntent 列表
        - source: 'llm' / 'fallback' / 'error'
        - error: 出错时的错误信息

    流程：
    1. LLM 可用 → 尝试 LLM 批量解析
    2. LLM 失败或不可用 → 走规则批量（按分隔符切句再逐条 parse_intent）
    3. 全部失败 → 返回 ([], 'error', 错误信息)
    """
    if not text or not text.strip():
        return [], "error", "输入为空"

    text = text.strip()

    if _llm_available():
        try:
            raw_entries = _batch_parse_llm(text)
            intents = []
            for e in raw_entries:
                try:
                    intent = _validate_intent_dict(e)
                    if intent.action == "record" and intent.amount and intent.type:
                        intents.append(intent)
                except (ValidationError, ValueError):
                    logger.warning("跳过无法校验的条目: %s", e)
            return intents, "llm", None
        except (LLMError, ValueError) as e:
            logger.warning("批量 LLM 解析失败，走规则兜底: %s", e)

    # ===== 规则兜底：按分隔符切句 =====
    return _batch_parse_fallback(text)


def _batch_parse_fallback(text: str) -> tuple[list[ParsedIntent], str, Optional[str]]:
    """规则兜底：把文本切成多个子句，逐条 parse_intent。"""
    # 常见分隔符：中文顿号、逗号、空格+数字前、"然后"、"还有"
    # 简单策略：先用标点切，再用空格+数字切
    chunks = re.split(
        r"[，,、；;\n]|(?:然后|还有|另外|接着)",
        text,
    )

    intents: list[ParsedIntent] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        result: IntentResult = parse_intent(chunk)
        if result.success and result.data and result.data.action == "record":
            intents.append(result.data)

    if not intents:
        return [], "error", "无法从文本中识别出任何账目"
    return intents, "fallback", None


def batch_entry_dicts(text: str) -> tuple[list[dict], str, Optional[str]]:
    """批量解析的简化入口：直接返回 entry dict 列表（便于入库）。"""
    intents, source, error = batch_parse_intent(text)
    entries = [
        {
            "amount": i.amount,
            "type": i.type,
            "category": i.category or "其他支出",
            "date": i.date.isoformat() if i.date else None,
            "note": i.note,
        }
        for i in intents
    ]
    return entries, source, error
