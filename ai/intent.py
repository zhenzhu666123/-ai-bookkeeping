# -*- coding: utf-8 -*-
"""
意图路由：把自然语言解析为结构化的 JSON 意图。

【JSON 正确性保障】5 层机制，任何一层失败自动降级：

1. Prompt 层
   - system prompt 里明确说"必须返回严格 JSON 对象，不要 markdown 代码块"
   - 用 Few-Shot 示例让模型看清格式

2. response_format 层（API 强制）
   - 调 OpenAI 兼容 API 时传 response_format={"type": "json_object"}
   - 服务端保证返回值是合法 JSON，不需要我们 parse 花括号

3. 解析层
   - json.loads 严格解析，失败立刻重试
   - 兼容 markdown 包裹（```json ... ```）以防某些模型不听话
   - 兼容裸字典字符串（有些模型返回 {"key": value} 但没 ```包裹）

4. Schema 层（Pydantic 校验）
   - 用 ParsedIntent 严格校验字段类型、必填、枚举值
   - 校验失败 → 触发重试（最多 settings.llm_max_retries 次）

5. 兜底层
   - LLM 全部失败 → 走正则规则兜底（fallback.py）
   - 兜底也失败 → 明确报错让用户知道（不静默丢弃）

【置信度阈值】
- LLM 返回但 confidence < 0.5 → 走兜底（LLM 自己也说不准）
- LLM 返回但 action 是 unknown → 走兜底
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from typing import Optional

from pydantic import ValidationError

from ai.fallback import parse_intent_by_rules
from ai.llm import LLMError, chat_completion
from config import settings
from models.schemas import IntentResult, ParsedIntent

logger = logging.getLogger(__name__)


# ============================================================
# Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是家庭记账助手的意图解析器。用户会用自然语言描述一笔账目，
你要把它解析成一个严格合法的 JSON 对象。

【当前日期】
今天是 {today}。用户说"今天"、"昨天"、"前天"时要按此推算具体日期。

【输出 JSON 的字段】
- action: 三选一
  * "record" - 记账（有金额时几乎一定是这个）
  * "query" - 查询（"我上个月花了多少"、"上次猪肉多少钱"）
  * "delete" - 删除
  * "unknown" - 无法判断
- amount: 数字，支出和收入都是正数（例：15、88.5、1000）
- type: "income"（收入）或 "expense"（支出），只有 record 时填。**必须用英文，不要用中文**
- category: 中文类别，只能从以下列表选一个最接近的：
  {category_list}
- date: 日期，格式 YYYY-MM-DD（例："2026-03-15"）；无法确定时填 null。**不要瞎猜日期**
- note: 简短备注，不超过 20 字；无法提取时填 null
- confidence: 0.0 到 1.0 的置信度，0.5 以下视为不确定

【类别区分规则】
- "吃饭" = 在餐厅吃饭、外卖、食堂（消费食物本身）
- "购物" = 买菜、买水果、买零食、超市购物（购买食材/商品）
- "医疗" = 买药、看病、挂号、医院
- "工资" / "奖金" / "理财" → income；其他大多为 expense

【严格 JSON 输出规则】
- 必须返回一个合法的 JSON 对象
- 不要包含任何 markdown 代码块（不要 ```json）
- 不要输出任何解释、思考过程、寒暄
- 所有字段必须出现，不确定的填 null

【示例】
输入: "午饭 15"
输出: {{"action":"record","amount":15,"type":"expense","category":"吃饭","date":null,"note":"午饭","confidence":0.9}}

输入: "买菜 88"
输出: {{"action":"record","amount":88,"type":"expense","category":"购物","date":null,"note":"买菜","confidence":0.9}}

输入: "药店买感冒药 35"
输出: {{"action":"record","amount":35,"type":"expense","category":"医疗","date":null,"note":"感冒药","confidence":0.9}}

输入: "昨天打车 20"
输出: {{"action":"record","amount":20,"type":"expense","category":"交通","date":"{yesterday}","note":"打车","confidence":0.85}}

输入: "工资 8000 到了"
输出: {{"action":"record","amount":8000,"type":"income","category":"工资","date":null,"note":null,"confidence":0.85}}

输入: "上次猪肉多少钱"
输出: {{"action":"query","amount":null,"type":null,"category":null,"date":null,"note":"猪肉","confidence":0.8}}
"""


CATEGORY_LIST = "吃饭, 交通, 购物, 水电, 医疗, 教育, 人情, 工资, 奖金, 理财, 其他支出, 其他收入"


# ============================================================
# JSON 解析辅助
# ============================================================

def _strip_markdown_codeblock(text: str) -> str:
    """去掉 LLM 常见的 markdown 代码块包裹。

    处理三种情况：
    1. ```json\n{...}\n```（标准）
    2. ```\n{...}\n```（无语言标记）
    3. 纯文本（无包裹，直接返回）
    """
    text = text.strip()
    # 情况 1 和 2
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 情况 3：尝试提取第一个 { 到最后一个 }
    if text.startswith("{") and text.endswith("}"):
        return text
    # 兜底：查找第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_json_safely(text: str) -> dict:
    """安全解析 JSON，失败抛 ValueError。"""
    cleaned = _strip_markdown_codeblock(text)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}; cleaned text = {cleaned[:200]}") from e
    if not isinstance(result, dict):
        raise ValueError(f"JSON 必须是对象（dict），实际是 {type(result).__name__}")
    return result


# ============================================================
# 意图解析入口
# ============================================================

def _llm_available() -> bool:
    """判断 LLM 是否可用（Key 是否配置）。"""
    return not (
        "placeholder" in settings.llm_api_key.lower()
        or len(settings.llm_api_key) < 20
    )


def parse_intent_with_llm(text: str) -> dict:
    """单次 LLM 调用，返回解析后的 dict。失败抛异常由上层处理。

    把"今天"和"昨天"的 ISO 日期注入 prompt，让推理模型能正确解析
    "昨天打车 20" 这类相对日期表述。
    """
    today_str = date.today().isoformat()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    system = SYSTEM_PROMPT.format(
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
        temperature=0,  # 意图解析必须确定性输出
        json_mode=True,  # 强制 JSON 模式
        max_tokens=1500,  # 推理模型（如 SenseNova）需要先思考再输出，需要足够 token
    )

    logger.debug("LLM raw output for %r: %s", text[:50], raw[:300])
    return parse_json_safely(raw)


def _validate_intent_dict(data: dict) -> ParsedIntent:
    """用 Pydantic 严格校验解析结果，失败抛 ValidationError。"""
    # 处理 LLM 可能返回的 date 字符串
    if data.get("date") is not None and isinstance(data.get("date"), str):
        try:
            data["date"] = date.fromisoformat(data["date"])
        except ValueError:
            data["date"] = None

    # 处理 amount 类型转换
    if data.get("amount") is not None:
        try:
            data["amount"] = round(float(data["amount"]), 2)
        except (TypeError, ValueError):
            data["amount"] = None

    return ParsedIntent(**data)


def parse_intent(text: str) -> IntentResult:
    """意图解析主入口，带重试和兜底。

    流程：
    1. LLM 不可用 → 直接走规则兜底
    2. LLM 可用 → 最多重试 settings.llm_max_retries 次
    3. 每次失败：记录日志，继续尝试
    4. 全部失败 → 走规则兜底
    5. 兜底也失败 → 返回明确的错误信息

    Args:
        text: 用户自然语言输入

    Returns:
        IntentResult，包含 success / data / source / error 四个字段
    """
    if not text or not text.strip():
        return IntentResult(success=False, error="输入为空", source="error")

    text = text.strip()

    # ===== 情况 1: LLM 不可用，直接兜底 =====
    if not _llm_available():
        logger.info("LLM 未配置，走规则兜底")
        return _try_fallback(text, reason="LLM 未配置")

    # ===== 情况 2: 尝试 LLM，失败重试 =====
    last_error: Optional[str] = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            raw_dict = parse_intent_with_llm(text)
            # 先保存原始 LLM 输出的 JSON 字符串（避免 _validate_intent_dict 会改 date 为 date 对象）
            raw_json_str = json.dumps(raw_dict, ensure_ascii=False)
            intent = _validate_intent_dict(raw_dict)

            # 校验通过
            if intent.confidence >= 0.5 and intent.action != "unknown":
                return IntentResult(
                    success=True,
                    data=intent,
                    source="llm",
                    raw_llm_output=raw_json_str,
                )

            # 置信度太低或 unknown → 记日志继续尝试
            last_error = f"置信度低 confidence={intent.confidence} action={intent.action}"
            logger.warning("LLM 返回低置信度 attempt=%d: %s", attempt, last_error)

        except (LLMError, ValueError, ValidationError) as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.warning("LLM 解析失败 attempt=%d: %s", attempt, last_error)
            continue

    # ===== 情况 3: LLM 全部失败，走规则兜底 =====
    logger.info("LLM 全部失败（%d 次），走规则兜底。最后错误: %s",
                settings.llm_max_retries + 1, last_error)
    return _try_fallback(text, reason=last_error or "LLM 全部失败")


def _try_fallback(text: str, *, reason: str) -> IntentResult:
    """尝试规则兜底。兜底成功返回 source=fallback，失败返回明确错误。"""
    try:
        fallback_dict = parse_intent_by_rules(text)
    except Exception as e:  # pragma: no cover
        logger.error("规则兜底异常: %s", e)
        return IntentResult(success=False, source="error",
                            error=f"兜底异常: {e}")

    if fallback_dict is None:
        return IntentResult(
            success=False,
            source="error",
            error=f"无法解析您的输入。请尝试类似 '午饭 15'、'工资 8000 到账' 的格式。"
                  f"（原因: {reason}）",
        )

    try:
        intent = _validate_intent_dict(fallback_dict)
    except ValidationError as e:
        logger.error("兜底结果校验失败: %s", e)
        return IntentResult(success=False, source="error",
                            error=f"兜底结果校验失败: {e}")

    return IntentResult(success=True, data=intent, source="fallback")
