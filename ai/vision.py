# -*- coding: utf-8 -*-
"""
截图识别记账：从收据/支付截图里提取金额、分类、日期、备注。

【场景】
用户拍了张外卖单、出租车发票、超市小票 → 系统识别金额/商户/时间 → 生成一条账目。

【实现】
调用视觉大模型（智谱 glm-4v / OpenAI gpt-4o / 通义 qwen-vl 等，OpenAI 兼容协议），
把图片 + 提示词发给模型，让它返回结构化 JSON。

【LLM 挂了怎么办】
返回错误，让用户改用文本记账。视觉识别没法用规则兜底（图片就是图片）。
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Optional

import httpx

from ai.llm import LLMError, LLMResponseFormatError, LLMUnavailable, _placeholder_key_warning
from config import settings

logger = logging.getLogger(__name__)


VISION_SYSTEM_PROMPT = """你是收据/支付截图的记账识别模块。
用户会给你一张图片或一段文字描述，你要从中提取**一条**账目的关键信息，返回严格 JSON。

【返回 JSON 结构】
{
    "amount": 15.50,              // 金额（float，单位元）
    "type": "expense",            // "income" 或 "expense"
    "category": "吃饭",           // 分类，从这张列表选一个：吃饭/交通/购物/水电/医疗/教育/人情/其他
    "note": "麦当劳",             // 备注（商户名、场所、事件名等，简短），可以省略
    "date_hint": "2026-10-19",    // 图片里如果有日期就填 ISO 格式 YYYY-MM-DD，没有就 null
    "confidence": 0.85            // 置信度 0-1，模糊不清就低
}

【规则】
1. **只提取一张图的一条主要账目**，如果是清单小票取最大的单笔
2. **金额**要提取数字，去掉货币符号（¥/￥/$）和千分位逗号
3. **分类**必须从列表里选一个，不要造新分类
4. **日期**：有明确年月日就填，只有"10-19"这种短格式按当前年补
5. **收支类型**：默认 expense，只有明确"工资/收入/退款"才用 income
6. **置信度**：清晰可读 0.85+，模糊/部分可见 0.5-0.8，几乎看不清 <0.5
7. **必须是严格 JSON**，不要 markdown 代码块，不要解释文字
"""


def image_to_base64_data_url(image_bytes: bytes, mime_type: str) -> str:
    """把图片 bytes 转成 data URL，方便发给 vision API。"""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _vision_request(payload: dict, model: str) -> str:
    """底层 HTTP 请求。"""
    if _placeholder_key_warning(settings.llm_api_key):
        raise LLMUnavailable("LLM API Key 未配置，使用占位符无法调用真实模型")

    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    payload["model"] = model
    payload["max_tokens"] = 1000
    payload["temperature"] = 0.1

    try:
        with httpx.Client(timeout=settings.llm_timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error("Vision HTTP %d: %s", resp.status_code, resp.text[:500])
                raise LLMUnavailable(f"Vision API HTTP {resp.status_code}")
            data = resp.json()
    except httpx.TimeoutException as e:
        raise LLMUnavailable(f"Vision 请求超时（{settings.llm_timeout}s）: {e}") from e
    except httpx.HTTPError as e:
        raise LLMUnavailable(f"Vision 网络错误: {e}") from e

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMResponseFormatError(f"Vision 响应结构异常: {data}") from e

    if not content or not isinstance(content, str):
        raise LLMResponseFormatError(f"Vision 返回空 content: {data}")
    return content.strip()


def _parse_llm_json(text: str) -> dict:
    """从 LLM 输出里抠出 JSON 对象。容忍 markdown 代码块包裹。"""
    # 去掉可能的 ```json ... ``` 包裹
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMResponseFormatError(f"Vision LLM 返回的不是合法 JSON: {text[:200]}") from e


VALID_CATEGORIES = {"吃饭", "交通", "购物", "水电", "医疗", "教育", "人情", "其他"}
VALID_TYPES = {"income", "expense"}


def _normalize_result(raw: dict) -> dict:
    """规整 LLM 返回的结果：类型转换、类目映射、字段补齐。"""
    result = {
        "amount": None,
        "type": "expense",
        "category": "其他",
        "note": None,
        "date_hint": None,
        "confidence": 0.5,
    }

    # 金额
    try:
        amt = float(raw.get("amount", 0))
        if amt > 0:
            result["amount"] = round(amt, 2)
    except (TypeError, ValueError):
        pass

    # 类型
    t = str(raw.get("type", "expense")).lower().strip()
    if t in VALID_TYPES:
        result["type"] = t
    elif "收入" in t or "income" in t or "ref" in t:
        result["type"] = "income"

    # 分类
    cat = str(raw.get("category", "其他")).strip()
    if cat not in VALID_CATEGORIES:
        # 尝试模糊匹配
        cat_map = {
            "餐": "吃饭", "外卖": "吃饭", "餐费": "吃饭", "food": "吃饭",
            "车": "交通", "打车": "交通", "taxi": "交通", "transport": "交通",
            "买": "购物", "超": "购物", "sho": "购物",
            "电": "水电", "water": "水电",
            "医": "医疗", "药": "医疗",
            "学": "教育", "学费": "教育",
            "红": "人情", "gift": "人情",
        }
        for k, v in cat_map.items():
            if k in cat:
                result["category"] = v
                break
        else:
            result["category"] = "其他"

    # 备注
    note = raw.get("note")
    if isinstance(note, str) and note.strip():
        result["note"] = note.strip()[:100]

    # 日期
    dh = raw.get("date_hint")
    if isinstance(dh, str) and dh.strip():
        result["date_hint"] = dh.strip()[:10]

    # 置信度
    try:
        conf = float(raw.get("confidence", 0.5))
        result["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        pass

    return result


def extract_entry_from_image(
    image_bytes: bytes,
    mime_type: str,
    model: Optional[str] = None,
) -> dict:
    """从图片提取一条账目的关键字段。

    Args:
        image_bytes: 图片二进制内容
        mime_type: 图片 MIME 类型（如 image/jpeg）
        model: 视觉模型 ID。默认用 settings.vision_model

    Returns:
        dict 结构：
        {
            "amount": 15.50,
            "type": "expense",
            "category": "吃饭",
            "note": "麦当劳",
            "date_hint": "2026-10-19",
            "confidence": 0.85,
            "source": "vision"
        }

    Raises:
        LLMUnavailable: API Key 或网络问题
        LLMResponseFormatError: LLM 返回的不是合法 JSON
    """
    if not image_bytes:
        raise LLMResponseFormatError("图片内容为空")

    data_url = image_to_base64_data_url(image_bytes, mime_type)
    payload = {
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "识别这张图片里的账目信息。"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }

    content = _vision_request(payload, model or settings.vision_model)
    raw = _parse_llm_json(content)
    result = _normalize_result(raw)
    result["source"] = "vision"
    return result
