# -*- coding: utf-8 -*-
"""
OpenAI 兼容协议的 LLM 客户端。

为什么用 httpx 直接调不用 openai 官方 SDK？
1. 官方 SDK 有版本锁，OpenAI 一升级 API 就炸
2. 直接调 HTTP 更透明，出问题一眼能定位
3. 一份代码切所有兼容厂商：OpenAI / DeepSeek / 通义 / 智谱 / Moonshot / Ollama

配置在 config.py，只改 base_url + api_key + model 三个字段就能切模型。
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 调用或响应异常。"""


class LLMUnavailable(LLMError):
    """网络/鉴权/超时错误，属于可重试的瞬态错误。"""


class LLMResponseFormatError(LLMError):
    """LLM 返回内容不是合法 JSON，或结构不符合预期。"""


def _placeholder_key_warning(api_key: str) -> bool:
    """API Key 是占位符时返回 True，调用方应走 fallback。"""
    return "placeholder" in api_key.lower() or len(api_key) < 20


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    json_mode: bool = False,
    max_tokens: int = 500,
) -> str:
    """调用 LLM 的 chat completions 接口。

    Args:
        messages: OpenAI 格式的消息列表，[{"role": "system", "content": "..."}]
        temperature: 随机性，None 时用 settings.llm_temperature（结构化任务用低值）
        json_mode: True 时启用 response_format=json_object，强制返回合法 JSON
        max_tokens: 生成长度上限，避免用户输入一句话 LLM 返回 2000 token

    Returns:
        LLM 回复的纯文本（assistant 消息的 content）

    Raises:
        LLMUnavailable: 网络/鉴权/超时错误
        LLMResponseFormatError: 响应格式异常（无 choices、content 为空）
    """
    if _placeholder_key_warning(settings.llm_api_key):
        raise LLMUnavailable("LLM API Key 未配置，使用占位符无法调用真实模型")

    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        # OpenAI 官方 + 大多数兼容厂商（DeepSeek、通义、智谱、Moonshot）都支持
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=settings.llm_timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error("LLM HTTP %d: %s", resp.status_code, resp.text[:500])
                raise LLMUnavailable(f"LLM API HTTP {resp.status_code}")
            data = resp.json()
    except httpx.TimeoutException as e:
        raise LLMUnavailable(f"LLM 请求超时（{settings.llm_timeout}s）: {e}") from e
    except httpx.HTTPError as e:
        raise LLMUnavailable(f"LLM 网络错误: {e}") from e

    # 响应结构校验
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMResponseFormatError(f"LLM 响应结构异常: {data}") from e

    if not content or not isinstance(content, str):
        raise LLMResponseFormatError(f"LLM 返回空 content: {data}")

    return content.strip()


def embed(texts: list[str]) -> list[list[float]]:
    """批量向量化文本。

    Args:
        texts: 待向量化的字符串列表（问题和文档要分开调，因为 embedding 通常要"角色前缀"）

    Returns:
        每个字符串对应一个向量（长度取决于模型，text-embedding-3-small 是 1536 维）

    Raises:
        LLMUnavailable: API Key 未配置或请求失败
    """
    if _placeholder_key_warning(settings.llm_api_key):
        raise LLMUnavailable("Embedding API Key 未配置，使用占位符无法调用真实模型")

    url = f"{settings.llm_base_url.rstrip('/')}/embeddings"
    payload = {
        "model": settings.embedding_model,
        "input": texts,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=settings.llm_timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise LLMUnavailable(f"Embedding API HTTP {resp.status_code}")
            data = resp.json()
    except httpx.HTTPError as e:
        raise LLMUnavailable(f"Embedding 请求失败: {e}") from e

    # 按 index 排序保证顺序一致（某些厂商可能乱序返回）
    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in items]
