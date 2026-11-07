# -*- coding: utf-8 -*-
"""
配置管理：三级优先级（环境变量 > .env 文件 > 默认值）

设计理由：
1. 团队协作时每人 API Key 不同 → .env 存在本地不提交
2. CI/CD 部署时环境变量覆盖 → 生产/测试环境不用改代码
3. 默认值兜底 → 新人 git clone 下来 python main.py 零配置就能跑

"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 相对 config.py 所在目录找，这样从任何 CWD 启动都能加载到
# （否则 pydantic-settings 只会从 CWD 找，从别处启动时 API Key 就丢了）
_ENV_FILE = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    """应用配置。字段按功能分组，用 Field 提供描述和约束。"""

    # ============ 应用基础 ============
    app_name: str = Field("AI 家庭记账系统", description="应用名称")
    debug: bool = Field(False, description="是否开启调试模式")

    # ============ 数据库 ============
    # sqlite:/// 前缀 → SQLite；mysql+pymysql:// → MySQL
    # 换库只改这一行，代码零修改
    db_url: str = Field(
        default="sqlite:///./data/bookkeeping.db",
        description="数据库连接串。开发默认 SQLite，生产可切 MySQL",
    )
    db_echo: bool = Field(False, description="是否打印 SQL 语句（调试用）")

    # ============ LLM（OpenAI 兼容协议）============
    # 换模型只改这三个字段：改 base_url + api_key + model
    # 支持 OpenAI / DeepSeek / 通义 / 智谱 / Moonshot / 本地 Ollama
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI 兼容 API 的 base URL。切模型只改这里",
    )
    llm_api_key: str = Field(
        default="sk-placeholder",
        description="API Key。用 placeholder 兜底，没配也能启动",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="模型 ID。gpt-4o-mini / deepseek-chat / qwen-plus 等",
    )
    llm_timeout: int = Field(15, ge=1, description="单次请求超时秒数")
    llm_max_retries: int = Field(2, ge=0, description="JSON 解析失败时的重试次数")
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0, description="结构化任务用低温度")
    vision_model: str = Field(
        default="glm-4v",
        description="视觉模型 ID。用于截图识别记账。智谱 glm-4v / OpenAI gpt-4o 等",
    )

    # ============ RAG（Embedding）============
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding 模型。同源模型：问题和文档用同一个 embedding",
    )
    rag_top_k: int = Field(5, ge=1, description="RAG 检索返回的 top-k 结果数")
    rag_similarity_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="余弦相似度阈值。低于此值走关键词兜底"
    )

    # ============ 应用启动 ============
    seed_on_first_run: bool = Field(
        True, description="首次启动时自动灌入示例数据，方便演示"
    )

    # ============ pydantic-settings 配置 ============
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略未知环境变量，避免配置漂移时炸掉
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例：全应用共用同一份配置对象。

    lru_cache 装饰器确保 Settings 只实例化一次。
    测试时可以用 get_settings.cache_clear() 清除后重新加载。
    """
    return Settings()


# 便捷全局访问
settings: Settings = get_settings()
