# -*- coding: utf-8 -*-
"""API 端点测试：不启动真实 LLM，验证接口协议。"""
from __future__ import annotations

import os
from datetime import date

import pytest
from fastapi.testclient import TestClient

# 测试前设置环境变量：
# 1. 用临时 SQLite 文件避免污染真实数据
# 2. 关闭 seed 避免污染
# 3. API Key 用占位符，测试只验证规则和接口逻辑，不依赖外部 LLM
os.environ["DB_URL"] = "sqlite:///./data/test_bookkeeping.db"
os.environ["SEED_ON_FIRST_RUN"] = "false"
os.environ["LLM_API_KEY"] = "sk-placeholder-test"  # 强制走规则兜底

from config import get_settings  # noqa: E402
from database import drop_db, init_db  # noqa: E402

# 重新加载配置（清 lru_cache）
get_settings.cache_clear()

from main import app  # noqa: E402


@pytest.fixture(autouse=True)
def setup_db():
    """每个测试前重建数据库：先删后建，保证测试隔离。"""
    drop_db()
    init_db()
    yield
    drop_db()


@pytest.fixture
def client() -> TestClient:
    """FastAPI 测试客户端。"""
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """健康检查端点。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db"] == "sqlite"


def test_create_and_get_entry(client: TestClient) -> None:
    """创建账目 → 按 ID 查询。"""
    payload = {
        "date": date.today().isoformat(),
        "amount": 15.5,
        "type": "expense",
        "category": "吃饭",
        "note": "午饭",
    }
    resp = client.post("/api/entries", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["amount"] == 15.5
    assert data["category"] == "吃饭"
    assert data["source"] == "manual"

    # 按 ID 查询
    resp = client.get(f"/api/entries/{data['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == data["id"]


def test_entry_validation_amount(client: TestClient) -> None:
    """金额必须大于 0，Pydantic 校验生效。"""
    payload = {"amount": -5, "type": "expense", "category": "吃饭"}
    resp = client.post("/api/entries", json=payload)
    assert resp.status_code == 422  # Unprocessable Entity


def test_entry_validation_type(client: TestClient) -> None:
    """type 必须是 income 或 expense。"""
    payload = {"amount": 15, "type": "unknown", "category": "吃饭"}
    resp = client.post("/api/entries", json=payload)
    assert resp.status_code == 422


def test_list_entries(client: TestClient) -> None:
    """列表接口：空库返回空，插入后返回正确数量。"""
    resp = client.get("/api/entries")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0

    # 插入两条
    for amt in (15, 30):
        client.post("/api/entries", json={
            "amount": amt,
            "type": "expense",
            "category": "吃饭",
        })

    resp = client.get("/api/entries")
    assert resp.json()["total"] == 2


def test_filter_by_type(client: TestClient) -> None:
    """按收支类型过滤。"""
    client.post("/api/entries", json={"amount": 100, "type": "income", "category": "工资"})
    client.post("/api/entries", json={"amount": 15, "type": "expense", "category": "吃饭"})

    resp = client.get("/api/entries?type=income")
    data = resp.json()
    assert data["total"] == 1
    assert data["entries"][0]["type"] == "income"


def test_delete_entry(client: TestClient) -> None:
    """删除账目：返回 204，再查返回 404。"""
    resp = client.post("/api/entries", json={
        "amount": 15, "type": "expense", "category": "吃饭"
    })
    entry_id = resp.json()["id"]

    resp = client.delete(f"/api/entries/{entry_id}")
    assert resp.status_code == 204

    resp = client.get(f"/api/entries/{entry_id}")
    assert resp.status_code == 404


def test_natural_language_entry_fallback(client: TestClient) -> None:
    """自然语言记账：LLM 未配置时走规则兜底，仍能记账。

    这个测试验证了哥哥的核心追问之一：
    "AI 挂了怎么办"—— answer: 规则兜底，用户还能记账。
    """
    resp = client.post("/api/entries/natural", json={"text": "午饭 15"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["stored"] is True
    assert data["entry"]["amount"] == 15.0
    assert data["entry"]["category"] == "吃饭"
    assert data["entry"]["type"] == "expense"
    assert data["intent"]["source"] == "fallback"  # 关键：走的是规则兜底
    assert data["entry"]["source"] == "nl"  # 来源标记为 AI 解析


def test_natural_language_query_action(client: TestClient) -> None:
    """自然语言查询动作不入库，只返回意图。"""
    resp = client.post("/api/entries/natural", json={"text": "我上个月花了多少"})
    assert resp.status_code == 201
    data = resp.json()
    # 兜底无法识别为 record → stored=False
    assert data["stored"] is False


def test_monthly_stats_empty(client: TestClient) -> None:
    """月度统计：空库返回全 0。"""
    resp = client.get("/api/stats/monthly", params={"year": 2026, "month": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_income"] == 0
    assert data["total_expense"] == 0
    assert data["balance"] == 0


def test_monthly_stats_with_data(client: TestClient) -> None:
    """月度统计：插入数据后正确汇总。"""
    client.post("/api/entries", json={
        "date": "2026-03-01", "amount": 8000, "type": "income", "category": "工资"
    })
    client.post("/api/entries", json={
        "date": "2026-03-05", "amount": 100, "type": "expense", "category": "吃饭"
    })
    client.post("/api/entries", json={
        "date": "2026-03-10", "amount": 50, "type": "expense", "category": "吃饭"
    })

    resp = client.get("/api/stats/monthly", params={"year": 2026, "month": 3})
    data = resp.json()
    assert data["total_income"] == 8000
    assert data["total_expense"] == 150
    assert data["balance"] == 7850
    assert data["entry_count"] == 3
    assert data["by_category"]["吃饭"] == 150


def test_rag_search_empty_db(client: TestClient) -> None:
    """RAG 搜索：空库返回 empty 模式。"""
    resp = client.post("/api/rag/search", json={"query": "猪肉"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "empty"


def test_rag_search_with_data_keyword_fallback(client: TestClient) -> None:
    """RAG 搜索：有数据但无 embedding API 时走关键词兜底。"""
    client.post("/api/entries", json={
        "date": "2026-03-05", "amount": 30, "type": "expense",
        "category": "购物", "note": "买猪肉 5 斤"
    })
    client.post("/api/entries", json={
        "date": "2026-03-10", "amount": 40, "type": "expense",
        "category": "购物", "note": "买牛肉"
    })

    resp = client.post("/api/rag/search", json={"query": "猪肉"})
    assert resp.status_code == 200
    data = resp.json()
    # 关键词兜底
    assert data["mode"] == "keyword"
    assert len(data["results"]) >= 1
    # 第一条应该是猪肉那条
    top = data["results"][0]
    assert "猪肉" in (top["note"] or "")
    assert top["score"] > 0
