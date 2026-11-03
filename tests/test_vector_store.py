# -*- coding: utf-8 -*-
"""向量库测试：数学正确性 + 检索排序。"""
from __future__ import annotations

import pytest

from ai.vector_store import VectorStore, cosine_similarity


def test_cosine_similar_identical() -> None:
    """两个相同向量：余弦相似度 = 1.0。"""
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similar_orthogonal() -> None:
    """两个正交向量：余弦相似度 = 0.0。"""
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similar_opposite() -> None:
    """两个反方向向量：余弦相似度 = -1.0。"""
    assert cosine_similarity([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


def test_cosine_scale_invariant() -> None:
    """同方向不同长度：相似度还是 1.0（只看方向，忽略量纲）。"""
    assert cosine_similarity([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)


def test_cosine_zero_vector() -> None:
    """零向量：返回 0.0，不报错。"""
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_dimension_mismatch_raises() -> None:
    """维度不一致：抛 ValueError，不静默返回 0。"""
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_search_returns_sorted() -> None:
    """search_similar 返回按分数降序。"""
    store = VectorStore()
    store.add(1, "a", [1.0, 0.0])
    store.add(2, "b", [0.0, 1.0])
    store.add(3, "c", [1.0, 1.0])

    results = store.search_similar([1.0, 0.5], k=2)
    assert len(results) == 2
    # 第一个分数应该 >= 第二个
    assert results[0]["score"] >= results[1]["score"]
    # 最相似的应该是 c：cos([1,0.5], [1,1]) = 0.949 > cos([1,0.5], [1,0]) = 0.894
    assert results[0]["record_id"] == 3


def test_threshold_filters() -> None:
    """低于阈值的被过滤掉。"""
    store = VectorStore()
    store.add(1, "a", [1.0, 0.0])
    store.add(2, "b", [0.0, 1.0])

    # 阈值 0.5 会过滤掉正交的 b
    results = store.search_similar([1.0, 0.0], k=5, score_threshold=0.5)
    assert all(r["score"] >= 0.5 for r in results)
    assert all(r["record_id"] != 2 for r in results)


def test_empty_store() -> None:
    """空库检索返回空列表，不抛异常。"""
    store = VectorStore()
    assert store.search_similar([1.0, 0.0]) == []


def test_remove_entry() -> None:
    """删除记录。"""
    store = VectorStore()
    store.add(1, "a", [1.0, 0.0])
    assert len(store) == 1
    store.remove(1)
    assert len(store) == 0
    # 重复删除不报错
    store.remove(1)
