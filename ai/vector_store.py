# -*- coding: utf-8 -*-
"""
内存向量库：存储 embedding + 元数据，支持余弦相似度检索。

【面试核心问题】"RAG 怎么确保检索到对的知识？"

【我们的答案】4 层保障：

1. 同源 Embedding 模型
   - 问题和文档必须用同一个 embedding 模型（config.embedding_model）
   - 混用模型会导致向量空间不对齐，相似度没意义

2. 余弦相似度（只看方向，忽略量纲）
   - 向量本身可能长度差很大，但方向相似就是语义相近
   - 归一化后点积 = 余弦值，范围 [-1, 1]

3. Top-K 截断 + 相似度阈值
   - 只返回 top-k 结果，避免 prompt 塞爆
   - 分数低于阈值（settings.rag_similarity_threshold）不返回

4. 关键词兜底
   - 向量分数都低 → 判定向量检索不靠谱
   - 自动降级到 BM25 风格的关键词命中
   - 家庭记账场景，"猪肉"、"午饭"这种字面匹配比向量更准

【为什么不存数据库？】
- 数据规模：家庭记账一年不到 500 条，内存够用
- 部署简单：不上 Chroma/Milvus/Qdrant 这些额外依赖
- 面试友好：能手写讲清楚原理，而不是"用了 XX 库"

面试时如果被追问"上千条数据怎么办"，答：换 ChromaDB（本地 SQLite 存向量，零运维）
或 Milvus（生产级），我们的检索逻辑（余弦+关键词兜底）不变，只是把存储层换掉。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VectorRecord:
    """单条向量记录。"""

    record_id: int  # 对应数据库主键
    text: str  # 原始文本（用于展示和关键词兜底）
    embedding: list[float]  # 向量
    meta: dict[str, Any] = field(default_factory=dict)  # 附加元数据（date、category 等）


def _dot(a: list[float], b: list[float]) -> float:
    """点积。"""
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    """欧几里得范数。"""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度。

    分子：点积；分母：两个向量的范数乘积。
    结果范围 [-1, 1]，越大越相似。
    分母为 0 时（零向量）返回 0.0。
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致: {len(a)} vs {len(b)}")
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


class VectorStore:
    """内存向量库。

    为什么用 dict[int, VectorRecord]：
    - record_id 是主键，直接 O(1) 查找
    - 家庭记账数据量小，遍历检索足够快（500 条毫秒级）
    """

    def __init__(self) -> None:
        self._records: dict[int, VectorRecord] = {}

    # ---------- 写入 ----------

    def add(self, record_id: int, text: str, embedding: list[float],
            meta: Optional[dict[str, Any]] = None) -> None:
        """添加一条记录。同 id 会覆盖。"""
        self._records[record_id] = VectorRecord(
            record_id=record_id,
            text=text,
            embedding=embedding,
            meta=meta or {},
        )

    def bulk_add(self, records: list[VectorRecord]) -> None:
        """批量添加。"""
        for r in records:
            self.add(r.record_id, r.text, r.embedding, r.meta)

    def remove(self, record_id: int) -> None:
        """删除一条。不存在不报错。"""
        self._records.pop(record_id, None)

    def clear(self) -> None:
        """清空。"""
        self._records.clear()

    # ---------- 查询 ----------

    def __len__(self) -> int:
        return len(self._records)

    def search_similar(
        self,
        query_embedding: list[float],
        k: int = 5,
        score_threshold: float = 0.0,
        only_ids: Optional[set[int]] = None,
    ) -> list[dict[str, Any]]:
        """按余弦相似度降序检索 top-k。

        Args:
            query_embedding: 查询向量
            k: 返回条数
            score_threshold: 分数阈值，低于此值不返回
            only_ids: 只在这批 record_id 里检索（用于日期过滤等场景）

        Returns:
            [{record_id, score, text, meta}, ...]，按 score 降序
        """
        if not self._records:
            return []

        scored: list[tuple[float, VectorRecord]] = []
        for rec in self._records.values():
            if only_ids is not None and rec.record_id not in only_ids:
                continue
            try:
                sim = cosine_similarity(query_embedding, rec.embedding)
            except ValueError:
                continue  # 维度不一致，跳过
            if sim >= score_threshold:
                scored.append((sim, rec))

        scored.sort(key=lambda x: -x[0])

        return [
            {
                "record_id": rec.record_id,
                "score": round(sim, 4),
                "text": rec.text,
                "meta": rec.meta,
            }
            for sim, rec in scored[:k]
        ]
