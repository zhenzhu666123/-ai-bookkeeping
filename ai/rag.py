# -*- coding: utf-8 -*-
"""
RAG 编排：把向量检索 + 关键词兜底 + LLM 可用判断 串起来。

对外只暴露两个函数：
- index_entry(entry_id, text, meta)   # 新增/更新一条索引
- remove_entry(entry_id)              # 删除索引
- search(query, k)                    # 混合检索，返回结构化结果

我们的四层设计（对应 vector_store.py 的注释）：
1. 同源 embedding 模型
2. 余弦相似度
3. Top-K + 阈值
4. 关键词兜底（向量分数都低时降级到 BM25 风格）

【关键词兜底什么时候触发】
- 没有配置 LLM API Key → 直接走关键词（零依赖模式）
- 向量检索返回结果的最高分 < 阈值 → 认为语义检索不靠谱，走关键词
- 向量分数和关键词命中并存 → 按混合分数（0.7 向量 + 0.3 关键词）排序

【为什么不是纯向量？】
家庭记账里"上次猪肉多少钱"这种查询，
"猪肉"两个字几乎一定会出现在原文里，字面命中比语义相似更靠谱。
向量是补充，关键词是保底。
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from ai.llm import LLMUnavailable, embed
from ai.vector_store import VectorStore
from config import settings

logger = logging.getLogger(__name__)

# 全局向量库单例
_vector_store: VectorStore = VectorStore()


def get_store() -> VectorStore:
    """获取全局向量库实例。"""
    return _vector_store


# ============================================================
# 索引构建：把一条账目变成"可被检索的文本"
# ============================================================

def _entry_to_text(entry: dict[str, Any]) -> str:
    """把账目 dict 序列化成"人话"用于 embedding。

    例如 {"date":"2026-03-15","category":"吃饭","amount":15,"note":"同事请客午饭"}
    → "2026-03-15 吃饭 15 元 同事请客午饭"

    为什么这么拼？
    - 让 embedding 模型看到"日期 + 类别 + 金额 + 备注"的完整语义
    - 备注通常含关键词（"猪肉"、"打车"），是最有价值的信息
    """
    parts = [
        str(entry.get("date", "")),
        str(entry.get("category", "")),
        f'{entry.get("amount", 0)} 元',
        str(entry.get("note", "") or ""),
        str(entry.get("type", "")),
    ]
    return " ".join(p for p in parts if p).strip()


async def index_entries(entries: list[dict[str, Any]]) -> int:
    """批量建立索引。返回成功建立的数量。

    失败不抛异常：embedding API 可能挂了，不能让整个系统崩。
    失败的条目打日志，后续可重跑。
    """
    if not entries:
        return 0

    texts = [_entry_to_text(e) for e in entries]
    ids = [e["id"] for e in entries]

    try:
        vectors = await _embed_texts(texts)
    except LLMUnavailable as e:
        logger.warning("Embedding API 不可用，跳过向量索引: %s", e)
        return 0
    except Exception as e:
        logger.error("Embedding 批量失败: %s", e)
        return 0

    count = 0
    for id_, text, vec in zip(ids, texts, vectors):
        try:
            _vector_store.add(id_, text, vec, meta=dict(entries[count]))
            count += 1
        except Exception as e:
            logger.error("索引条目 id=%s 失败: %s", id_, e)

    logger.info("向量索引完成: 成功 %d/%d 条", count, len(entries))
    return count


async def index_entry(entry: dict[str, Any]) -> bool:
    """单条索引。"""
    n = await index_entries([entry])
    return n > 0


def remove_entry(entry_id: int) -> None:
    """删除索引。"""
    _vector_store.remove(entry_id)


def clear_index() -> None:
    """清空索引（测试用）。"""
    _vector_store.clear()


# ============================================================
# 关键词兜底
# ============================================================

def _keyword_score(query: str, text: str) -> float:
    """简单的关键词命中率打分（非 BM25，够用）。

    策略：
    1. 把 query 按标点/空白分词
    2. 对每个词统计在 text 里出现的次数
    3. 归一化：命中词数 / 查询总词数 * 0.5 + 命中密度 * 0.5

    为什么不做真 BM25？
    1. 家庭记账一年几百条，性能没压力
    2. jieba 分词对"午饭"这种两字词有割裂风险（变成"午"+"饭"）
    3. 生产数据量增大时，可以平滑替换为 BM25
    """
    if not query or not text:
        return 0.0

    # 简单分词：按标点和空白
    query_terms = [t for t in _split_terms(query) if t]
    if not query_terms:
        return 0.0

    text_lower = text.lower()
    hit_count = 0
    hit_density = 0.0

    for term in query_terms:
        occurrences = text_lower.count(term.lower())
        if occurrences > 0:
            hit_count += 1
            hit_density += min(occurrences, 3)  # 单次最多算 3 次

    term_hit_ratio = hit_count / len(query_terms)  # 命中了多少比例的查询词
    density_norm = min(hit_density / (len(query_terms) * 3), 1.0)

    # 综合分数：0.6 命中率 + 0.4 密度
    return round(0.6 * term_hit_ratio + 0.4 * density_norm, 4)


def _split_terms(text: str) -> list[str]:
    """把字符串按常见分隔符切开。"""
    import re
    # 按标点、空白、中英文冒号逗号等切
    parts = re.split(r"[\s，,。.、；;：:！!？?/\-()（）【】\[\]]+", text)
    return [p for p in parts if len(p) >= 1]


def keyword_search(query: str, entries: list[dict[str, Any]],
                   k: int = 5) -> list[dict[str, Any]]:
    """关键词检索兜底。"""
    if not entries:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for e in entries:
        text = _entry_to_text(e)
        score = _keyword_score(query, text)
        if score > 0:
            scored.append((score, e))

    scored.sort(key=lambda x: -x[0])
    return [
        {
            "entry_id": e["id"],
            "score": round(s, 4),
            "text": _entry_to_text(e),
            "meta": dict(e),
        }
        for s, e in scored[:k]
    ]


# ============================================================
# Embedding 异步包装
# ============================================================

async def _embed_texts(texts: list[str]) -> list[list[float]]:
    """异步调用 embedding API。

    为什么用 asyncio.to_thread：
    - httpx.Client 是同步阻塞
    - 在 FastAPI async 路由里不能直接调阻塞函数，会阻塞事件循环
    - 用 to_thread 丢到线程池跑，主线程继续处理其他请求
    """
    import asyncio
    return await asyncio.to_thread(embed, texts)


# ============================================================
# 混合检索入口
# ============================================================

async def hybrid_search(
    query: str,
    entries: list[dict[str, Any]],
    k: int = 5,
) -> tuple[Literal["vector", "keyword", "hybrid", "empty"], list[dict[str, Any]]]:
    """混合检索主入口。

    策略：
    1. 先尝试向量检索
    2. 如果向量不可用或分数都太低 → 走关键词
    3. 如果向量分数够高 → 用向量结果，但把关键词命中的补充进来（hybrid）

    Args:
        query: 查询文本
        entries: 候选条目列表（外部已过滤，向量检索也会限制在这批 id 里）
        k: 返回条数

    Returns:
        (mode, results)，mode 是实际命中的检索策略
    """
    if not entries:
        return "empty", []

    # 允许向量检索的 record_id 集合（限制候选范围）
    allowed_ids: set[int] = {int(e["id"]) for e in entries if "id" in e}

    # ===== 尝试向量检索 =====
    vector_results: list[dict[str, Any]] = []
    try:
        [query_vec] = await _embed_texts([query])
        vector_results = _vector_store.search_similar(
            query_vec,
            k=k * 2,  # 多取一些候选，给混合阶段留余地
            score_threshold=0.0,  # 先不设阈值，看完分数再决定
            only_ids=allowed_ids,
        )
        # 按阈值过滤
        vector_results = [
            r for r in vector_results
            if r["score"] >= settings.rag_similarity_threshold
        ]
    except LLMUnavailable:
        logger.info("Embedding 不可用，走关键词兜底")
        return "keyword", _format_results(keyword_search(query, entries, k=k), "keyword")
    except Exception as e:
        logger.error("向量检索异常，走关键词兜底: %s", e)
        return "keyword", _format_results(keyword_search(query, entries, k=k), "keyword")

    # ===== 判断是否走关键词 =====
    if not vector_results:
        # 向量没返回或都低于阈值 → 关键词兜底
        logger.info("向量检索无有效结果，走关键词兜底")
        return "keyword", _format_results(keyword_search(query, entries, k=k), "keyword")

    # ===== 向量有效，看是否需要混合 =====
    top_vec_score = vector_results[0]["score"]

    # 关键词检索补充
    kw_results = keyword_search(query, entries, k=k)
    kw_hit_ids = {r["entry_id"] for r in kw_results}

    # 如果向量 top 分数很高，直接返回向量结果
    if top_vec_score >= 0.6 and len(vector_results) >= k:
        return "vector", _format_results(vector_results[:k], "vector")

    # 否则混合：向量分数 * 0.7 + 关键词命中 * 0.3
    combined: list[tuple[float, dict[str, Any]]] = []

    seen_ids = set()
    for r in vector_results[:k * 2]:
        eid = r["record_id"]  # 向量库返回的是 record_id
        if eid in seen_ids:
            continue
        vec_score = r["score"]
        kw_bonus = 0.3 if eid in kw_hit_ids else 0.0
        combined.append((0.7 * vec_score + kw_bonus, r))
        seen_ids.add(eid)

    # 补上关键词独有的结果
    for r in kw_results[:k]:
        eid = r["entry_id"]  # 关键词库返回的是 entry_id
        if eid in seen_ids:
            continue
        # 统一 key：把 entry_id 别名成 record_id
        combined.append((0.3 * r["score"], {"record_id": eid, "score": r["score"], "meta": r.get("meta", {})}))
        seen_ids.add(eid)

    combined.sort(key=lambda x: -x[0])
    # combined 是 (score, result_dict) 元组列表，取 result_dict 传给 _format_results
    return "hybrid", _format_results([item for _, item in combined[:k]], "hybrid")


def _format_results(items: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """统一格式化成 API 返回格式。"""
    formatted = []
    for item in items:
        meta = item.get("meta", {})
        formatted.append({
            "entry_id": item["record_id"] if "record_id" in item else item["entry_id"],
            "date": str(meta.get("date", "")),
            "category": str(meta.get("category", "")),
            "amount": float(meta.get("amount", 0)),
            "note": meta.get("note"),
            "score": round(float(item["score"]), 4),
            "mode": mode,
        })
    return formatted
