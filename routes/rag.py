# -*- coding: utf-8 -*-
"""
RAG 搜索 API 路由。

RAG 检索路径透明，响应里明确告诉你：
- mode: 这次是 vector（向量）、keyword（关键词）还是 hybrid（混合）
- total_entries_scanned: 扫描了多少条
用户能看到检索走的是哪条路径，出问题可以 debug。"
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ai.ask import ask_bookkeeping_question
from database import get_db
from models.schemas import RAGSearchRequest, RAGSearchResponse
from services.bookkeeping import rebuild_index, search_by_query

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.post("/search", response_model=RAGSearchResponse)
async def search(req: RAGSearchRequest,
                 db: Session = Depends(get_db)) -> RAGSearchResponse:
    """RAG 搜索：查询历史账目。

    请求示例：{"query": "上次猪肉多少钱", "top_k": 5}
    响应会告诉你：
    - mode: 实际命中的检索策略（vector / keyword / hybrid / empty）
    - results: top-k 结果，每条带相似度分数
    - total_entries_scanned: 扫描范围
    """
    return await search_by_query(db, req.query, top_k=req.top_k)


@router.post("/ask")
def ask(req: RAGSearchRequest,
        db: Session = Depends(get_db)) -> dict:
    """对话式账单问答：把自然语言问题转成数据查询 + 生成自然语言答案。

    请求：{"query": "这个月餐饮花了多少"}
    返回：{"question", "answer", "answer_source", "filter", "data"}

    跟 /search 的区别：
    - /search 只返回匹配条目
    - /ask  返回 AI 写的一句自然语言答案
    """
    return ask_bookkeeping_question(db, req.query)


@router.post("/reindex")
def reindex(db: Session = Depends(get_db)) -> dict:
    """重建向量索引。

    什么时候用：
    1. 换了 embedding 模型
    2. 索引损坏（比如手动清了 SQLite 但向量库还有）
    3. 想重新跑一遍（比如 embedding API 之前挂了，现在好了）
    """
    n = rebuild_index(db)
    return {"indexed_count": n, "status": "ok"}
