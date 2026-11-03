# -*- coding: utf-8 -*-
"""
应用入口：FastAPI 启动文件。

启动方式：
    python main.py
    # 或
    uvicorn main:app --reload

首次启动会自动建表 + 灌入示例数据，方便直接给爸妈演示。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from database import init_db
from routes import bookkeeping as bookkeeping_routes
from routes import rag as rag_routes
from routes import stats as stats_routes

# ============ 日志配置 ============

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ============ 启动钩子 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建表 + 灌种子数据，关闭时清理资源。

    为什么用 lifespan 而不用 @app.on_event("startup")：
    - on_event 是 FastAPI 的旧写法，官方已弃用
    - lifespan 支持 async with，能管理跨资源的生命周期
    - 单元测试时可以用 TestClient(app) 手动触发
    """
    logger.info("初始化数据库...")
    init_db()

    # 首次启动灌种子数据
    if settings.seed_on_first_run:
        from data.seed_data import seed_if_empty
        n = seed_if_empty()
        if n > 0:
            logger.info("灌入示例数据 %d 条（首次启动）", n)

    # 启动时为所有账目建向量索引（保证 RAG 向量检索有数据可用）
    await _build_vector_index_if_needed()

    logger.info("应用启动完成：%s", settings.app_name)
    yield
    logger.info("应用关闭")


async def _build_vector_index_if_needed() -> None:
    """启动时为所有账目建向量索引。

    幂等：VectorStore 内部按 id 去重，重复调用不会产生重复条目。
    失败不阻塞启动：embedding API 挂了也能起来，RAG 会自动降级到关键词。
    """
    from ai.rag import get_store, index_entries
    from database import SessionLocal
    from models.db import BookkeepingEntryORM
    from sqlalchemy import select

    try:
        if len(get_store()) > 0:
            return  # 已有索引，跳过

        db = SessionLocal()
        try:
            entries = [
                {
                    "id": e.id,
                    "date": e.date.isoformat() if e.date else "",
                    "type": e.type,
                    "amount": e.amount,
                    "category": e.category,
                    "note": e.note,
                }
                for e in db.scalars(select(BookkeepingEntryORM)).all()
            ]
            if not entries:
                return
            n = await index_entries(entries)
            logger.info("向量索引已建立：%d 条", n)
        finally:
            db.close()
    except Exception as e:
        logger.warning("向量索引构建失败（不影响启动）: %s", e)


# ============ 应用实例 ============

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="家庭记账系统：自然语言记账 + RAG 检索 + 月度统计",
    lifespan=lifespan,
)

# 允许前端跨域（前端 HTML 可能从 file:// 打开）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 挂载路由 ============

app.include_router(bookkeeping_routes.router)
app.include_router(stats_routes.router)
app.include_router(rag_routes.router)


# ============ 静态资源 ============

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============ 首页 ============

@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """返回手机友好的前端页面。"""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "index.html 未找到，请检查 static/ 目录"}


@app.get("/health")
def health() -> dict:
    """健康检查。运维监控用。"""
    return {
        "status": "ok",
        "app": settings.app_name,
        "db": "sqlite" if settings.db_url.startswith("sqlite") else "other",
        "llm_configured": "placeholder" not in settings.llm_api_key.lower(),
        "time": date.today().isoformat(),
    }


# ============ 启动入口 ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
