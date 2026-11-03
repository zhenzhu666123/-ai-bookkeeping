# -*- coding: utf-8 -*-
"""
数据库连接层：靠 DB_URL 一键切换数据源

设计要点：
1. 开发环境用 SQLite（零配置、文件即数据库）
2. 生产环境改一行 DB_URL 切 MySQL（不用改代码）
3. SQLite 需要 check_same_thread=False 让 FastAPI 多线程访问
4. 用 FastAPI 的 yield 依赖注入，请求结束自动关连接

面试话术：
"数据库抽象层用 SQLAlchemy ORM，切数据源只改 DB_URL 一个环境变量。
开发本地跑 SQLite（零依赖），部署到服务器切 MySQL。
连接用 FastAPI 依赖注入管理生命周期，请求结束自动释放。"
"""
from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, scoped_session

from config import settings


def _build_engine():
    """根据 DB_URL 前缀构建不同的 engine。

    SQLite 特殊处理：check_same_thread=False 让跨线程共享连接，
    StaticPool 让所有线程复用同一个连接（SQLite 是文件数据库，适合单连接）
    """
    from sqlalchemy.pool import StaticPool

    is_sqlite = settings.db_url.startswith("sqlite")
    connect_args: dict = {"check_same_thread": False} if is_sqlite else {}
    poolclass = StaticPool if is_sqlite else None

    return create_engine(
        settings.db_url,
        echo=settings.db_echo,
        connect_args=connect_args,
        poolclass=poolclass,
    )


engine = _build_engine()

# SessionLocal：应用主线程用
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# TestingSessionLocal：单元测试用（方便 fixture 里换连接）
TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖注入：每个请求拿一个 Session，结束自动关闭。

    用法：
        @app.get("/entries")
        def list_entries(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """脚本/后台任务用的 Session 上下文管理器。

    用法：
        with session_scope() as db:
            db.add(entry)
        # 出块自动 commit + close
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """首次启动时建表。

    从 models.db 导入 ORM 类，让 SQLAlchemy 感知所有 model 再建表。
    checkfirst=True 让重复调用不报错。
    """
    # 延迟导入避免循环依赖
    from models.db import Base  # noqa: F401

    # SQLite 数据库文件可能在 data/ 目录下，确保目录存在
    if settings.db_url.startswith("sqlite:///"):
        db_path = settings.db_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    Base.metadata.create_all(bind=engine)


def drop_db() -> None:
    """删除所有表（测试用）。"""
    from models.db import Base  # noqa: F401

    Base.metadata.drop_all(bind=engine)
