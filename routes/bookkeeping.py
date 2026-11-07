# -*- coding: utf-8 -*-
"""
记账相关 API 路由。

REST 设计：
- POST   /api/entries              手动创建一条账目
- POST   /api/entries/natural      自然语言记账（走 AI 意图解析）
- GET    /api/entries              查询列表（支持时间/类别/类型过滤）
- GET    /api/entries/{id}         查询单条
- PATCH  /api/entries/{id}         局部更新
- DELETE /api/entries/{id}         删除
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from sqlalchemy.orm import Session

from ai import csv_parser
from ai.vision import extract_entry_from_image
from database import get_db
from models.db import BookkeepingEntryORM
from models.schemas import (
    BatchNaturalLanguageRequest,
    BatchNaturalLanguageResponse,
    BookkeepingEntryCreate,
    BookkeepingEntryResponse,
    BookkeepingEntryUpdate,
    CsvImportEntry,
    CsvImportError,
    CsvImportResponse,
    EntryListResponse,
    ImageExtractResult,
    NaturalLanguageEntryRequest,
    NaturalLanguageEntryResponse,
)
from services.bookkeeping import (
    _index_after_write,
    _orm_to_response,
    create_entry,
    delete_entry,
    get_entry,
    last_n_days,
    list_entries,
    record_batch_from_natural_language,
    record_from_natural_language,
    update_entry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/entries", tags=["bookkeeping"])


@router.get("", response_model=EntryListResponse)
def list_all(
    start: Optional[date] = Query(default=None, description="起始日期（含）"),
    end: Optional[date] = Query(default=None, description="截止日期（不含）"),
    category: Optional[str] = Query(default=None, description="类别过滤"),
    type: Optional[Literal["income", "expense"]] = Query(default=None, description="收支类型"),
    limit: int = Query(default=100, ge=1, le=1000, description="分页大小"),
    offset: int = Query(default=0, ge=0, description="分页偏移"),
    db: Session = Depends(get_db),
) -> EntryListResponse:
    """查询账目列表。支持按日期范围、类别、收支类型过滤。"""
    return list_entries(
        db,
        month_start=start,
        month_end=end,
        category=category,
        entry_type=type,
        limit=limit,
        offset=offset,
    )


@router.get("/recent", response_model=EntryListResponse)
def recent(days: int = Query(default=7, ge=1, le=90),
           db: Session = Depends(get_db)) -> EntryListResponse:
    """最近 n 天账目，前端首页展示用。"""
    return last_n_days(db, n=days)


@router.get("/{entry_id}", response_model=BookkeepingEntryResponse)
def get_single(entry_id: int, db: Session = Depends(get_db)) -> BookkeepingEntryResponse:
    """查询单条账目。"""
    result = get_entry(db, entry_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"账目不存在: id={entry_id}")
    return result


@router.post("", response_model=BookkeepingEntryResponse,
             status_code=status.HTTP_201_CREATED)
def create_new(entry: BookkeepingEntryCreate,
               db: Session = Depends(get_db)) -> BookkeepingEntryResponse:
    """手动创建账目（表单提交、脚本录入用）。"""
    return create_entry(db, entry)


@router.post("/natural", response_model=NaturalLanguageEntryResponse,
             status_code=status.HTTP_201_CREATED)
def create_from_natural(req: NaturalLanguageEntryRequest,
                        db: Session = Depends(get_db)) -> NaturalLanguageEntryResponse:
    """自然语言记账入口。

    请求示例：{"text": "午饭 15"}
    响应会告诉用户：
    - intent.success: AI 是否成功解析
    - intent.source: 用的是 LLM 还是规则兜底
    - stored: 是否成功入库
    - entry: 入库后的完整记录（含 id）

    """
    return record_from_natural_language(db, req.text)


@router.post("/batch", response_model=BatchNaturalLanguageResponse,
             status_code=status.HTTP_201_CREATED)
def create_batch_from_natural(req: BatchNaturalLanguageRequest,
                              db: Session = Depends(get_db)) -> BatchNaturalLanguageResponse:
    """批量自然语言记账：一段文本可能包含多笔账。

    场景：晚上复盘一天
    请求：{"text": "早饭 15 中午外卖 30 打车 22 晚饭 45"}
    返回：4 笔账都入库，同时返回每笔的详情供前端展示
    """
    return record_batch_from_natural_language(db, req.text)


@router.post("/from-image", response_model=BookkeepingEntryResponse,
             status_code=status.HTTP_201_CREATED)
async def create_from_image(
    file: UploadFile = File(..., description="收据/支付截图，支持 jpg/png/webp"),
    date_override: Optional[date] = Query(default=None,
                                          description="手动指定日期，覆盖图片里识别的日期"),
    db: Session = Depends(get_db),
) -> BookkeepingEntryResponse:
    """截图识别记账：上传图片 → 视觉模型提取金额/分类/日期 → 入库。

    支持图片类型：jpeg、png、webp。大小限制 10MB。
    使用 settings.vision_model（默认 glm-4v）识别。
    """
    # 校验文件类型
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的图片类型：{file.content_type}。仅支持 jpg/png/webp",
        )

    # 读取字节 + 大小限制
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片过大，请压缩到 10MB 以内")
    if not content:
        raise HTTPException(status_code=400, detail="图片内容为空")

    # 调用视觉模型
    from ai.llm import LLMError
    try:
        extracted = extract_entry_from_image(content, file.content_type)
    except LLMError as e:
        logger.error("视觉识别失败：%s", e)
        raise HTTPException(status_code=502, detail=f"图片识别失败：{e}")

    # 组装账目
    final_date = date_override or (date.fromisoformat(extracted["date_hint"])
                                   if extracted.get("date_hint") else date.today())
    entry = BookkeepingEntryCreate(
        date=final_date,
        amount=extracted["amount"],
        type=extracted["type"],
        category=extracted["category"],
        note=extracted.get("note"),
        source="nl",  # 复用 nl 来源标记（视觉也是 AI 识别的）
    )
    return create_entry(db, entry)


@router.post("/import-csv", response_model=CsvImportResponse)
async def import_csv(
    file: UploadFile = File(..., description="CSV 文件，UTF-8/GBK 编码，需含表头"),
    db: Session = Depends(get_db),
) -> CsvImportResponse:
    """批量导入 CSV 记账文件。

    支持的列名（模糊匹配，大小写不敏感）：
    - date: 日期 / 时间 / date
    - amount: 金额 / amt / amount / 交易额
    - type: 类型 / 收支 / type（可选，负金额默认支出）
    - category: 分类 / 类别 / category
    - note: 备注 / 说明 / note / 商户

    大小限制 5MB。
    """
    # 校验
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in {
        "text/csv", "text/plain", "application/csv",
        "application/vnd.ms-excel", "application/octet-stream",
    }:
        # 常见浏览器把 .csv 报为 text/csv 或 application/vnd.ms-excel，允许这些
        pass

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件过大，请控制在 5MB 以内")
    if not content.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")

    # 解析
    result = csv_parser.parse_csv_content(content)

    # 入库成功的条目
    imported: list[BookkeepingEntryResponse] = []
    import_errors: list[CsvImportError] = []
    for pe in result.entries:
        try:
            orm = BookkeepingEntryORM(
                date=pe.date,
                type=pe.type,
                amount=pe.amount,
                category=pe.category,
                note=pe.note,
                source="nl",  # 从文件导入的算 AI 辅助录入
            )
            db.add(orm)
            db.flush()  # 拿 id
            _index_after_write(orm)
            imported.append(_orm_to_response(orm))
        except Exception as e:
            import_errors.append(CsvImportError(
                row_index=pe.row_index,
                reason=f"入库失败：{e}",
            ))
    db.commit()

    # 汇总错误
    all_errors = [
        CsvImportError(row_index=err.row_index, reason=err.reason)
        for err in result.errors
    ] + import_errors

    return CsvImportResponse(
        total_rows=result.total_rows,
        parsed_count=len(result.entries),
        imported_count=len(imported),
        errors=all_errors,
        entries=imported,
    )


@router.patch("/{entry_id}", response_model=BookkeepingEntryResponse)
def update_single(entry_id: int, update: BookkeepingEntryUpdate,
                  db: Session = Depends(get_db)) -> BookkeepingEntryResponse:
    """局部更新账目。未传的字段不改。"""
    result = update_entry(db, entry_id, update)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"账目不存在: id={entry_id}")
    return result


@router.delete("/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_single(entry_id: int, db: Session = Depends(get_db)) -> Response:
    """删除账目。返回 204 No Content（无响应体）。"""
    ok = delete_entry(db, entry_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"账目不存在: id={entry_id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
