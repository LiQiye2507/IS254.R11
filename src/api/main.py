"""
FastAPI Backend Inference Service cho Hệ Hỗ trợ Ra quyết định DSS (IS254.R11)
-----------------------------------------------------------------------------
Triển khai đầy đủ các REST API theo Mục 5.1 của Đề cương (`DSS_plan.docx`):
- GET  /health          : Kiểm tra trạng thái dịch vụ & mô hình đã nạp.
- GET  /meta/options    : Danh sách môn học, học kỳ & tham số cấu hình.
- GET  /cohort/overview : Thống kê phân tầng rủi ro (Low, Medium, High) & danh sách cảnh báo sớm tại tuần T.
- POST /predict/dbn     : Dự báo P(At-risk) + Giải thích nhân quả bằng Mạng Bayesian Động 5 tầng.
- POST /predict/lstm    : Dự báo P(At-risk) + Đường cong xu hướng qua các tuần bằng Masked LSTM.
- POST /predict/hybrid  : Dự báo tổng hợp DSS (LSTM + DBN Explainability + Khuyến nghị cố vấn học tập).
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.schemas import (
    DBNPredictResponse,
    HybridDSSResponse,
    LSTMPredictResponse,
    StudentPredictRequest,
)
from src.api.service import DSSInferenceEngine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Khởi tạo sẵn Engine khi server bật để mọi request phản hồi tức thì (< 5ms)
    DSSInferenceEngine.get_instance()
    yield


app = FastAPI(
    title="OULAD Early-Warning DSS Inference API",
    description=(
        "Backend suy luận Hệ Hỗ trợ Ra quyết định (IS254.R11 - Nhóm 8) "
        "kết hợp Dynamic Bayesian Network (5-Tier DBN) và Masked LSTM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    engine = DSSInferenceEngine.get_instance()
    return {
        "status": "healthy",
        "service": "IS254.R11 Early-Warning DSS API",
        "models_loaded": ["DBN_5Tier (pgmpy)", "LSTM_Masked (PyTorch)"],
        "total_enrollments": len(engine.enrollments),
        "unique_students": int(engine.enrollments['id_student'].nunique()),
    }


@app.get("/meta/options")
def get_options():
    engine = DSSInferenceEngine.get_instance()
    return engine.get_metadata_options()


@app.get("/cohort/overview")
def cohort_overview(
    week_t: int = Query(default=8, ge=1, le=20),
    code_module: str = Query(default="ALL"),
    code_presentation: str = Query(default="ALL"),
    model_source: str = Query(default="hybrid", pattern="^(hybrid|lstm|dbn)$"),
    low_th: float = Query(default=0.30, ge=0.05, le=0.50),
    high_th: float = Query(default=0.60, ge=0.40, le=0.95),
    top_limit: int = Query(default=200, ge=10, le=1000),
):
    engine = DSSInferenceEngine.get_instance()
    return engine.get_cohort_overview(
        week_t=week_t,
        code_module=code_module,
        code_presentation=code_presentation,
        model_source=model_source,
        low_th=low_th,
        high_th=high_th,
        top_limit=top_limit,
    )


@app.post("/predict/dbn", response_model=DBNPredictResponse)
def predict_dbn_endpoint(req: StudentPredictRequest):
    engine = DSSInferenceEngine.get_instance()
    try:
        return engine.predict_dbn(
            id_student=req.id_student,
            code_module=req.code_module,
            code_presentation=req.code_presentation,
            week_t=req.week_t,
            low_th=req.low_threshold,
            high_th=req.high_threshold,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict/lstm", response_model=LSTMPredictResponse)
def predict_lstm_endpoint(req: StudentPredictRequest):
    engine = DSSInferenceEngine.get_instance()
    try:
        return engine.predict_lstm(
            id_student=req.id_student,
            code_module=req.code_module,
            code_presentation=req.code_presentation,
            week_t=req.week_t,
            low_th=req.low_threshold,
            high_th=req.high_threshold,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict/hybrid", response_model=HybridDSSResponse)
def predict_hybrid_endpoint(req: StudentPredictRequest):
    engine = DSSInferenceEngine.get_instance()
    try:
        return engine.predict_hybrid_dss(
            id_student=req.id_student,
            code_module=req.code_module,
            code_presentation=req.code_presentation,
            week_t=req.week_t,
            low_th=req.low_threshold,
            high_th=req.high_threshold,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
