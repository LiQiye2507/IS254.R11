"""
Pydantic Schemas kiểm tra dữ liệu đầu vào & đầu ra cho dịch vụ suy luận FastAPI (Mục 5.1 DSS_plan.docx)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class StudentPredictRequest(BaseModel):
    id_student: int = Field(..., description="Mã định danh sinh viên trong OULAD", examples=[74372])
    code_module: str = Field(..., description="Mã môn học (AAA..GGG)", examples=["AAA"])
    code_presentation: str = Field(..., description="Mã học kỳ (2013B, 2013J, 2014B, 2014J)", examples=["2013J"])
    week_t: int = Field(default=8, ge=1, le=20, description="Tuần dự đoán cảnh báo sớm T (1..20)")
    low_threshold: float = Field(default=0.30, ge=0.05, le=0.50)
    high_threshold: float = Field(default=0.60, ge=0.40, le=0.95)


class CausalFactorImpact(BaseModel):
    factor: str
    factor_label: str
    value: str
    delta_prob: float


class DBNPredictResponse(BaseModel):
    model: str = "DBN_5Tier"
    id_student: int
    code_module: str
    code_presentation: str
    week_t: int
    prob_at_risk: float
    risk_level: str
    top_factors: List[CausalFactorImpact]


class WeeklyTrendPoint(BaseModel):
    week: int
    prob_at_risk: float


class LSTMPredictResponse(BaseModel):
    model: str = "LSTM_Masked"
    id_student: int
    code_module: str
    code_presentation: str
    week_t: int
    prob_at_risk: float
    risk_level: str
    weekly_trend: List[WeeklyTrendPoint]


class AssessmentRecordItem(BaseModel):
    id_assessment: int
    assessment_type: str
    deadline_week: int
    weight: float
    status_at_week_t: str
    submission_week: Optional[int] = None
    score: Optional[float] = None


class HybridDSSResponse(BaseModel):
    id_student: int
    code_module: str
    code_presentation: str
    week_t: int
    primary_predictor: str
    prob_hybrid: float
    prob_lstm: float
    prob_dbn: float
    risk_level: str
    student_profile: dict
    weekly_trend: List[WeeklyTrendPoint]
    dbn_explanations: List[CausalFactorImpact]
    assessments_timeline: List[AssessmentRecordItem]
    advisor_recommendations: List[str]
