"""
Mô-đun dịch vụ API suy luận thời gian thực xây dựng bằng FastAPI (`main.py`, `service.py`, `schemas.py`)
"""

from .service import DSSInferenceEngine

__all__ = ["DSSInferenceEngine"]
