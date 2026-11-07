"""
Giao diện Bảng điều khiển Hệ Hỗ trợ Ra quyết định (Streamlit DSS Dashboard - IS254.R11)
---------------------------------------------------------------------------------------
Triển khai đầy đủ 3 phân hệ nghiệp vụ trong Mục 5.3 Đề cương (`DSS_plan.docx`) + Tab Benchmark:
1. Tổng quan rủi ro học phần (Low / Medium / High theo môn học, học kỳ & tuần dự báo T).
2. Danh sách cảnh báo sớm (Early-Warning Watchlist ưu tiên liên hệ cố vấn + xuất CSV).
3. Phân tích cá nhân & Giải thích nhân quả (Đường cong xu hướng tuần từ LSTM + Giải thích Delta P từ DBN
   + Lịch sử bài đánh giá + Khuyến nghị can thiệp sư phạm tự động).
4. Đối sánh hiệu năng thực nghiệm 5-Fold (DBN vs LSTM vs Baseline qua các tuần T).

Hỗ trợ kết nối tự động qua FastAPI REST API (`DSS_API_URL`, mặc định `http://127.0.0.1:8000`)
và tự động chuyển sang chế độ Direct Engine nếu chưa bật server API riêng biệt.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.service import DSSInferenceEngine

st.set_page_config(
    page_title="DSS Cảnh Báo Sớm Học Tập (IS254.R11)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.environ.get("DSS_API_URL", "http://127.0.0.1:8000").rstrip("/")


# ==============================================================================
# DATA / API CLIENT ADAPTER (HỖ TRỢ DUAL-MODE: FASTAPI HOẶC DIRECT ENGINE)
# ==============================================================================
@st.cache_resource(show_spinner="Đang nạp mô hình DBN 5 tầng & Masked LSTM vào bộ nhớ...")
def get_local_engine() -> DSSInferenceEngine:
    return DSSInferenceEngine.get_instance()


def check_api_online() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=0.8)
        return r.status_code == 200
    except Exception:
        return False


def fetch_meta(use_api: bool) -> dict:
    if use_api:
        r = requests.get(f"{API_BASE_URL}/meta/options", timeout=5)
        r.raise_for_status()
        return r.json()
    return get_local_engine().get_metadata_options()


def fetch_cohort(
    use_api: bool,
    week_t: int,
    code_module: str,
    code_presentation: str,
    model_source: str,
    low_th: float,
    high_th: float,
    top_limit: int = 300,
) -> dict:
    if use_api:
        r = requests.get(
            f"{API_BASE_URL}/cohort/overview",
            params={
                "week_t": week_t,
                "code_module": code_module,
                "code_presentation": code_presentation,
                "model_source": model_source,
                "low_th": low_th,
                "high_th": high_th,
                "top_limit": top_limit,
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    return get_local_engine().get_cohort_overview(
        week_t=week_t,
        code_module=code_module,
        code_presentation=code_presentation,
        model_source=model_source,
        low_th=low_th,
        high_th=high_th,
        top_limit=top_limit,
    )


def fetch_student_hybrid(
    use_api: bool,
    id_student: int,
    code_module: str,
    code_presentation: str,
    week_t: int,
    low_th: float,
    high_th: float,
) -> dict:
    payload = {
        "id_student": int(id_student),
        "code_module": str(code_module),
        "code_presentation": str(code_presentation),
        "week_t": int(week_t),
        "low_threshold": float(low_th),
        "high_threshold": float(high_th),
    }
    if use_api:
        r = requests.post(f"{API_BASE_URL}/predict/hybrid", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    return get_local_engine().predict_hybrid_dss(
        id_student=id_student,
        code_module=code_module,
        code_presentation=code_presentation,
        week_t=week_t,
        low_th=low_th,
        high_th=high_th,
    )


# ==============================================================================
# SIDEBAR CẤU HÌNH QUYẾT ĐỊNH CỦA CỐ VẤN HỌC TẬP
# ==============================================================================
api_online = check_api_online()
meta = fetch_meta(api_online)

with st.sidebar:
    st.markdown("## 🎓 Hệ DSS Cảnh Báo Sớm")
    st.caption("Môn học: **IS254.R11** – Nhóm 8 (UIT)")
    if api_online:
        st.success(f"🟢 Kết nối FastAPI (`{API_BASE_URL}`)")
    else:
        st.info("🔵 Chế độ Direct Engine (Tự động nạp nội bộ)")

    st.divider()
    st.markdown("### 1. Bộ lọc Học phần & Thời điểm")

    mod_options = ["ALL"] + sorted(list(meta["modules_presentations"].keys()))
    sel_module = st.selectbox("Chọn Môn học (Module):", mod_options, index=0)

    if sel_module == "ALL":
        pres_options = ["ALL", "2013B", "2013J", "2014B", "2014J"]
    else:
        pres_options = ["ALL"] + meta["modules_presentations"][sel_module]
    sel_pres = st.selectbox("Chọn Học kỳ (Presentation):", pres_options, index=0)

    sel_week = st.slider(
        "Mốc Tuần dự báo hiện tại (T):",
        min_value=1,
        max_value=int(meta["max_week"]),
        value=8,
        step=1,
        help="Hệ thống chỉ sử dụng dữ liệu phát sinh từ tuần 1 đến hết tuần T (không rò rỉ tương lai).",
    )

    st.divider()
    st.markdown("### 2. Tham số Mô hình & Ngưỡng Rủi ro")
    model_choice = st.radio(
        "Cơ chế dự báo chính:",
        options=["hybrid", "lstm", "dbn"],
        format_func=lambda x: {
            "hybrid": "⚡ Hybrid DSS (Khuyến nghị: LSTM + DBN)",
            "lstm": "📈 Masked LSTM (Chuỗi thời gian)",
            "dbn": "🕸️ 5-Tier DBN (Mạng xác suất Bayes)",
        }[x],
        index=0,
    )

    low_th, high_th = st.slider(
        "Ngưỡng phân tầng Rủi ro (Low / Medium / High):",
        min_value=0.10,
        max_value=0.90,
        value=(0.30, 0.60),
        step=0.05,
        help="Mặc định theo đề cương: Low < 0.30 | Medium: 0.30–0.60 | High >= 0.60",
    )


# ==============================================================================
# HEADER & SỐ LIỆU KPI TỔNG QUAN
# ==============================================================================
st.title("🎓 Hệ Hỗ Trợ Ra Quyết Định Cảnh Báo Sớm Nguy Cơ Học Tập")
st.markdown(
    f"**Phạm vi đang xem:** Môn `{sel_module}` | Học kỳ `{sel_pres}` | "
    f"**Thời điểm đánh giá:** Tuần `T = {sel_week}` | "
    f"**Ngưỡng High-Risk:** `P(At-risk) ≥ {high_th:.2f}`"
)

cohort_data = fetch_cohort(
    use_api=api_online,
    week_t=sel_week,
    code_module=sel_module,
    code_presentation=sel_pres,
    model_source=model_choice,
    low_th=low_th,
    high_th=high_th,
    top_limit=500,
)

total_stu = cohort_data["total_students"]
tiers = cohort_data["tier_counts"]
n_high = tiers.get("High", 0)
n_med = tiers.get("Medium", 0)
n_low = tiers.get("Low", 0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("👥 Tổng lượt Sinh viên", f"{total_stu:,}")
k2.metric(
    "🔴 Nguy cơ Cao (High Risk)",
    f"{n_high:,}",
    delta=f"{(n_high / max(total_stu, 1)) * 100:.1f}% cần can thiệp ngay",
    delta_color="inverse",
)
k3.metric(
    "🟠 Nguy cơ Trung bình (Medium)",
    f"{n_med:,}",
    delta=f"{(n_med / max(total_stu, 1)) * 100:.1f}% cần theo dõi",
    delta_color="off",
)
k4.metric(
    "🟢 An toàn (Low Risk)",
    f"{n_low:,}",
    delta=f"{(n_low / max(total_stu, 1)) * 100:.1f}% ổn định",
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Tổng quan Rủi ro Học phần",
    "🚨 2. Danh sách Cảnh báo Sớm (Watchlist)",
    "🔍 3. Chẩn đoán & Giải thích Nhân quả Cá nhân",
    "📈 4. Thực nghiệm Mô hình (5-Fold Benchmark)",
])


# ==============================================================================
# TAB 1: COHORT RISK OVERVIEW
# ==============================================================================
with tab1:
    c_left, c_right = st.columns([1, 1.45])

    with c_left:
        st.subheader(f"Tỷ trọng Phân tầng Rủi ro tại Tuần T = {sel_week}")
        fig_pie, ax_pie = plt.subplots(figsize=(5.2, 4.2))
        vals = [n_low, n_med, n_high]
        lbls = [f"Low (<{low_th:.2f})", f"Medium ({low_th:.2f}-{high_th:.2f})", f"High (≥{high_th:.2f})"]
        clrs = ["#2ecc71", "#f39c12", "#e74c3c"]
        if sum(vals) > 0:
            wedges, texts, autotexts = ax_pie.pie(
                vals, labels=lbls, autopct="%1.1f%%", startangle=140,
                colors=clrs, wedgeprops={"edgecolor": "white", "linewidth": 1.5}
            )
            for at in autotexts:
                at.set_color("white")
                at.set_fontweight("bold")
        ax_pie.set_title(f"Phân bố {total_stu:,} lượt đăng ký", fontweight="bold")
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with c_right:
        st.subheader("Phân bố Mức độ Rủi ro theo Môn học (Module)")
        mod_df = pd.DataFrame(cohort_data["by_module"])
        if not mod_df.empty:
            fig_bar, ax_bar = plt.subplots(figsize=(7.2, 4.2))
            x = np.arange(len(mod_df))
            b1 = ax_bar.bar(x, mod_df["Low"], label="Low", color="#2ecc71")
            b2 = ax_bar.bar(x, mod_df["Medium"], bottom=mod_df["Low"], label="Medium", color="#f39c12")
            b3 = ax_bar.bar(
                x, mod_df["High"],
                bottom=mod_df["Low"] + mod_df["Medium"],
                label="High", color="#e74c3c"
            )
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(mod_df["code_module"])
            ax_bar.set_ylabel("Số lượt sinh viên")
            ax_bar.set_xlabel("Mã môn học")
            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)
            ax_bar.legend(frameon=True)
            st.pyplot(fig_bar)
            plt.close(fig_bar)


# ==============================================================================
# TAB 2: EARLY-WARNING WATCHLIST
# ==============================================================================
with tab2:
    st.subheader(f"🚨 Danh sách Sinh viên Ưu tiên Can thiệp Sớm (Tuần T = {sel_week})")
    watch_df = pd.DataFrame(cohort_data["watchlist"])

    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        only_high = st.checkbox("Chỉ hiển thị mức High Risk", value=False)
    with f_col2:
        filter_missed = st.checkbox("Chỉ lọc sinh viên đã BỎ NỘP bài quá hạn (missed_count > 0)", value=False)

    if not watch_df.empty:
        view_df = watch_df.copy()
        if only_high:
            view_df = view_df[view_df["risk_level"] == "High"]
        if filter_missed:
            view_df = view_df[view_df["missed_count"] > 0]

        st.caption(f"Đang hiển thị **{len(view_df):,}** sinh viên sắp xếp từ nguy cơ cao nhất xuống thấp nhất.")
        display_cols = {
            "id_student": "Mã SV",
            "code_module": "Môn",
            "code_presentation": "Học kỳ",
            "risk_level": "Mức Rủi ro",
            "prob_at_risk": "P(At-risk) Chung",
            "prob_lstm": "P(LSTM)",
            "prob_dbn": "P(DBN)",
            "submitted_count": "Số bài đã nộp",
            "missed_count": "Số bài bỏ quá hạn",
            "late_count": "Số bài nộp trễ",
            "weighted_score": "Điểm TB trọng số",
            "num_of_prev_attempts": "Số lần học lại",
        }
        st.dataframe(
            view_df[list(display_cols.keys())].rename(columns=display_cols),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = view_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Xuất danh sách Cảnh báo Sớm (CSV) cho Cố vấn Học tập",
            data=csv_bytes,
            file_name=f"DSS_Watchlist_Week{sel_week}_{sel_module}_{sel_pres}.csv",
            mime="text/csv",
        )
    else:
        st.warning("Không có sinh viên nào khớp điều kiện lọc.")


# ==============================================================================
# TAB 3: INDIVIDUAL STUDENT DIAGNOSIS (LSTM TREND + DBN CAUSAL EXPLANATION)
# ==============================================================================
with tab3:
    st.subheader("🔍 Hồ sơ Chẩn đoán Sâu & Giải thích Nhân quả cho Từng Sinh viên")
    if not watch_df.empty:
        options_records = watch_df.head(150).to_dict(orient="records")
        selected_rec = st.selectbox(
            "Chọn sinh viên từ danh sách ưu tiên để phân tích chi tiết:",
            options=options_records,
            format_func=lambda r: (
                f"SV #{r['id_student']} | Môn {r['code_module']}-{r['code_presentation']} | "
                f"Rủi ro: {r['risk_level']} ({r['prob_at_risk']*100:.1f}%) | "
                f"Bỏ bài: {int(r['missed_count'])} | Điểm TB: {r['weighted_score']}"
            ),
        )

        detail = fetch_student_hybrid(
            use_api=api_online,
            id_student=int(selected_rec["id_student"]),
            code_module=str(selected_rec["code_module"]),
            code_presentation=str(selected_rec["code_presentation"]),
            week_t=sel_week,
            low_th=low_th,
            high_th=high_th,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Xác suất Rủi ro Hybrid DSS", f"{detail['prob_hybrid']*100:.1f}%", detail["risk_level"])
        m2.metric("📈 Dự báo Chuỗi LSTM", f"{detail['prob_lstm']*100:.1f}%")
        m3.metric("🕸️ Dự báo Mạng Bayes DBN", f"{detail['prob_dbn']*100:.1f}%")
        m4.metric(
            "📚 Điểm tích lũy đến Tuần T",
            f"{detail['student_profile']['weighted_score_by_T']}/100",
            f"Bỏ quá hạn: {detail['student_profile']['missed_by_T']} bài",
            delta_color="inverse" if detail['student_profile']['missed_by_T'] > 0 else "normal",
        )

        st.markdown("#### 💡 Khuyến nghị Can thiệp Sư phạm cho Cố vấn Học tập")
        for rec_text in detail["advisor_recommendations"]:
            if "🚨" in rec_text or "📉" in rec_text:
                st.error(rec_text)
            elif "⚠️" in rec_text or "⏰" in rec_text or "🔄" in rec_text:
                st.warning(rec_text)
            else:
                st.success(rec_text)

        ch_col1, ch_col2 = st.columns(2)

        with ch_col1:
            st.markdown(f"#### 📈 Quỹ đạo Xác suất Rủi ro Tuần 1 → {sel_week} (Masked LSTM)")
            trend_df = pd.DataFrame(detail["weekly_trend"])
            fig_tr, ax_tr = plt.subplots(figsize=(6.4, 4.0))
            ax_tr.plot(
                trend_df["week"], trend_df["prob_at_risk"] * 100,
                marker="o", linewidth=2.3, color="#2980b9", label="LSTM P(At-risk) %"
            )
            ax_tr.axhline(high_th * 100, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Ngưỡng High ({high_th*100:.0f}%)")
            ax_tr.axhline(low_th * 100, color="#27ae60", linestyle=":", linewidth=1.2, label=f"Ngưỡng Low ({low_th*100:.0f}%)")
            ax_tr.set_xlim(1, max(sel_week, 2))
            ax_tr.set_ylim(0, 102)
            ax_tr.set_xlabel("Tuần học (Week)")
            ax_tr.set_ylabel("Xác suất Rủi ro (%)")
            ax_tr.spines["top"].set_visible(False)
            ax_tr.spines["right"].set_visible(False)
            ax_tr.grid(True, linestyle=":", alpha=0.5)
            ax_tr.legend(loc="best", fontsize=8.5)
            st.pyplot(fig_tr)
            plt.close(fig_tr)

        with ch_col2:
            st.markdown(f"#### 🕸️ Giải thích Nhân quả các Yếu tố Ảnh hưởng (5-Tier DBN)")
            exp_items = detail["dbn_explanations"]
            if exp_items:
                labels_y = [f"{it['factor_label']} = {it['value']}" for it in exp_items][::-1]
                deltas_x = [it["delta_prob"] * 100 for it in exp_items][::-1]
                bar_clrs = ["#e74c3c" if d > 0 else "#27ae60" for d in deltas_x]

                fig_dbn, ax_dbn = plt.subplots(figsize=(6.4, 4.0))
                bars = ax_dbn.barh(labels_y, deltas_x, color=bar_clrs, height=0.55)
                ax_dbn.axvline(0, color="black", linewidth=1)
                ax_dbn.set_xlabel("Độ dịch chuyển rủi ro ΔP (%) [Đỏ: Tăng rủi ro | Xanh: Giảm rủi ro]")
                lim_min = min(min(deltas_x), 0) - 10
                lim_max = max(max(deltas_x), 0) + 12
                ax_dbn.set_xlim(lim_min, lim_max)
                ax_dbn.spines["top"].set_visible(False)
                ax_dbn.spines["right"].set_visible(False)
                for b, d in zip(bars, deltas_x):
                    off = 0.8 if d >= 0 else -0.8
                    ha = "left" if d >= 0 else "right"
                    ax_dbn.text(d + off, b.get_y() + b.get_height() / 2, f"{d:+.1f}%", va="center", ha=ha, fontsize=8.5, fontweight="bold")
                st.pyplot(fig_dbn)
                plt.close(fig_dbn)

        p_col1, p_col2 = st.columns([1, 1.6])
        with p_col1:
            st.markdown("#### 👤 Hồ sơ Nhân khẩu & Nền tảng")
            prof = detail["student_profile"]
            prof_table = pd.DataFrame([
                ("Giới tính", prof["gender"]),
                ("Nhóm tuổi", prof["age_band"]),
                ("Trình độ đầu vào", prof["highest_education"]),
                ("Chỉ số kinh tế (IMD)", prof["imd_band"]),
                ("Khuyết tật", prof["disability"]),
                ("Số lần học lại", str(prof["num_of_prev_attempts"])),
                ("Số tín chỉ đăng ký", str(prof["studied_credits"])),
                ("Kết quả thật cuối kỳ (Đối chiếu)", prof["actual_final_result"]),
            ], columns=["Thuộc tính", "Giá trị"])
            st.dataframe(prof_table, use_container_width=True, hide_index=True)

        with p_col2:
            st.markdown(f"#### 📝 Tiến độ Bài đánh giá của Môn học tính đến Tuần T = {sel_week}")
            tl_df = pd.DataFrame(detail["assessments_timeline"]).rename(columns={
                "id_assessment": "Mã bài thi",
                "assessment_type": "Loại bài",
                "deadline_week": "Tuần hạn nộp",
                "weight": "Trọng số (%)",
                "status_at_week_t": f"Trạng thái tại Tuần {sel_week}",
                "submission_week": "Tuần thực nộp",
                "score": "Điểm đạt được",
            })
            st.dataframe(tl_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 4: 5-FOLD MODEL BENCHMARK VIEWER
# ==============================================================================
with tab4:
    st.subheader("📈 Kết quả Thực nghiệm Kiểm định chéo 5-Fold (StratifiedGroupKFold theo Sinh viên)")
    sum_csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "benchmark_summary.csv")
    if os.path.exists(sum_csv_path):
        bm_df = pd.read_csv(sum_csv_path)
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            fig_auc, ax_auc = plt.subplots(figsize=(6.2, 4.0))
            for m_code, lbl, clr in [
                ("DBN", "DBN (5-Tier)", "#e74c3c"),
                ("LSTM", "LSTM (Masked)", "#2980b9"),
                ("Baseline_RF", "Baseline (RF)", "#27ae60"),
                ("Baseline_LR", "Baseline (LR)", "#7f8c8d"),
            ]:
                s = bm_df[bm_df["model"] == m_code].sort_values("week")
                ax_auc.plot(s["week"], s["auc_roc_mean"], marker="o", linewidth=2, color=clr, label=lbl)
            ax_auc.set_title("Đường cong AUC-ROC theo Tuần dự báo T", fontweight="bold")
            ax_auc.set_xlabel("Tuần T")
            ax_auc.set_ylabel("AUC-ROC")
            ax_auc.set_xticks([2, 4, 8, 12, 16, 20])
            ax_auc.grid(True, linestyle=":", alpha=0.5)
            ax_auc.legend()
            st.pyplot(fig_auc)
            plt.close(fig_auc)

        with b_col2:
            fig_f1, ax_f1 = plt.subplots(figsize=(6.2, 4.0))
            for m_code, lbl, clr in [
                ("DBN", "DBN (5-Tier)", "#e74c3c"),
                ("LSTM", "LSTM (Masked)", "#2980b9"),
                ("Baseline_RF", "Baseline (RF)", "#27ae60"),
                ("Baseline_LR", "Baseline (LR)", "#7f8c8d"),
            ]:
                s = bm_df[bm_df["model"] == m_code].sort_values("week")
                ax_f1.plot(s["week"], s["f1_mean"], marker="s", linewidth=2, color=clr, label=lbl)
            ax_f1.set_title("Đường cong F1-Score theo Tuần dự báo T", fontweight="bold")
            ax_f1.set_xlabel("Tuần T")
            ax_f1.set_ylabel("F1-Score")
            ax_f1.set_xticks([2, 4, 8, 12, 16, 20])
            ax_f1.grid(True, linestyle=":", alpha=0.5)
            ax_f1.legend()
            st.pyplot(fig_f1)
            plt.close(fig_f1)

        st.markdown("#### Bảng tổng hợp số liệu Benchmark trên 6 mốc tuần")
        pretty_bm = pd.DataFrame({
            "Tuần (T)": bm_df["week"],
            "Mô hình": bm_df["model"],
            "Accuracy": bm_df.apply(lambda r: f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}", axis=1),
            "Precision": bm_df.apply(lambda r: f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f}", axis=1),
            "Recall": bm_df.apply(lambda r: f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f}", axis=1),
            "F1-Score": bm_df.apply(lambda r: f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}", axis=1),
            "AUC-ROC": bm_df.apply(lambda r: f"{r['auc_roc_mean']:.4f} ± {r['auc_roc_std']:.4f}", axis=1),
            "AUC-PR": bm_df.apply(lambda r: f"{r['auc_pr_mean']:.4f} ± {r['auc_pr_std']:.4f}", axis=1),
        })
        st.dataframe(pretty_bm, use_container_width=True, hide_index=True)
