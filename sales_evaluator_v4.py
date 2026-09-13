# ==========================================
# Thành viên 4 - Phân Tích Mô Tả & Kiểm Định Dữ Liệu
# ==========================================
import pandas as pd
import numpy as np
import streamlit as st

class SalesMetricEvaluator:
    """
    Xử lý tính toán thống kê chi tiết, kiểm tra phân phối định lượng 
    và rà soát các điểm bất thường trên tập dữ liệu lớn.
    """
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset.copy()

    def compute_distribution_metrics(self) -> pd.DataFrame:
        """Tổng hợp các chỉ số trung tâm và phân tán của các trường tài chính."""
        target_fields = ['Amount', 'Boxes_Shipped', 'Marketing_Spend', 'Discount_Pct', 'Price_per_Box']
        active_fields = [field for field in target_fields if field in self.dataset.columns]
        
        if not active_fields:
            return pd.DataFrame()
        
        metrics_summary = self.dataset[active_fields].agg(['count', 'mean', 'median', 'std', 'min', 'max']).T
        metrics_summary.columns = ['Tổng mẫu', 'Trung bình', 'Trung vị', 'Độ lệch chuẩn', 'Min', 'Max']
        return metrics_summary

    def generate_segment_breakdown(self, category_col: str, metric_col: str = 'Amount') -> pd.DataFrame:
        """Thống kê tổng hợp theo từng phân khúc kinh doanh cụ thể."""
        if category_col in self.dataset.columns and metric_col in self.dataset.columns:
            breakdown_table = self.dataset.pivot_table(
                index=category_col,
                values=metric_col,
                aggfunc=['sum', 'mean', 'count']
            )
            breakdown_table.columns = ['Tổng giá trị', 'Giá trị trung bình', 'Tần suất xuất hiện']
            return breakdown_table.sort_values(by='Tổng giá trị', ascending=False)
        return pd.DataFrame()

    def filter_statistical_outliers(self, evaluation_col: str = 'Amount') -> pd.DataFrame:
        """Nhận diện bản ghi ngoại lai thông qua phương pháp khoảng tứ phân vị (IQR)."""
        if evaluation_col not in self.dataset.columns:
            return pd.DataFrame()
            
        numeric_series = pd.to_numeric(self.dataset[evaluation_col], errors='coerce')
        q_first = numeric_series.quantile(0.25)
        q_third = numeric_series.quantile(0.75)
        iqr_value = q_third - q_first
        
        lower_bound = q_first - 1.5 * iqr_value
        upper_bound = q_third + 1.5 * iqr_value
        
        outliers_subset = self.dataset[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
        return outliers_subset

# ==========================================
# GIAO DIỆN TRỰC TUYẾN STREAMLIT (GÓC THÀNH VIÊN 4)
# ==========================================
def render_tab_4(df_clean: pd.DataFrame):
    st.header("Tab 4: Thống Kê Định Lượng & Kiểm Định Phân Phối (Quy mô 200.000 dòng)")
    
    if df_clean is None or df_clean.empty:
        st.warning("Dữ liệu nguồn chưa sẵn sàng. Vui lòng kiểm tra lại quá trình thiết lập ở Tab 1.")
        return

    evaluator = SalesMetricEvaluator(df_clean)

    st.subheader("1. Bảng Thống Kê Mô Tả Định Lượng Tổng Quát")
    summary_table = evaluator.compute_distribution_metrics()
    if not summary_table.empty:
        st.dataframe(summary_table.style.format("{:.2f}"))
    else:
        st.info("Không có trường dữ liệu phù hợp để thống kê.")

    st.markdown("---")
    st.subheader("2. Phân Tích Tổng Hợp Theo Tiêu Chí Đa Chiều")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        selected_category = st.selectbox(
            "Chọn nhóm phân loại:", 
            ['Country', 'Product', 'Channel', 'Salesperson'], 
            key="member4_category_selector"
        )
    with col_input2:
        selected_metric = st.selectbox(
            "Chọn chỉ tiêu phân tích:", 
            ['Amount', 'Boxes_Shipped', 'Marketing_Spend'], 
            key="member4_metric_selector"
        )

    if selected_category and selected_metric:
        segment_result = evaluator.generate_segment_breakdown(selected_category, selected_metric)
        st.dataframe(segment_result.style.format("{:,.2f}"))

    st.markdown("---")
    st.subheader("3. Kiểm Tra Giá Trị Ngoại Lai (IQR Method)")
    target_column = st.selectbox(
        "Chọn trường dữ liệu kiểm tra biên độ:", 
        ['Amount', 'Boxes_Shipped', 'Marketing_Spend'], 
        key="member4_outlier_selector"
    )
    
    if target_column:
        outliers_df = evaluator.filter_statistical_outliers(target_column)
        st.metric(
            label=f"Tổng số bản ghi vượt ngưỡng ngoại lai ({target_column})", 
            value=f"{len(outliers_df):,}"
        )
        with st.expander("Hiển thị chi tiết danh sách bản ghi ngoại lai"):
            st.dataframe(outliers_df.head(50))
