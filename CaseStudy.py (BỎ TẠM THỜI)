# ==========================================
# Thành viên 1 - Nhóm trưởng
# ==========================================

import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def process_data(self) -> pd.DataFrame:
        """
        Hàm thực hiện Trích xuất, Định dạng và Chắt lọc 200.000 dòng dữ liệu
        """
        print("Đang đọc và xử lý dữ liệu...")
        
        # ==========================================
        # BƯỚC 1: TRÍCH XUẤT (Extract)
        # ==========================================
        # Đọc file Excel thô
        raw_df = pd.read_excel(self.filepath)
        
        # Vì dữ liệu đang dồn vào 1 cột, ta cần tách nó ra thành nhiều cột
        col_name = raw_df.columns[0]  # Lấy tên của cột duy nhất đó
        
        # Tách dữ liệu bằng dấu phẩy (',') để tạo thành bảng hoàn chỉnh
        self.df = raw_df[col_name].str.split(',', expand=True)
        self.df.columns = col_name.split(',') # Gán lại tên cột chuẩn

        # ==========================================
        # BƯỚC 2: ĐỊNH DẠNG (Format)
        # ==========================================
        # Mặc định sau khi tách, mọi thứ là chuỗi (string). Phải ép kiểu!
        
        # Ép kiểu ngày tháng
        self.df['Order_Date'] = pd.to_datetime(self.df['Order_Date'], errors='coerce')
        
        # Ép kiểu số thực (float) cho các cột tài chính
        numeric_cols = ['Discount_Pct', 'Price_per_Box', 'Marketing_Spend', 'Boxes_Shipped', 'Amount']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Thêm các cột định dạng mới (Ví dụ: tách Năm và Tháng để dễ thống kê sau này)
        self.df['Year'] = self.df['Order_Date'].dt.year
        self.df['Month'] = self.df['Order_Date'].dt.month

        # ==========================================
        # BƯỚC 3: CHẮT LỌC & LÀM SẠCH (Clean)
        # ==========================================
        # 1. Bỏ qua các dòng bị thiếu dữ liệu quan trọng (ví dụ: ngày bán và doanh thu)
        self.df = self.df.dropna(subset=['Order_Date', 'Amount'])
        
        # 2. Xử lý dữ liệu nhiễu/vô lý: Số lượng hộp bán ra bị âm
        # Lọc để chỉ giữ lại các đơn hàng có Boxes_Shipped > 0
        self.df = self.df[self.df['Boxes_Shipped'] > 0]
        
        # 3. Điền giá trị trung bình cho các cột bị khuyết (Ví dụ: Chi phí Marketing)
        mean_marketing = self.df['Marketing_Spend'].mean()
        self.df['Marketing_Spend'] = self.df['Marketing_Spend'].fillna(mean_marketing)

        print(f"Xử lý thành công! Số dòng dữ liệu sạch còn lại: {len(self.df)}")
        return self.df

# ==========================================
# CÁCH CHẠY THỬ TRONG NHÓM
# ==========================================
if __name__ == "__main__":
    loader = DataLoader("Chocolate_Sales.xlsx")
    clean_table = loader.process_data()
    
    # In thử 5 dòng đầu tiên của bảng Table hoàn chỉnh
    print("\nBảng dữ liệu hoàn chỉnh:")
    print(clean_table.head())


# ==========================================
# THÀNH VIÊN 5
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class DataPredictor:
    def __init__(self):
        """Khởi tạo lớp dự đoán với mô hình Hồi quy tuyến tính."""
        self.model = LinearRegression()
        self.r2_score = None
        self.mse_score = None
        self.is_trained = False

    def train_model(self, df):
        """
        Huấn luyện mô hình từ DataFrame (áp dụng cho toàn bộ 200.000 dòng).
        Biến độc lập (X): Boxes_Shipped, Price_per_Box, Marketing_Spend
        Biến phụ thuộc (y): Amount
        """
        # Kiểm tra các cột cần thiết
        features = ['Boxes_Shipped', 'Price_per_Box', 'Marketing_Spend']
        if not all(col in df.columns for col in features + ['Amount']):
            raise ValueError("DataFrame thiếu các cột cần thiết (Boxes_Shipped, Price_per_Box, Marketing_Spend, Amount).")

        # Loại bỏ các dòng bị khuyết trên toàn bộ dữ liệu để tránh lỗi
        df_clean = df.dropna(subset=features + ['Amount'])

        X = df_clean[features]
        y = df_clean['Amount']

        # Chia tập dữ liệu: 80% (khoảng 160.000 dòng) để huấn luyện, 20% (khoảng 40.000 dòng) để kiểm thử
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Huấn luyện mô hình trên 160.000 dòng
        self.model.fit(X_train, y_train)
        
        # Đánh giá trên 40.000 dòng Test
        y_pred = self.model.predict(X_test)

        self.r2_score = r2_score(y_test, y_pred)
        self.mse_score = mean_squared_error(y_test, y_pred)
        self.is_trained = True

        return self.r2_score, self.mse_score

    def predict_revenue(self, boxes, price, marketing):
        """Dự đoán doanh thu dựa trên thông số đầu vào."""
        if not self.is_trained:
            raise Exception("Mô hình chưa được huấn luyện.")
        
        input_data = np.array([[boxes, price, marketing]])
        prediction = self.model.predict(input_data)
        return prediction[0]
        
         st.markdown("---")
    with st.expander("Bảo vệ phương án: Tại sao chọn Linear Regression?"):
        # TODO: Ghi chú các gạch đầu dòng lý luận
        st.write("- Lý do không dùng If/Else...")
        st.write("- Lý do không dùng Deep Learning...")
        st.write("- Ưu điểm của Linear Regression với dữ liệu này...")
# ==========================================
# GIAO DIỆN UI
# ==========================================
import streamlit as st
def render_tab_5(df_clean):
    st.header("Tab 5: Dự đoán Doanh thu (Machine Learning)")
    
    # ==========================================
    # PHẦN 1: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH
    # ==========================================
    st.subheader("1. Đánh giá mô hình (Trên toàn bộ dữ liệu sạch)")
    
    # TODO: Khởi tạo class DataPredictor và gọi hàm train_model(df_clean) ở đây
    # Ví dụ: 
    # predictor = DataPredictor()
    # r2, mse = predictor.train_model(df_clean)
    
    # Khung hiển thị kết quả R2 và MSE
    col1, col2 = st.columns(2)
    col1.metric("R² Score", "[Giá trị R2 sẽ hiện ở đây]")
    col2.metric("MSE", "[Giá trị MSE sẽ hiện ở đây]")
    
    st.markdown("---")
    
    # ==========================================
    # PHẦN 2: FORM NHẬP LIỆU DỰ ĐOÁN
    # ==========================================
    st.subheader("2. Công cụ Mô phỏng (Simulator)")
    
    # Tạo 3 cột trống làm sườn cho 3 ô nhập liệu
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        # TODO: Tạo st.number_input cho Boxes_Shipped
        st.write("[Ô nhập Số lượng hộp]")
        
    with col_in2:
        # TODO: Tạo st.number_input cho Price_per_Box
        st.write("[Ô nhập Giá/Hộp]")
        
    with col_in3:
        # TODO: Tạo st.number_input cho Marketing_Spend
        st.write("[Ô nhập Chi phí Marketing]")
    if st.button("Chạy Dự đoán"):
        # TODO: Lấy dữ liệu từ 3 ô input trên truyền vào predictor.predict_revenue()
        # In ra kết quả
        st.success("Doanh thu dự kiến: [$ Kết quả tính toán]")

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
