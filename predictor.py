#=====================
# THÀNH VIÊN 5
#=====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class DataPredictor:
    """
    Class DataPredictor chịu trách nhiệm dự đoán Doanh thu (Amount) dựa trên dữ liệu.
    
    =================================================================================
    [BẢO VỆ PHƯƠNG ÁN CHỌN MÔ HÌNH]:
    1. Không dùng luật cố định (If/Else): 
       - If/else cố định không có khả năng tự học hỏi từ lịch sử dữ liệu (200.000 dòng),
         không tự tối ưu hóa trọng số khi có biến số mới.
    2. Không dùng Deep Learning (Mạng nơ-ron): 
       - Quá nặng nề, tốn tài nguyên tính toán không cần thiết, khó giải thích (black-box),
         và không phù hợp cho dạng dữ liệu bảng đơn giản.
    3. Lý do chọn Hồi quy tuyến tính (Linear Regression - Machine Learning truyền thống):
       - Tối ưu, huấn luyện cực nhanh trên 200.000 dòng.
       - Dễ giải thích toán học (thông qua chỉ số R2, MSE).
       - Hoàn toàn phù hợp cho bài toán dự đoán giá trị liên tục (Amount).
    =================================================================================
    """
    def __init__(self):
        self.model = LinearRegression()
        self.r2_score = None
        self.mse_score = None
        self.is_trained = False

    def train_model(self, df):
        """
        Huấn luyện mô hình Hồi quy tuyến tính trên toàn bộ dữ liệu (200.000 dòng).
        
        Input:
            df (DataFrame): Tập dữ liệu 200.000 dòng đã qua xử lý làm sạch.
        Output:
            r2_score (float), mse_score (float)
        """
        features = ['Boxes_Shipped', 'Price_per_Box', 'Marketing_Spend']
        target = 'Amount'

        # Kiểm tra sự tồn tại của các cột dữ liệu
        if not all(col in df.columns for col in features + [target]):
            raise ValueError(f"Dữ liệu truyền vào thiếu các cột bắt buộc: {features} hoặc {target}")

        # Làm sạch các dòng NaN nếu có trước khi đưa vào mô hình
        df_clean = df.dropna(subset=features + [target])

        X = df_clean[features]
        y = df_clean[target]

        # Chia dữ liệu: 80% Train (~160.000 dòng), 20% Test (~40.000 dòng)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Huấn luyện mô hình
        self.model.fit(X_train, y_train)

        # Dự đoán trên tập Test để tính độ đo đánh giá
        y_pred = self.model.predict(X_test)

        # Tính chỉ số R2 và MSE
        self.r2_score = r2_score(y_test, y_pred)
        self.mse_score = mean_squared_error(y_test, y_pred)
        self.is_trained = True

        return self.r2_score, self.mse_score

    def predict_revenue(self, boxes, price, marketing):
        """
        Dự đoán Doanh thu (Amount) dựa trên thông số người dùng nhập từ UI.
        """
        if not self.is_trained:
            raise Exception("Mô hình chưa được huấn luyện. Vui lòng gọi hàm train_model() trước.")

        input_data = np.array([[boxes, price, marketing]])
        prediction = self.model.predict(input_data)
        return prediction[0]
#===============================
# GIAO DIỆN UI
#===============================
import streamlit as st
from src.predictor import DataPredictor

def render_tab_5(df_clean):
    st.header("📈 Tab 5: Dự đoán Doanh thu (Machine Learning)")
    st.caption("Huấn luyện mô hình Hồi quy tuyến tính trên toàn bộ 200.000 dòng dữ liệu")

    # -------------------------------------------------------------------------
    # KHU VỰC 1: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH (R2 & MSE)
    # -------------------------------------------------------------------------
    st.subheader("1. Đánh giá độ chính xác của mô hình")
    
    predictor = DataPredictor()
    
    try:
        # Huấn luyện mô hình trên dữ liệu 200k dòng
        r2, mse = predictor.train_model(df_clean)
        
        # Sườn hiển thị KPI chỉ số đánh giá
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric(label="Chỉ số R² (R-Squared Score)", value=f"{r2:.4f}")
        with col_metric2:
            st.metric(label="Sai số MSE (Mean Squared Error)", value=f"{mse:,.2f}")
            
    except Exception as e:
        st.error(f"Chưa thể huấn luyện mô hình: {e}")
        return

    st.markdown("---")

    # -------------------------------------------------------------------------
    # KHU VỰC 2: FORM NHẬP LIỆU DỰ ĐOÁN (SIMULATOR)
    # -------------------------------------------------------------------------
    st.subheader("2. Mô phỏng dự đoán Doanh thu")
    
    # Sườn 3 ô nhập liệu chia làm 3 cột
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        boxes_input = st.number_input(
            label="Số lượng hộp (Boxes Shipped)", 
            min_value=0, 
            value=100
        )
    with col_in2:
        price_input = st.number_input(
            label="Giá mỗi hộp (Price per Box $)", 
            min_value=0.0, 
            value=10.0, 
            step=0.5
        )
    with col_in3:
        marketing_input = st.number_input(
            label="Chi phí Marketing ($)", 
            min_value=0, 
            value=500, 
            step=100
        )

    # Nút bấm dự đoán
    if st.button("🚀 Thực hiện dự đoán", use_container_width=True):
        try:
            predicted_amount = predictor.predict_revenue(boxes_input, price_input, marketing_input)
            st.success(f"### 🎯 Doanh thu dự kiến (Amount): **${predicted_amount:,.2f}**")
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

    # -------------------------------------------------------------------------
    # KHU VỰC 3: BẢO VỆ PHƯƠNG ÁN (Ghi chú trực quan trên UI)
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("📌 Bảo vệ phương án: Lý do chọn Hồi quy tuyến tính"):
        st.markdown("""
        - **Không chọn If/Else:** Luật rẽ nhánh không thể tự học hỏi hay cập nhật trọng số từ 200.000 dòng dữ liệu lịch sử.
        - **Không chọn Deep Learning:** Mạng nơ-ron quá nặng nề, khó giải thích (black-box) và không phù hợp với bài toán dự đoán đơn giản trên dữ liệu bảng.
        - **Ưu điểm của Linear Regression:** Nhẹ, huấn luyện cực nhanh, cho kết quả chính xác cao đối với các biến có mối quan hệ tuyến tính và dễ trình bày các chỉ số toán học ($R^2$, $MSE$).
        """)
