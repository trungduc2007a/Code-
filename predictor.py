#=====================
# THÀNH VIÊN 5
# File: data_predictor_2.py
#=====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

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
        self.features = ['Boxes_Shipped', 'Price_per_Box', 'Marketing_Spend']

    def train_model(self, df: pd.DataFrame):
        """
        Huấn luyện mô hình Hồi quy tuyến tính trên toàn bộ dữ liệu.
        """
        target = 'Amount'

        # Kiểm tra sự tồn tại của các cột dữ liệu
        if not all(col in df.columns for col in self.features + [target]):
            raise ValueError(f"Dữ liệu truyền vào thiếu các cột bắt buộc: {self.features} hoặc {target}")

        # Làm sạch các dòng NaN nếu có trước khi đưa vào mô hình
        df_clean = df.dropna(subset=self.features + [target])

        if df_clean.empty:
            raise ValueError("Dữ liệu sau khi loại bỏ NaN bị rỗng, không thể huấn luyện.")

        X = df_clean[self.features]
        y = df_clean[target]

        # Chia dữ liệu: 80% Train, 20% Test
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

        # Định dạng input thành DataFrame với tên cột chuẩn để tránh UserWarning của scikit-learn
        input_data = pd.DataFrame(
            [[boxes, price, marketing]], 
            columns=self.features
        )
        prediction = self.model.predict(input_data)
        return prediction[0]


#===============================
# GIAO DIỆN UI STREAMLIT
#===============================

def render_tab_5(df_clean: pd.DataFrame):
    st.header("📈 Tab 5: Dự đoán Doanh thu (Machine Learning)")
    st.caption("Huấn luyện mô hình Hồi quy tuyến tính trên toàn bộ dữ liệu")

    if df_clean is None or df_clean.empty:
        st.warning("⚠️ Chưa có dữ liệu sạch từ Cleaner. Vui lòng kiểm tra lại luồng chạy.")
        return

    # -------------------------------------------------------------------------
    # KHU VỰC 1: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH (R2 & MSE)
    # -------------------------------------------------------------------------
    st.subheader("1. Đánh giá độ chính xác của mô hình")
    
    # Khởi tạo mô hình ngay tại đây
    predictor = DataPredictor()
    
    try:
        # Huấn luyện mô hình trên dữ liệu được truyền vào
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
    # KHU VỰC 3: BẢO VỆ PHƯƠNG ÁN
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("📌 Bảo vệ phương án: Lý do chọn Hồi quy tuyến tính"):
        st.markdown("""
        - **Không chọn If/Else:** Luật rẽ nhánh không thể tự học hỏi hay cập nhật trọng số từ dữ liệu lịch sử.
        - **Không chọn Deep Learning:** Mạng nơ-ron quá nặng nề, khó giải thích (black-box) và không phù hợp với bài toán dự đoán đơn giản trên dữ liệu bảng.
        - **Ưu điểm của Linear Regression:** Nhẹ, huấn luyện cực nhanh, cho kết quả chính xác cao đối với các biến có mối quan hệ tuyến tính và dễ trình bày các chỉ số toán học ($R^2$, $MSE$).
        """)


# =====================================================================
# BẢN DEMO ĐỘC LẬP: TEST MACHINE LEARNING TRÊN TRÌNH DUYỆT
# Lệnh chạy: streamlit run data_predictor_2.py
# =====================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Test ML Predictor", layout="wide")
    st.title("Giao Diện Test Độc Lập - Thành Viên 5")
    
    # Tạo dữ liệu giả có tính tuyến tính (Amount bị ảnh hưởng bởi Boxes, Price, Marketing)
    np.random.seed(42)
    boxes = np.random.randint(50, 500, 200)
    price = np.random.uniform(5.0, 20.0, 200)
    marketing = np.random.randint(100, 1000, 200)
    
    # Amount = Boxes * Price + 0.5 * Marketing + noise
    amount = (boxes * price) + (0.5 * marketing) + np.random.normal(0, 100, 200)
    
    mock_data = pd.DataFrame({
        'Boxes_Shipped': boxes,
        'Price_per_Box': price,
        'Marketing_Spend': marketing,
        'Amount': amount
    })
    
    st.success("Đã tạo tập dữ liệu giả lập để huấn luyện mô hình!")
    st.dataframe(mock_data.head())
    
    # Chạy giao diện Tab 5
    render_tab_5(mock_data)
