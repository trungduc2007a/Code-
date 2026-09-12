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
# OOP
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
        
    # Sườn cho nút bấm dự đoán
    if st.button("Chạy Dự đoán"):
        # TODO: Lấy dữ liệu từ 3 ô input trên truyền vào predictor.predict_revenue()
        # In ra kết quả
        st.success("Doanh thu dự kiến: [$ Kết quả tính toán]")

    # ==========================================
    # PHẦN 3: BẢO VỆ PHƯƠNG ÁN (Theo yêu cầu docx)
    # ==========================================
    st.markdown("---")
    with st.expander("Bảo vệ phương án: Tại sao chọn Linear Regression?"):
        # TODO: Ghi chú các gạch đầu dòng lý luận
        st.write("- Lý do không dùng If/Else...")
        st.write("- Lý do không dùng Deep Learning...")
        st.write("- Ưu điểm của Linear Regression với dữ liệu này...")
