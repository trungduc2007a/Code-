import streamlit as st
import pandas as pd
from cleaner import DataCleaner # Lấy class em vừa viết lúc nãy sang đây

# 1. Cấu hình trang web cơ bản
st.set_page_config(page_title="Phân tích Chocolate", layout="wide")
st.title("🍫 Ứng dụng Phân tích Doanh thu Chocolate")

# 2. Tạo các Tab cho 5 thành viên nhóm
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Thành viên 1: Nạp dữ liệu", 
    "Thành viên 2: Xử lí dữ liệu", 
    "Thành viên 3: Thống kê", 
    "Thành viên 4: Trực quan", 
    "Thành viên 5: Dự đoán"
])

# ==========================================
# KHU VỰC ĐẤT DIỄN CỦA THÀNH VIÊN 2 
# ==========================================
# ==========================================
# KHU VỰC ĐẤT DIỄN CỦA THÀNH VIÊN 2 
# ==========================================
with tab2:
    st.header("🛠️ Tiền xử lý & Làm sạch dữ liệu")
    
    # --- TẠO THANH SIDEBAR ĐỂ NGƯỜI DÙNG TỰ NHẬP CẤU HÌNH ---
    st.sidebar.header("⚙️ Cấu hình Database")
    db_type = st.sidebar.selectbox("Loại hệ quản trị CSDL:", ["PostgreSQL", "MySQL"])
    db_host = st.sidebar.text_input("Host:", "localhost")
    db_port = st.sidebar.text_input("Port:", "5432" if db_type == "PostgreSQL" else "3306")
    db_name = st.sidebar.text_input("Tên Database:", "postgres")
    db_user = st.sidebar.text_input("Tài khoản:", "postgres" if db_type == "PostgreSQL" else "root")
    db_password = st.sidebar.text_input("Mật khẩu:", type="password") # Ẩn mật khẩu khi gõ
    # --------------------------------------------------------

    # Hàm tải dữ liệu có nhận các thông số từ Sidebar
    @st.cache_data 
    def get_raw_data(t_db, t_host, t_name, t_user, t_pass, t_port):
        cleaner = DataCleaner(
            db_type=t_db, db_host=t_host, db_name=t_name, 
            db_user=t_user, db_password=t_pass, db_port=t_port
        )
        df = cleaner.load_data_from_db()
        return cleaner, df

    # Chỉ bắt đầu kết nối khi người dùng đã nhập mật khẩu
    if db_password: 
        try:
            cleaner_obj, df_raw = get_raw_data(db_type, db_host, db_name, db_user, db_password, db_port)
            
            if df_raw is not None:
                st.success("✅ Đã kết nối thành công tới Database!")
                
                st.subheader("1. Bảng dữ liệu gốc (Chưa xử lý)")
                st.dataframe(df_raw, use_container_width=True)
                
                st.subheader("2. Tùy chọn Làm sạch")
                phuong_phap = st.selectbox("Chọn phương pháp:", ["mean", "median", "drop"])
                
                if st.button("🚀 Thực hiện Làm sạch & Tính toán"):
                    with st.spinner("Đang xử lý dữ liệu..."):
                        df_cleaned = cleaner_obj.clean_missing_and_negative_data(fill_method=phuong_phap)
                        df_final = cleaner_obj.feature_engineering()
                        
                        st.subheader("3. Bảng dữ liệu hoàn chỉnh")
                        st.dataframe(df_final, use_container_width=True)
            else:
                st.error("❌ Kết nối thất bại. Vui lòng kiểm tra lại thông tin Database hoặc Bảng chưa tồn tại.")
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
    else:
        st.info("👈 Vui lòng nhập Mật khẩu Database ở cột bên trái để bắt đầu.")
