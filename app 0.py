import streamlit as st
import pandas as pd

# ==========================================
# IMPORT CÁC CLASS ĐỘC LẬP TỪ 5 FILE
# ==========================================
from data_loader_2 import DataLoader
from data_cleaner_2 import DataCleaner
from data_analyzer_3 import DataAnalyzer
from data_visualizer_4 import render_tab_4
from data_predictor_5 import render_tab_5

# ==========================================
# CẤU HÌNH TRANG & KHỞI TẠO BỘ NHỚ TẠM
# ==========================================
st.set_page_config(page_title="Phân tích Chocolate", layout="wide")
st.title("🍫 Ứng dụng Phân tích Doanh thu Chocolate")

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None      
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None    

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 1. Nạp dữ liệu", 
    "🛠️ 2. Xử lý & Làm sạch", 
    "📊 3. Thống kê", 
    "📈 4. Trực quan", 
    "🚀 5. Dự đoán ML"
])

# ==========================================
# TAB 1: THÀNH VIÊN 1 - DATA LOADER
# ==========================================
with tab1:
    st.header("📥 Thành viên 1: Nạp và Định dạng Dữ liệu")
    
    file_path = st.text_input("Đường dẫn file Excel:", "Chocolate_Sales.xlsx")
    
    if st.button("🚀 Nạp và Trích xuất Dữ liệu", type="primary"):
        with st.spinner("Đang đọc dữ liệu..."):
            try:
                loader = DataLoader(file_path)
                df_raw = loader.process_data()
                st.session_state.df_raw = df_raw
                st.success(f"✅ Đã nạp thành công {len(df_raw)} dòng dữ liệu thô!")
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {e}")
                
    if st.session_state.df_raw is not None:
        st.subheader("Bản xem trước Dữ liệu Thô (100 dòng đầu)")
        st.dataframe(st.session_state.df_raw.head(100), use_container_width=True)

# ==========================================
# TAB 2: THÀNH VIÊN 2 - DATA CLEANER (LÀ EM)
# ==========================================
with tab2:
    st.header("🛠️ Thành viên 2: Tiền xử lý & Làm sạch dữ liệu")
    
    if st.session_state.df_raw is None:
        st.info("👈 Vui lòng sang Tab 1 để Nạp dữ liệu trước.")
    else:
        st.success("✅ Đã nhận được dữ liệu thô từ Tab 1. Sẵn sàng xử lý!")
        phuong_phap = st.selectbox("Chọn phương pháp điền khuyết dữ liệu (NaN):", ["mean", "median", "drop"])
        
        if st.button("🚀 Thực hiện Làm sạch & Tạo đặc trưng", type="primary"):
            with st.spinner("Đang làm sạch dữ liệu..."):
                try:
                    # LƯU Ý: Phải đảm bảo file data_cleaner_2.py đã sửa lại __init__ để nhận df
                    cleaner = DataCleaner(st.session_state.df_raw)
                    df_cleaned = cleaner.clean_missing_and_negative_data(fill_method=phuong_phap)
                    df_final = cleaner.feature_engineering()
                    
                    st.session_state.df_clean = df_final
                    st.success(f"✅ Làm sạch hoàn tất! Số dòng hợp lệ còn lại: {len(df_final)}")
                except Exception as e:
                    st.error(f"❌ Lỗi khi làm sạch: {e}")

        # Hiển thị bảng sạch (Đã phá giới hạn)
        if st.session_state.df_clean is not None:
            # Đếm xem bảng còn chính xác bao nhiêu dòng để in lên tiêu đề cho ngầu
            tong_so_dong = len(st.session_state.df_clean)
            
            st.subheader("Bảng dữ liệu Sạch toàn tập")
            
            # Đã xóa .head(100), truyền nguyên cái bảng df_clean vào
            st.dataframe(st.session_state.df_clean, use_container_width=True)

# ==========================================
# TAB 3: THÀNH VIÊN 3 - DATA ANALYZER
# ==========================================
with tab3:
    st.header("📊 Thành viên 3: Phân tích Thống kê")
    
    if st.session_state.df_clean is None:
        st.info("👈 Vui lòng sang Tab 2 để Làm sạch dữ liệu trước.")
    else:
        analyzer = DataAnalyzer(st.session_state.df_clean)
        st.subheader("1. Chỉ số KPI Cốt lõi")
        kpis = analyzer.get_kpi()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng Doanh Thu", f"${kpis.get('Tổng doanh thu', 0):,.2f}")
        col2.metric("Tổng Sản Lượng (Hộp)", f"{kpis.get('Tổng sản lượng', 0):,.0f}")
        col3.metric("Giá Trị Đơn Trung Bình (AOV)", f"${kpis.get('AOV', 0):,.2f}")
        col4.metric("Tỷ lệ Marketing / Doanh thu", f"{kpis.get('Marketing / Doanh thu', 0):.2%}")
        
        st.divider()
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("2. Top 5 Sản Phẩm Bán Chạy")
            st.dataframe(analyzer.top_products(), use_container_width=True, hide_index=True)
            st.subheader("3. Tương quan dữ liệu (Pearson)")
            corr_data = analyzer.correlation()
            for cap_bien, he_so in corr_data.items():
                st.write(f"🔹 **{cap_bien}**: `{he_so:.4f}`")
            
        with col_right:
            st.subheader("4. Doanh thu theo Quốc gia")
            st.dataframe(analyzer.revenue_by_country(), use_container_width=True, hide_index=True)
            st.subheader("5. Nhân viên Xuất sắc nhất")
            st.dataframe(analyzer.top_salesperson(), use_container_width=True, hide_index=True)

# ==========================================
# TAB 4 & 5: THÀNH VIÊN 4 VÀ 5
# ==========================================
with tab4:
    if st.session_state.df_clean is not None:
        render_tab_4(st.session_state.df_clean)
    else:
        st.info("👈 Vui lòng sang Tab 2 để Làm sạch dữ liệu trước khi vẽ biểu đồ.")

with tab5:
    if st.session_state.df_clean is not None:
        render_tab_5(st.session_state.df_clean)
    else:
        st.info("👈 Vui lòng sang Tab 2 để Làm sạch dữ liệu trước khi chạy Machine Learning.")