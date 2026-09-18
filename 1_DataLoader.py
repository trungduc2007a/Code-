import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def process_data(self) -> pd.DataFrame:
        """
        Hàm thực hiện Trích xuất và Định dạng dữ liệu.
        (Phần Làm sạch được nhường lại cho class DataCleaner để không bị giẫm chân)
        """
        print("Đang đọc và định dạng dữ liệu...")
        
        # ==========================================
        # BƯỚC 1: TRÍCH XUẤT (Extract)
        # ==========================================
        # Đọc file Excel thô
        raw_df = pd.read_excel(self.filepath)
        
        # Tách dữ liệu bằng dấu phẩy (',') nếu bị dồn vào 1 cột
        col_name = raw_df.columns[0]
        self.df = raw_df[col_name].str.split(',', expand=True)
        self.df.columns = col_name.split(',')

        # ==========================================
        # BƯỚC 2: ĐỊNH DẠNG (Format)
        # ==========================================
        # Ép kiểu ngày tháng
        if 'Order_Date' in self.df.columns:
            self.df['Order_Date'] = pd.to_datetime(self.df['Order_Date'], errors='coerce')
            self.df['Year'] = self.df['Order_Date'].dt.year
            self.df['Month'] = self.df['Order_Date'].dt.month
        
        # Ép kiểu số thực (float) cho các cột tài chính
        numeric_cols = ['Discount_Pct', 'Price_per_Box', 'Marketing_Spend', 'Boxes_Shipped', 'Amount']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # ---------------------------------------------------------
        # ĐÃ LƯỢC BỎ BƯỚC 3 (LÀM SẠCH) Ở ĐÂY 
        # -> DataLoader chỉ xuất ra bảng đã định dạng đúng kiểu dữ liệu, 
        #    để DataCleaner xử lý NaN và số âm.
        # ---------------------------------------------------------

        print(f"Trích xuất thành công! Chuyển tiếp {len(self.df)} dòng cho DataCleaner.")
        return self.df

# =====================================================================
# PIPELINE TỔNG: CÁCH KẾT NỐI MẠCH LẠC VỚI 4 CLASS CÒN LẠI
# =====================================================================
if __name__ == "__main__":
    # Import các class từ các file khác (Giả sử các file nằm cùng thư mục)
    # from data_cleaner import DataCleaner
    # from data_analyzer import DataAnalyzer
    # from data_visualizer import render_tab_4
    # from data_predictor import render_tab_5
    # import streamlit as st

    print("--- 1. CHẠY DATA LOADER ---")
    loader = DataLoader("Chocolate_Sales.xlsx")
    df_raw = loader.process_data() # df_raw lúc này đã định dạng cột nhưng vẫn còn khuyết (NaN)
    
    print("\n--- 2. CHẠY DATA CLEANER ---")
    # Khởi tạo Cleaner: Không cần kết nối SQL, ta sẽ "tiêm" trực tiếp dữ liệu từ Loader vào.
    # cleaner = DataCleaner(db_type="", db_host="", db_name="", db_user="", db_password="", db_port="")
    # cleaner.df = df_raw.copy()  <-- BƯỚC QUAN TRỌNG: Kết nối 2 class với nhau
    
    # Thực hiện làm sạch và thêm cột (Profit, Total Cost)
    # df_clean = cleaner.clean_missing_and_negative_data(fill_method="median")
    # df_final = cleaner.feature_engineering()
    
    print("\n--- 3. CHẠY DATA ANALYZER ---")
    # Truyền dữ liệu sạch vào Analyzer để tính toán ngầm
    # analyzer = DataAnalyzer(df_final)
    # print(analyzer.get_kpi())
    
    print("\n--- 4 & 5. KHỞI CHẠY GIAO DIỆN STREAMLIT ---")
    # Tạo giao diện tổng để gọi Visualizer và Predictor
    # st.title("Dashboard Phân Tích Bán Hàng")
    # tab_viz, tab_pred = st.tabs(["Trực Quan Hóa", "Dự Đoán ML"])
    
    # with tab_viz:
    #     render_tab_4(df_final)
    # with tab_pred:
    #     render_tab_5(df_final)
