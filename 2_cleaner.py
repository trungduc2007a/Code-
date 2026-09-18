import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        """
        Nhận trực tiếp DataFrame (dữ liệu thô) từ DataLoader (Class 1)
        thay vì tự kết nối Database để đảm bảo luồng chạy xuyên suốt.
        """
        if df is None or df.empty:
            raise ValueError("❌ Lỗi: Dữ liệu đầu vào trống. Vui lòng kiểm tra lại DataLoader.")
        
        # Tạo một bản sao (copy) để không làm ảnh hưởng đến dữ liệu gốc
        self.df = df.copy() 

    def clean_missing_and_negative_data(self, fill_method="drop"):
        """
        Bước 2: Hàm làm sạch dữ liệu
        - fill_method có thể là: "drop" (xóa dòng), "mean" (điền trung bình), "median" (điền trung vị)
        """
        print(f"\nBắt đầu làm sạch dữ liệu (Phương pháp điền khuyết: {fill_method})...")
        
        # 1. Xử lý giá trị âm (Dữ liệu bán hàng không thể âm, nếu âm có thể do gõ nhầm)
        cols_to_check_negative = ['Amount', 'Boxes_Shipped', 'Price_per_Box']
        for col in cols_to_check_negative:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col] = self.df[col].abs()
        
        # 2. Xử lý dữ liệu khuyết (NaN)
        # Riêng Order_Date (thời gian), nếu thiếu thì chỉ có cách xóa bỏ dòng đó
        if 'Order_Date' in self.df.columns:
            self.df = self.df.dropna(subset=['Order_Date'])
            
        # Đối với các cột số
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        if fill_method == "drop":
            self.df = self.df.dropna(subset=numeric_cols)
        elif fill_method == "mean":
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif fill_method == "median":
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                
        print(f"✅ Làm sạch hoàn tất! Số dòng hiện tại: {len(self.df)}")
        return self.df

    def feature_engineering(self):
        """
        Bước 3: Tạo cột tính toán mới (Total Cost và Profit)
        """
        print("Bắt đầu tạo cột tính toán mới (Total Cost, Profit)...")
        
        # Công thức: Total Cost = Boxes_Shipped * Price_per_Box
        if 'Boxes_Shipped' in self.df.columns and 'Price_per_Box' in self.df.columns:
            self.df['Total Cost'] = self.df['Boxes_Shipped'] * self.df['Price_per_Box']
        else:
            print("⚠️ Cảnh báo: Thiếu cột Boxes_Shipped hoặc Price_per_Box.")
            
        # Công thức: Profit = Amount - Total Cost
        if 'Amount' in self.df.columns and 'Total Cost' in self.df.columns:
            self.df['Profit'] = self.df['Amount'] - self.df['Total Cost']
        else:
            print("⚠️ Cảnh báo: Thiếu cột Amount hoặc Total Cost.")
            
        print("✅ Tính toán hoàn tất!")
        return self.df


# =====================================================================
# CÁCH KẾT NỐI MẠCH LẠC GIỮA LOADER VÀ CLEANER
# =====================================================================
if __name__ == "__main__":
    # GIẢ LẬP BƯỚC 1: Lấy dữ liệu từ DataLoader
    # (Lưu ý: Bạn phải có sẵn class DataLoader hoặc import từ file data_loader.py)
    try:
        from data_loader import DataLoader
        print("--- 1. LOADER ĐANG CHẠY ---")
        loader = DataLoader("Chocolate_Sales.xlsx")
        df_raw = loader.process_data()
    except ImportError:
        # Nếu chạy độc lập file này mà không có DataLoader, tạo dữ liệu giả để test
        print("⚠️ Không tìm thấy file data_loader.py, dùng dữ liệu giả để test Cleaner.")
        df_raw = pd.DataFrame({
            'Order_Date': ['2023-01-01', None, '2023-01-03'],
            'Boxes_Shipped': [100, -50, np.nan],
            'Price_per_Box': [10, 15, 12],
            'Amount': [1000, -750, 600]
        })

    # BƯỚC 2: Truyền thẳng df_raw vào DataCleaner
    print("\n--- 2. CLEANER ĐANG CHẠY ---")
    cleaner = DataCleaner(df_raw)
    
    # Thực thi các hàm làm sạch và tính toán
    df_cleaned = cleaner.clean_missing_and_negative_data(fill_method="median")
    df_final = cleaner.feature_engineering()
    
    print("\n--- KẾT QUẢ SAU KHI CLEANER XỬ LÝ ---")
    print(df_final.head())
