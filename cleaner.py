import pandas as pd
import numpy as np
from sqlalchemy import create_engine # Công cụ kết nối đa năng được Pandas khuyên dùng

class DataCleaner:
    def __init__(self, db_type, db_host, db_name, db_user, db_password, db_port):
        self.db_type = db_type
        self.db_host = db_host
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_port = db_port
        self.df = None 

    def _create_connection(self):
        """Tạo kết nối linh hoạt cho cả PostgreSQL và MySQL"""
        try:
            # Tạo chuỗi kết nối (Connection URI) dựa trên loại Database
            if self.db_type == "PostgreSQL":
                url = f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            elif self.db_type == "MySQL":
                url = f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            
            # Khởi tạo động cơ kết nối
            engine = create_engine(url)
            return engine
        except Exception as e:
            print(f"Lỗi tạo kết nối: {e}")
            return None

    def load_data_from_db(self, table_name="chocolate_sales"):
        engine = self._create_connection()
        if engine is not None:
            try:
                # pandas rất thích dùng SQLAlchemy engine, nó sẽ tự động mở và đóng kết nối
                query = f"SELECT * FROM {table_name}"
                self.df = pd.read_sql(query, engine)
                print(f"Tải thành công {len(self.df)} dòng từ {self.db_type}.")
                return self.df
            except Exception as e:
                print(f"Lỗi truy vấn: {e}")
                return None
        return None

    # ... (Các hàm clean_missing_and_negative_data và feature_engineering của em giữ nguyên ở dưới nhé) ...

    # Viết logic Làm sạch và Kỹ thuật Đặc trưng (Feature Engineering)

    def clean_missing_and_negative_data(self, fill_method="drop"):
        """
        Bước 2: Hàm làm sạch dữ liệu
        - fill_method có thể là: "drop" (xóa dòng), "mean" (điền trung bình), "median" (điền trung vị)
        """
        if self.df is None or self.df.empty:
            print("❌ Lỗi: Chưa có dữ liệu. Hãy chạy hàm load_data_from_db() trước.")
            return None
            
        print(f"Bắt đầu làm sạch dữ liệu (Phương pháp: {fill_method})...")
        
        # 1. Xử lý giá trị âm (Dữ liệu bán hàng không thể âm, nếu âm có thể do gõ nhầm)
        # Chuyển tất cả giá trị âm thành dương bằng hàm trị tuyệt đối .abs()
        cols_to_check_negative = ['Amount', 'Boxes_Shipped', 'Price_per_Box']
        for col in cols_to_check_negative:
            if col in self.df.columns:
                # Ép kiểu về dạng số (numeric) để chắc chắn tính toán được
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col] = self.df[col].abs()
        
        # 2. Xử lý dữ liệu khuyết (NaN)
        # Các cột thường bị thiếu dữ liệu
        cols_with_nan = ['Amount', 'Boxes_Shipped', 'Order_Date']
        
        # Riêng Order_Date (thời gian), nếu thiếu thì chỉ có cách xóa bỏ dòng đó
        if 'Order_Date' in self.df.columns:
            self.df = self.df.dropna(subset=['Order_Date'])
            
        # Đối với các cột số (Amount, Boxes_Shipped...)
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        if fill_method == "drop":
            # Xóa sạch các dòng chứa NaN
            self.df = self.df.dropna(subset=numeric_cols)
        elif fill_method == "mean":
            # Điền khuyết bằng giá trị trung bình của cột
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif fill_method == "median":
            # Điền khuyết bằng giá trị trung vị của cột
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                
        print(f"✅ Làm sạch hoàn tất! Số dòng hiện tại: {len(self.df)}")
        return self.df

    def feature_engineering(self):
        """
        Bước 3: Tạo cột tính toán mới (Total Cost và Profit)
        """
        if self.df is None or self.df.empty:
            print("❌ Lỗi: Dữ liệu trống.")
            return None
            
        print("Bắt đầu tính toán tính năng mới (Total Cost, Profit)...")
        
        # Công thức: Total Cost = Boxes_Shipped * Price_per_Box
        if 'Boxes_Shipped' in self.df.columns and 'Price_per_Box' in self.df.columns:
            self.df['Total Cost'] = self.df['Boxes_Shipped'] * self.df['Price_per_Box']
        else:
            print("⚠️ Cảnh báo: Thiếu cột Boxes_Shipped hoặc Price_per_Box để tính Total Cost.")
            
        # Công thức: Profit = Amount - Total Cost
        if 'Amount' in self.df.columns and 'Total Cost' in self.df.columns:
            self.df['Profit'] = self.df['Amount'] - self.df['Total Cost']
        else:
            print("⚠️ Cảnh báo: Thiếu cột Amount hoặc Total Cost để tính Profit.")
            
        print("✅ Tính toán hoàn tất!")
        return self.df

if __name__ == "__main__":
    # Khởi tạo đối tượng (Nhớ thay mật khẩu cho đúng với máy em)
    cleaner = DataCleaner(
        db_host="localhost",
        db_name="postgres", 
        db_user="postgres",
        db_password="123456",  # Cập nhật mật khẩu pgAdmin của em vào đây
        db_port="5432"
    )
    
    # 1. Gọi hàm kéo dữ liệu
    df_raw = cleaner.load_data_from_db("chocolate_sales")
    
    if df_raw is not None:
        # 2. Gọi hàm làm sạch (dùng phương pháp trung vị)
        df_cleaned = cleaner.clean_missing_and_negative_data(fill_method="median")
        
        # 3. Gọi hàm tạo cột tính toán mới
        df_final = cleaner.feature_engineering()
        
        # In thử 5 dòng đầu tiên để kiểm chứng
        print("\n--- KẾT QUẢ DỮ LIỆU SAU KHI XỬ LÝ ---")
        print(df_final[['Boxes_Shipped', 'Price_per_Box', 'Amount', 'Total Cost', 'Profit']].head())