# Các thành viên sẽ nhập code trong Project này

# Class DataLoader:
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
