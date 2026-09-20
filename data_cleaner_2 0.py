import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        # Tạo bản sao để không làm hỏng dữ liệu gốc của Tab 1
        self.df = df.copy()

    def clean_missing_and_negative_data(self, fill_method="mean"):
        # 1. Biến các giá trị âm (lỗi) thành NaN để xử lý chung
        for col in ['Amount', 'Boxes_Shipped']:
            if col in self.df.columns:
                # Nếu giá trị nhỏ hơn 0, chuyển thành NaN
                self.df.loc[self.df[col] < 0, col] = np.nan

        # 2. Xử lý khoảng trống (NaN) theo phương pháp được chọn
        if fill_method == "drop":
            # Xóa sạch các dòng chứa NaN ở 3 cột quan trọng
            self.df = self.df.dropna(subset=['Amount', 'Boxes_Shipped', 'Order_Date'])
        else:
            for col in ['Amount', 'Boxes_Shipped']:
                if col in self.df.columns:
                    # Tính toán mean hoặc median bỏ qua NaN
                    if fill_method == "mean":
                        fill_value = self.df[col].mean()
                    elif fill_method == "median":
                        fill_value = self.df[col].median()
                    
                    # Lấp đầy lỗ hổng
                    self.df[col] = self.df[col].fillna(fill_value)
            
            # Riêng cột Ngày tháng (Order_Date) không thể tính trung bình được
            # Ta dùng phương pháp ffill (lấy ngày của đơn hàng trước đó điền vào)
            if 'Order_Date' in self.df.columns:
                self.df['Order_Date'] = self.df['Order_Date'].ffill()
                
        return self.df

    def feature_engineering(self):
        # 3. Tạo 2 cột tính toán mới theo đúng yêu cầu đề bài
        # Giả định dữ liệu có sẵn cột Marketing_Spend và Price_per_Box
        if 'Boxes_Shipped' in self.df.columns and 'Amount' in self.df.columns:
            
            # Công thức 1: Tổng chi phí (Giả sử bằng số hộp * 5$ phí vận chuyển + phí Marketing)
            # (Em có thể tự thay đổi công thức cho khớp với file Excel của nhóm)
            if 'Marketing_Spend' in self.df.columns:
                self.df['Total Cost'] = (self.df['Boxes_Shipped'] * 5) + self.df['Marketing_Spend']
            else:
                self.df['Total Cost'] = self.df['Boxes_Shipped'] * 5

            # Công thức 2: Lợi nhuận dự kiến = Doanh thu - Tổng chi phí
            self.df['Expected Profit'] = self.df['Amount'] - self.df['Total Cost']
            
        return self.df