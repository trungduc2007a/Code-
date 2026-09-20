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
                self.df.loc[self.df[col] < 0, col] = np.nan

        # 2. Xử lý khoảng trống (Áp dụng cho toàn bộ dữ liệu)
        if fill_method == "drop":
            self.df = self.df.dropna(subset=['Amount', 'Boxes_Shipped'])
        else:
            for col in ['Amount', 'Boxes_Shipped']:
                if col in self.df.columns:
                    if fill_method == "mean":
                        fill_value = self.df[col].mean()
                    elif fill_method == "zero":
                        fill_value = 0 # Điền số 0 vào ô trống
                    
                    # Lệnh này tự động quét và lấp đầy lỗ hổng trên toàn bộ cột
                    self.df[col] = self.df[col].fillna(fill_value)
                    
        return self.df

    def feature_engineering(self):
        # 3. Tạo 2 cột tính toán mới
        if 'Boxes_Shipped' in self.df.columns and 'Amount' in self.df.columns:
            if 'Marketing_Spend' in self.df.columns:
                self.df['Total Cost'] = (self.df['Boxes_Shipped'] * 5) + self.df['Marketing_Spend']
            else:
                self.df['Total Cost'] = self.df['Boxes_Shipped'] * 5

            self.df['Expected Profit'] = self.df['Amount'] - self.df['Total Cost']
            
        return self.df
