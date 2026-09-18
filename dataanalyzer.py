# ==========================================
# Thành viên 3 - Data Analyzer
# ==========================================

import pandas as pd

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Nhận dữ liệu sạch (df_final) từ Thành viên 2 (DataCleaner).
        """
        if df is None or df.empty:
            raise ValueError("❌ Lỗi: Dữ liệu truyền vào Analyzer bị trống. Kiểm tra lại Cleaner.")
        self.df = df.copy()

    # ==========================================
    # 1. KPI
    # ==========================================
    def get_kpi(self):
        required_cols = ["Amount", "Boxes_Shipped", "Marketing_Spend"]
        if not all(col in self.df.columns for col in required_cols):
            print("⚠️ Cảnh báo: Thiếu cột dữ liệu để tính KPI.")
            return {}

        revenue = self.df["Amount"].sum()
        boxes = self.df["Boxes_Shipped"].sum()
        aov = self.df["Amount"].mean()
        marketing_ratio = (self.df["Marketing_Spend"].sum() / revenue) if revenue != 0 else 0

        return {
            "Tổng doanh thu": revenue,
            "Tổng sản lượng": boxes,
            "AOV": aov,
            "Marketing / Doanh thu": marketing_ratio
        }

    # ==========================================
    # 2. TOP 5 SẢN PHẨM
    # ==========================================
    def top_products(self):
        if "Product" not in self.df.columns or "Boxes_Shipped" not in self.df.columns:
            return pd.DataFrame() # Trả về bảng rỗng nếu thiếu dữ liệu để tránh crash
            
        return (
            self.df.groupby("Product")["Boxes_Shipped"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )

    # ==========================================
    # 3. NHÂN VIÊN XUẤT SẮC
    # ==========================================
    def top_salesperson(self):
        if "Salesperson" not in self.df.columns or "Amount" not in self.df.columns:
            return pd.DataFrame()

        return (
            self.df.groupby("Salesperson")["Amount"]
            .sum()
            .sort_values(ascending=False)
            .head(1)
            .reset_index()
        )

    # ==========================================
    # 4. DOANH THU TRUNG BÌNH THEO COUNTRY
    # ==========================================
    def revenue_by_country(self):
        if "Country" not in self.df.columns or "Amount" not in self.df.columns:
            return pd.DataFrame()

        return (
            self.df.groupby("Country")["Amount"]
            .mean()
            .sort_values(ascending=False)
            .reset_index(name="Average_Revenue")
        )

    # ==========================================
    # 5. TƯƠNG QUAN PEARSON
    # ==========================================
    def correlation(self):
        required_cols = ["Marketing_Spend", "Amount", "Discount_Pct", "Boxes_Shipped"]
        if not all(col in self.df.columns for col in required_cols):
            return {"Marketing - Amount": 0, "Discount - Boxes": 0}

        marketing_amount = self.df[["Marketing_Spend", "Amount"]].corr().iloc[0, 1]
        discount_boxes = self.df[["Discount_Pct", "Boxes_Shipped"]].corr().iloc[0, 1]

        return {
            "Marketing - Amount": marketing_amount,
            "Discount - Boxes": discount_boxes
        }

# =====================================================================
# PIPELINE: CÁCH KẾT NỐI MẠCH LẠC TỪ LOADER -> CLEANER -> ANALYZER
# =====================================================================
if __name__ == "__main__":
    try:
        from data_loader import DataLoader
        from data_cleaner import DataCleaner
        
        print("--- 1. CHẠY LOADER ---")
        loader = DataLoader("Chocolate_Sales.xlsx")
        df_raw = loader.process_data()
        
        print("\n--- 2. CHẠY CLEANER ---")
        cleaner = DataCleaner(df_raw)
        df_cleaned = cleaner.clean_missing_and_negative_data(fill_method="median")
        df_final = cleaner.feature_engineering()
        
        print("\n--- 3. CHẠY ANALYZER ---")
        # Truyền df_final (kết quả của Cleaner) thẳng vào Analyzer
        analyzer = DataAnalyzer(df_final)
        
        # In thử kết quả để kiểm chứng
        print("KPIs:", analyzer.get_kpi())
        print("\nTop 5 Sản phẩm:\n", analyzer.top_products())
        print("\nĐộ tương quan:", analyzer.correlation())
        
    except ImportError:
        print("⚠️ Không tìm thấy file data_loader.py hoặc data_cleaner.py để test luồng.")
        # Dữ liệu giả lập nếu muốn test độc lập file này
        import numpy as np
        df_mock = pd.DataFrame({
            "Amount": [100, 200, 300],
            "Boxes_Shipped": [10, 20, 30],
            "Marketing_Spend": [10, 15, 20],
            "Product": ["A", "B", "A"],
            "Salesperson": ["John", "Jane", "John"],
            "Country": ["USA", "UK", "USA"],
            "Discount_Pct": [0.1, 0.2, 0.1]
        })
        analyzer = DataAnalyzer(df_mock)
        print("Test KPI (Dữ liệu giả):", analyzer.get_kpi())
