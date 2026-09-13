```python
# ==========================================
# Thành viên 3 - Data Analyzer
# File: src/analyzer.py
# ==========================================

import pandas as pd


class DataAnalyzer:

    def __init__(self, df):
        # Nhận dữ liệu sạch từ Thành viên 2
        self.df = df.copy()

    # ==========================================
    # 1. KPI
    # ==========================================

    def get_kpi(self):

        revenue = self.df["Amount"].sum()

        boxes = self.df["Boxes_Shipped"].sum()

        aov = self.df["Amount"].mean()

        marketing_ratio = (
            self.df["Marketing_Spend"].sum() / revenue
            if revenue != 0 else 0
        )

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

        marketing_amount = self.df[
            ["Marketing_Spend", "Amount"]
        ].corr().iloc[0, 1]

        discount_boxes = self.df[
            ["Discount_Pct", "Boxes_Shipped"]
        ].corr().iloc[0, 1]

        return {
            "Marketing - Amount": marketing_amount,
            "Discount - Boxes": discount_boxes
        }
```
