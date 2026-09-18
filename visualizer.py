# ==============================================================================
# THÀNH VIÊN 4: MODULE src/visualizer.py & TAB "TRỰC QUAN HÓA"
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


class DataVisualizer:
    """
    Class DataVisualizer nhận kết quả dữ liệu từ RAM (Thành viên 3)
    và lập trình các hàm vẽ biểu đồ bằng Matplotlib và Seaborn.
    """
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("❌ Lỗi: Dữ liệu truyền vào Visualizer bị trống.")
            
        self.df = df.copy()
        # Thiết lập giao diện biểu đồ chuẩn
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

    def plot_line_chart(self, freq: str = 'ME') -> plt.Figure:
        """
        1. Biểu đồ đường (Line Chart): Thể hiện xu hướng doanh thu theo tháng/quý.
        :param freq: 'ME' (Tháng) hoặc 'QE' (Quý) - Cập nhật cú pháp mới của Pandas
        """
        fig, ax = plt.subplots(figsize=(10, 4.5))
        
        if 'Order_Date' in self.df.columns and 'Amount' in self.df.columns:
            temp_df = self.df.copy()
            temp_df['Order_Date'] = pd.to_datetime(temp_df['Order_Date'], errors='coerce')
            
            # Xóa các dòng có ngày tháng bị lỗi NaT trước khi resample
            temp_df = temp_df.dropna(subset=['Order_Date'])
            
            if not temp_df.empty:
                # Nhóm dữ liệu theo tháng/quý
                resampled = temp_df.resample(freq, on='Order_Date')['Amount'].sum().reset_index()
                label_text = "Tháng" if freq == 'ME' else "Quý"
                
                ax.plot(
                    resampled['Order_Date'], 
                    resampled['Amount'], 
                    marker='o', 
                    color='#1f77b4', 
                    linewidth=2,
                    markersize=6
                )
                ax.set_title(f"📈 Xu Hướng Doanh Thu Theo {label_text}", fontsize=12, fontweight='bold', pad=12)
                ax.set_xlabel("Thời Gian")
                ax.set_ylabel("Tổng Doanh Thu ($)")
                ax.grid(True, linestyle='--', alpha=0.5)
                fig.autofmt_xdate()
            else:
                ax.text(0.5, 0.5, "Dữ liệu ngày tháng không hợp lệ", ha='center', va='center', color='red')
        else:
            ax.text(0.5, 0.5, "Thiếu cột Order_Date hoặc Amount trong dữ liệu", ha='center', va='center', color='red')
            
        return fig

    def plot_bar_chart(self, group_by: str = 'Country') -> plt.Figure:
        """
        2. Biểu đồ thanh ngang/Cột (Bar/Horizontal Bar Chart): 
           So sánh doanh thu giữa các quốc gia (Country) hoặc top sản phẩm (Product).
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if group_by in self.df.columns and 'Amount' in self.df.columns:
            if group_by == 'Product':
                # Top 10 sản phẩm
                grouped = self.df.groupby('Product')['Amount'].sum().sort_values(ascending=True).tail(10)
                title = " Top 10 Sản Phẩm Có Doanh Thu Cao Nhất"
                color = '#2ca02c'
            else:
                # Doanh thu theo quốc gia
                grouped = self.df.groupby('Country')['Amount'].sum().sort_values(ascending=True)
                title = " So Sánh Doanh Thu Giữa Các Quốc Gia (Country)"
                color = '#1f77b4'

            bars = ax.barh(grouped.index, grouped.values, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel("Tổng Doanh Thu ($)")
            ax.set_ylabel(group_by)
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            
            # Hiển thị giá trị cụ thể ở từng thanh
            for bar in bars:
                w = bar.get_width()
                ax.text(w * 1.01, bar.get_y() + bar.get_height()/2, f"${w:,.0f}", va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, f"Thiếu cột {group_by} hoặc Amount trong dữ liệu", ha='center', va='center', color='red')

        return fig

    def plot_pie_chart(self) -> plt.Figure:
        """
        3. Biểu đồ tròn (Pie Chart): 
           Thể hiện tỷ trọng doanh thu theo kênh bán hàng (Channel: Retail, Wholesale, Online).
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        
        if 'Channel' in self.df.columns and 'Amount' in self.df.columns:
            channel_data = self.df.groupby('Channel')['Amount'].sum()
            colors = sns.color_palette("pastel")[:len(channel_data)]
            
            ax.pie(
                channel_data, 
                labels=channel_data.index, 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=colors,
                explode=[0.03] * len(channel_data),
                shadow=True,
                textprops={'fontsize': 11, 'weight': 'bold'}
            )
            ax.set_title(" Tỷ Trọng Doanh Thu Theo Kênh Bán Hàng (Channel)", fontsize=12, fontweight='bold', pad=12)
        else:
            ax.text(0.5, 0.5, "Thiếu cột Channel hoặc Amount trong dữ liệu", ha='center', va='center', color='red')

        return fig

    def plot_scatter_and_heatmap(self) -> plt.Figure:
        """
        4. Biểu đồ phân tán (Scatter Plot) & Nhiệt (Heatmap): 
           Thể hiện độ tương quan giữa Boxes_Shipped, Marketing_Spend, Discount_Pct, Amount.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Biểu đồ phân tán (Scatter Plot)
        if 'Boxes_Shipped' in self.df.columns and 'Amount' in self.df.columns:
            sns.scatterplot(
                data=self.df, 
                x='Boxes_Shipped', 
                y='Amount', 
                alpha=0.4, 
                ax=axes[0], 
                color='purple'
            )
            sns.regplot(
                data=self.df, 
                x='Boxes_Shipped', 
                y='Amount', 
                ax=axes[0], 
                scatter=False, 
                color='red', 
                line_kws={'linewidth': 1.5}
            )
            axes[0].set_title(" Phân Tán: Boxes_Shipped vs Amount", fontsize=11, fontweight='bold')
            axes[0].set_xlabel("Số Hộp Vận Chuyển (Boxes_Shipped)")
            axes[0].set_ylabel("Doanh Thu (Amount)")
            axes[0].grid(True, linestyle='--', alpha=0.5)
        else:
            axes[0].text(0.5, 0.5, "Thiếu cột Boxes_Shipped hoặc Amount", ha='center', va='center')

        # Bản đồ nhiệt (Heatmap) độ tương quan 4 biến
        target_cols = ['Boxes_Shipped', 'Marketing_Spend', 'Discount_Pct', 'Amount']
        valid_cols = [c for c in target_cols if c in self.df.columns]
        
        if len(valid_cols) > 1:
            corr_matrix = self.df[valid_cols].corr()
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f", 
                linewidths=0.5, 
                ax=axes[1],
                vmin=-1, vmax=1
            )
            axes[1].set_title(" Bản Đồ Nhiệt Tương Quan (Heatmap)", fontsize=11, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, "Thiếu dữ liệu vẽ Heatmap", ha='center', va='center')

        return fig


# ==============================================================================
# GIAO DIỆN STREAMLIT - TAB "TRỰC QUAN HÓA"
# ==============================================================================

def render_tab_4(df_clean: pd.DataFrame):
    """
    Hàm dựng giao diện Tab 4 cho ứng dụng Streamlit.
    """
    st.header("📈 Tab 4: Trực Quan Hóa Dữ Liệu")
    st.caption("Module `src/visualizer.py` - Lập trình bởi Thành viên 4")
    
    if df_clean is None or df_clean.empty:
        st.warning("⚠️ Chưa có dữ liệu sạch từ Cleaner. Vui lòng kiểm tra lại luồng chạy.")
        return

    try:
        # Khởi tạo đối tượng DataVisualizer
        visualizer = DataVisualizer(df_clean)
    except Exception as e:
        st.error(f"Lỗi khởi tạo biểu đồ: {e}")
        return

    st.markdown("---")

    # Giao diện UI: Tạo các hộp chọn (Dropdown) theo đúng phân công
    chart_choice = st.selectbox(
        "📊 Chọn thuộc tính / loại biểu đồ muốn xem trực quan:",
        [
            "1. Biểu đồ đường - Xu hướng doanh thu theo Tháng",
            "2. Biểu đồ đường - Xu hướng doanh thu theo Quý",
            "3. Biểu đồ thanh ngang - Doanh thu theo Quốc gia (Country)",
            "4. Biểu đồ thanh ngang - Top 10 Sản phẩm có doanh thu cao nhất (Product)",
            "5. Biểu đồ tròn - Tỷ trọng doanh thu theo Kênh bán hàng (Channel)",
            "6. Biểu đồ phân tán & Bản đồ nhiệt tương quan (Scatter Plot & Heatmap)"
        ],
        key="member_4_select_box"
    )

    st.markdown("### 🖼️ Kết Quả Hiển Thị Biểu Đồ")

    # Render biểu đồ tương ứng với lựa chọn của người dùng
    # Sử dụng 'ME' và 'QE' thay vì 'M' và 'Q' để tương thích với Pandas phiên bản mới
    if "Tháng" in chart_choice:
        st.pyplot(visualizer.plot_line_chart(freq='ME'))
    elif "Quý" in chart_choice:
        st.pyplot(visualizer.plot_line_chart(freq='QE'))
    elif "Quốc gia" in chart_choice:
        st.pyplot(visualizer.plot_bar_chart(group_by='Country'))
    elif "Sản phẩm" in chart_choice:
        st.pyplot(visualizer.plot_bar_chart(group_by='Product'))
    elif "Kênh bán hàng" in chart_choice:
        st.pyplot(visualizer.plot_pie_chart())
    elif "tương quan" in chart_choice:
        st.pyplot(visualizer.plot_scatter_and_heatmap())

# =====================================================================
# BẢN DEMO ĐỘC LẬP: CHẠY FILE NÀY ĐỂ TEST TRỰC QUAN HÓA TRÊN TRÌNH DUYỆT
# Lệnh chạy: streamlit run data_visualizer_2.py
# =====================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Test Visualizer", layout="wide")
    st.title("Giao Diện Test Độc Lập - Thành Viên 4")
    
    # 1. Tạo dữ liệu giả (Mock Data) để mô phỏng đầu ra của DataCleaner
    # Dữ liệu giả này đảm bảo có đủ các cột mà Visualizer cần vẽ
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
    
    mock_data = pd.DataFrame({
        'Order_Date': dates,
        'Amount': np.random.randint(1000, 5000, 100),
        'Boxes_Shipped': np.random.randint(10, 100, 100),
        'Country': np.random.choice(['USA', 'UK', 'Canada', 'Australia'], 100),
        'Product': np.random.choice(['Mint', 'Dark', 'Milk', 'White', 'Caramel'], 100),
        'Channel': np.random.choice(['Retail', 'Online', 'Wholesale'], 100),
        'Marketing_Spend': np.random.randint(100, 500, 100),
        'Discount_Pct': np.random.uniform(0.0, 0.2, 100)
    })
    
    st.success("Đã tạo dữ liệu giả lập thành công!")
    st.dataframe(mock_data.head())
    
    # 2. Gọi hàm render_tab_4 truyền dữ liệu giả vào
    # Nếu code chạy tốt ở đây, nó chắc chắn sẽ chạy tốt khi ghép vào Main App
    render_tab_4(mock_data)
