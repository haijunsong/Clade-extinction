import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 确保使用兼容的后端
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def calculate_correlation(input_file):
    try:
        # 读取数据表格
        data = pd.read_excel(input_file)  # 读取Excel文件
        print("数据加载成功，数据预览：")
        print(data.head())

        # 筛选出数值类型的列
        numeric_data = data.select_dtypes(include=[float, int])
        print("\n筛选后的数值数据预览：")
        print(numeric_data.head())

        # 获取数值数据的列名
        columns = numeric_data.columns
        num_columns = len(columns)

        # 创建一个空的DataFrame来存储相关系数和p值
        correlation_matrix = pd.DataFrame(np.zeros((num_columns, num_columns)), index=columns, columns=columns)
        p_value_matrix = pd.DataFrame(np.zeros((num_columns, num_columns)), index=columns, columns=columns)

        # 计算每两列之间的相关系数和p值
        for i in range(num_columns):
            for j in range(i + 1, num_columns):
                col1 = columns[i]
                col2 = columns[j]
                # 确保列中没有缺失值
                valid_data = numeric_data[[col1, col2]].dropna()
                if len(valid_data) > 2:  # 至少需要2个数据点才能计算相关性
                    r, p = pearsonr(valid_data[col1], valid_data[col2])
                    correlation_matrix.loc[col1, col2] = r
                    correlation_matrix.loc[col2, col1] = r
                    p_value_matrix.loc[col1, col2] = p
                    p_value_matrix.loc[col2, col1] = p
                else:
                    print(f"警告：列 {col1} 和 {col2} 的有效数据点不足，无法计算相关性。")

        # 打印结果
        print("\n相关系数矩阵：")
        print(correlation_matrix)
        print("\np值矩阵：")
        print(p_value_matrix)

        # 保存结果到Excel文件
        with pd.ExcelWriter("correlation_results.xlsx") as writer:
            correlation_matrix.to_excel(writer, sheet_name="Correlation Coefficients")
            p_value_matrix.to_excel(writer, sheet_name="P Values")

        # 可视化
        visualize_correlation(correlation_matrix, p_value_matrix)

    except Exception as e:
        print(f"发生错误：{e}")

def visualize_correlation(correlation_matrix, p_value_matrix):
    # 获取列名
    columns = correlation_matrix.columns
    num_columns = len(columns)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 12))

    # 设置坐标轴范围
    ax.set_xlim(-0.5, num_columns - 0.5)
    ax.set_ylim(-0.5, num_columns - 0.5)

    # 绘制椭圆
    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            r = correlation_matrix.iloc[i, j]
            p = p_value_matrix.iloc[i, j]

            # 计算椭圆的偏平程度
            width = 1.0
            height = abs(r) * width

            # 设置颜色
            color = 'blue' if p <= 0.05 else 'none'

            # 绘制椭圆
            ellipse = Ellipse((i, j), width, height, facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(ellipse)

            # 添加文本标注
            ax.text(i, j, f"{r:.2f}\n{p:.3f}", ha='center', va='center', fontsize=8)

    # 设置坐标轴刻度
    ax.set_xticks(range(num_columns))
    ax.set_yticks(range(num_columns))
    ax.set_xticklabels(columns, rotation=90)
    ax.set_yticklabels(columns)

    # 添加标题
    ax.set_title("Correlation and Significance Visualization", fontsize=14)

    # 保存图形到文件
    plt.tight_layout()
    plt.savefig("correlation_visualization.png")  # 保存图形
    print("图形已保存到 'correlation_visualization.png'")

# 示例用法
if __name__ == "__main__":
    input_file = r"F:\1-投稿\2022-clades灭绝\PBDB Data\SQS results\SQS result vs Environmental factros v0.4.xlsx"
    calculate_correlation(input_file)