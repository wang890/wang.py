"""
PDF 转 Excel 工具
将 PDF 文件中的表格数据提取并转换为 Excel xlsx 格式
"""

import os
import pandas as pd
import pdfplumber
from pathlib import Path


def pdf_to_xlsx(pdf_path, xlsx_path):
    """
    将 PDF 文件中的表格转换为 Excel xlsx 格式
    
    参数:
        pdf_path: PDF 文件路径
        xlsx_path: 输出的 Excel 文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(xlsx_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 存储所有提取的表格
    all_tables = []
    
    # 打开 PDF 文件
    with pdfplumber.open(pdf_path) as pdf:
        # 遍历每一页
        for page_num, page in enumerate(pdf.pages, start=1):
            print(f"正在处理第 {page_num} 页...")
            
            # 提取页面中的表格
            tables = page.extract_tables()
            
            if tables:
                for table_num, table in enumerate(tables, start=1):
                    if table:  # 确保表格不为空
                        # 将表格转换为 DataFrame
                        # 第一行作为列名
                        if len(table) > 1:
                            # 处理列名：确保唯一性，处理空值和重复
                            headers = table[0]
                            # 处理空值和重复列名
                            unique_headers = []
                            header_count = {}
                            for h in headers:
                                if h is None or h == '':
                                    h = f'列{len(unique_headers) + 1}'
                                # 处理重复列名
                                if h in header_count:
                                    header_count[h] += 1
                                    h = f"{h}_{header_count[h]}"
                                else:
                                    header_count[h] = 0
                                unique_headers.append(h)
                            
                            df = pd.DataFrame(table[1:], columns=unique_headers)
                            all_tables.append(df)
                            print(f"  找到表格 {table_num}，包含 {len(df)} 行数据")
                        elif len(table) == 1:
                            # 如果只有一行，作为列名
                            headers = table[0]
                            unique_headers = []
                            header_count = {}
                            for h in headers:
                                if h is None or h == '':
                                    h = f'列{len(unique_headers) + 1}'
                                if h in header_count:
                                    header_count[h] += 1
                                    h = f"{h}_{header_count[h]}"
                                else:
                                    header_count[h] = 0
                                unique_headers.append(h)
                            df = pd.DataFrame(columns=unique_headers)
                            all_tables.append(df)
                            print(f"  找到表格 {table_num}（仅列名）")
    
    # 合并所有表格（如果有多个）
    if all_tables:
        # 获取所有表格的所有列名，确保列名统一
        all_columns = set()
        for df in all_tables:
            all_columns.update(df.columns)
        all_columns = sorted(list(all_columns))
        
        # 为每个 DataFrame 添加缺失的列（填充 NaN）
        for i, df in enumerate(all_tables):
            missing_cols = set(all_columns) - set(df.columns)
            if missing_cols:
                for col in missing_cols:
                    df[col] = None
        
        # 重新排列列顺序，确保所有表格列顺序一致
        all_tables = [df[all_columns] for df in all_tables]
        
        # 合并所有 DataFrame
        combined_df = pd.concat(all_tables, ignore_index=True, sort=False)
        
        # 保存为 Excel 文件
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='成绩单', index=False)
        
        print(f"\n[成功] 转换完成！")
        print(f"  输入文件: {pdf_path}")
        print(f"  输出文件: {xlsx_path}")
        print(f"  数据行数: {len(combined_df)}")
        return True
    else:
        print(f"\n[错误] 未在 PDF 中找到表格数据")
        return False


if __name__ == '__main__':
    # 设置路径
    base_dir = Path(__file__).parent
    pdf_file = base_dir / 'scores1_raw' / 'grd21yr23__何凤霞_2022-2023 (2) 《概率论与数理统计B》课程-学生成绩单.pdf'
    xlsx_file = base_dir / 'scores3_xlsx' / 'grd21yr23__何凤霞.xlsx'
    
    # 检查 PDF 文件是否存在
    if not pdf_file.exists():
        print(f"[错误] 找不到 PDF 文件: {pdf_file}")
        exit(1)
    
    # 执行转换
    pdf_to_xlsx(str(pdf_file), str(xlsx_file))
    print(f"转换完成，文件已保存到 {xlsx_file}")
