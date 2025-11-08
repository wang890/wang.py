"""
PDF 转 Excel 工具（精确提取版本）
从 PDF 文件中提取指定起始行和终止行之间的表格数据，转换为 Excel xlsx 格式
只提取从包含起始关键词的行到包含终止关键词的行之间的数据
"""

import os
import pandas as pd
import pdfplumber
from pathlib import Path


def pdf_to_xlsx(pdf_path, xlsx_path, start_keyword="姓名", end_keyword="熊天乐", header=None):
    """
    将 PDF 文件中的表格转换为 Excel xlsx 格式
    只提取从包含起始关键词的行到包含终止关键词的行之间的数据
    
    参数:
        pdf_path: PDF 文件路径
        xlsx_path: 输出的 Excel 文件路径
        start_keyword: 起始行的关键词（默认："姓名"）
        end_keyword: 终止行的关键词（默认："熊天乐"）
        header: 表头，可以是：
            - None: 自动检测表头（查找包含"姓名"、"序号"等关键词的行），如果找不到则生成默认列名
            - 列表: 手动指定的列名，如 ["序号", "姓名", "学号", "成绩"]
            - "auto": 强制使用默认列名（列1、列2等）
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(xlsx_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 存储所有提取的表格行
    all_rows = []
    header_row = None
    header_found = False
    start_found = False
    end_found = False
    start_page_idx = 0  # 起始行所在的页面索引
    start_table_idx = 0  # 起始行所在的表格索引
    start_row_idx = 0  # 起始行的索引
    header_page_idx = 0  # 表头所在的页面索引
    header_table_idx = 0  # 表头所在的表格索引
    header_row_idx = 0  # 表头的索引
    
    # 如果手动指定了表头
    if isinstance(header, list):
        header_row = header
        header_found = True
        print(f"使用手动指定的表头: {header}")
    
    # 打开 PDF 文件
    with pdfplumber.open(pdf_path) as pdf:
        pages_list = list(pdf.pages)  # 转换为列表以便多次遍历
        
        # 第一遍：查找表头（如果未手动指定）和起始行
        if not header_found and header != "auto":
            # 查找表头：包含"姓名"、"序号"等常见表头关键词的行
            header_keywords = ["姓名", "序号", "学号", "编号"]
            for page_num, page in enumerate(pages_list, start=1):
                if header_found:
                    break
                print(f"正在查找表头... 第 {page_num} 页...")
                
                # 提取页面中的表格
                tables = page.extract_tables()
                
                if tables:
                    for table_num, table in enumerate(tables, start=1):
                        if table and len(table) > 0:
                            # 查找表头行（包含表头关键词的行）
                            for i, row in enumerate(table):
                                row_str = ' '.join([str(cell) if cell else '' for cell in row])
                                # 检查是否包含多个表头关键词
                                keyword_count = sum(1 for kw in header_keywords if kw in row_str)
                                if keyword_count >= 2:  # 至少包含2个表头关键词
                                    header_row = row
                                    header_found = True
                                    header_page_idx = page_num - 1
                                    header_table_idx = table_num - 1
                                    header_row_idx = i
                                    print(f"  找到表头行: 第 {page_num} 页，表格 {table_num}，第 {i+1} 行")
                                    print(f"  表头内容: {header_row}")
                                    break
                        if header_found:
                            break
                if header_found:
                    break
        
        # 查找起始行（包含起始关键词的行）
        for page_num, page in enumerate(pages_list, start=1):
            if start_found:
                break
            print(f"正在查找起始行... 第 {page_num} 页...")
            
            # 提取页面中的表格
            tables = page.extract_tables()
            
            if tables:
                for table_num, table in enumerate(tables, start=1):
                    if table and len(table) > 0:
                        # 查找起始行（包含起始关键词的行）
                        for i, row in enumerate(table):
                            # 检查行中是否包含起始关键词
                            row_str = ' '.join([str(cell) if cell else '' for cell in row])
                            if start_keyword in row_str:
                                start_found = True
                                start_page_idx = page_num - 1
                                start_table_idx = table_num - 1
                                start_row_idx = i
                                print(f"  找到起始行（包含'{start_keyword}'）: 第 {page_num} 页，表格 {table_num}，第 {i+1} 行")
                                break
                        if start_found:
                            break
            
            if start_found:
                break
        
        # 第二遍：从起始行开始收集数据，直到找到终止行
        if start_found:
            print(f"开始收集数据...")
            collecting_started = False
            
            for page_num, page in enumerate(pages_list, start=1):
                if end_found:
                    break
                
                # 提取页面中的表格
                tables = page.extract_tables()
                
                if tables:
                    for table_num, table in enumerate(tables, start=1):
                        if table and len(table) > 0:
                            # 确定开始收集数据的起始位置
                            start_i = 0
                            if page_num - 1 == start_page_idx and table_num - 1 == start_table_idx:
                                # 在起始行所在的表格中
                                if header_found and header_row_idx < start_row_idx:
                                    # 如果表头在起始行之前，从起始行开始收集（跳过表头）
                                    start_i = start_row_idx
                                else:
                                    # 如果没有表头或表头在起始行之后，从起始行开始收集
                                    start_i = start_row_idx
                                collecting_started = True
                            elif collecting_started:
                                # 在后续表格中，从第一行开始
                                start_i = 0
                            else:
                                # 还没到起始行，跳过
                                continue
                            
                            # 从指定位置开始收集数据
                            for i in range(start_i, len(table)):
                                if end_found:
                                    break
                                
                                row = table[i]
                                
                                # 如果找到了表头，跳过表头行
                                if header_found and page_num - 1 == header_page_idx and table_num - 1 == header_table_idx and i == header_row_idx:
                                    continue
                                
                                row_str = ' '.join([str(cell) if cell else '' for cell in row])
                                
                                # 检查是否到达终止行
                                if end_keyword in row_str:
                                    all_rows.append(row)
                                    end_found = True
                                    print(f"  找到终止行（包含'{end_keyword}'）: 第 {page_num} 页，表格 {table_num}，第 {i+1} 行")
                                    break
                                
                                # 添加数据行
                                all_rows.append(row)
                        
                        if end_found:
                            break
                
                if end_found:
                    break
    
    # 检查是否找到了起始行和终止行
    if not start_found:
        print(f"\n[错误] 未找到包含起始关键词 '{start_keyword}' 的行")
        return False
    
    if not end_found:
        print(f"\n[警告] 未找到包含终止关键词 '{end_keyword}' 的行，将提取到文件末尾")
    
    # 不生成表头，直接使用数据行
    
    # 创建 DataFrame（不包含表头）
    if all_rows:
        # 确定最大列数
        max_cols = max(len(row) for row in all_rows) if all_rows else 0
        normalized_rows = []
        for row in all_rows:
            # 补齐或截断行，使其与最大列数一致
            normalized_row = list(row[:max_cols]) if len(row) >= max_cols else list(row) + [None] * (max_cols - len(row))
            normalized_rows.append(normalized_row)
        
        # 创建DataFrame时不指定列名（使用默认整数索引）
        df = pd.DataFrame(normalized_rows)
        
        # 保存为 Excel 文件（header=False 表示不写入表头行）
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='成绩单', index=False, header=False)
        
        print(f"\n[成功] 转换完成！")
        print(f"  输入文件: {pdf_path}")
        print(f"  输出文件: {xlsx_path}")
        print(f"  数据行数: {len(df)}")
        print(f"  起始关键词: {start_keyword}")
        print(f"  终止关键词: {end_keyword}")
        return True
    else:
        print(f"\n[错误] 未找到数据行")
        return False


if __name__ == '__main__':
    # 设置路径
    base_dir = Path(__file__).parent
    # pdf_file = base_dir / 'scores1_raw' / 'grd21yr23__何凤霞_2022-2023 (2) 《概率论与数理统计B》课程-学生成绩单.pdf'
    # xlsx_file = base_dir / 'scores3_xlsx' / 'grd21yr23__何凤霞2.xlsx'

    # pdf_file = base_dir / 'scores1_raw' / 'grd21yr23__张金平_成绩单2023spring-signed.pdf'
    # xlsx_file = base_dir / 'scores3_xlsx' / 'grd21yr23__张金平.xlsx'

    pdf_file = base_dir / 'scores1_raw' / 'grd23yr25__张金平_2024-2-25-2-概率统计成绩单下载(教学班)-signed.pdf'
    xlsx_file = base_dir / 'scores3_xlsx' / 'grd23yr25__张金平.xlsx'
    
    # 检查 PDF 文件是否存在
    if not pdf_file.exists():
        print(f"[错误] 找不到 PDF 文件: {pdf_file}")
        exit(1)
    
    # 执行转换，传入起始和终止关键词
    # 由于PDF没有表头，使用 header="auto" 自动生成默认列名
    # 或者可以手动指定表头，例如: header=["序号", "姓名", "学号", "成绩", "备注"]
    pdf_to_xlsx(str(pdf_file), str(xlsx_file), start_keyword="申楚钰", end_keyword="王欣悦", header="auto")
    print(f"转换完成，文件已保存到 {xlsx_file}")
