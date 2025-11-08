"""
PDF 转 Excel 工具（多页面范围提取版本）
从 PDF 文件中按页面范围提取表格数据，转换为 Excel xlsx 格式
支持为每个页面指定不同的起始和终止关键词
基于 8_1_scores__pdf_to_xlsx2.py 改进
支持OCR处理扫描版PDF
"""

import os
import pandas as pd
import pdfplumber
from pathlib import Path
import io
import numpy as np

OCR_AVAILABLE = False
_paddle_ocr = None
_ocr_init_attempted = False

def init_ocr():
    """初始化OCR，如果失败则返回False（只尝试一次）"""
    global OCR_AVAILABLE, _paddle_ocr, _ocr_init_attempted
    
    # 如果已经初始化成功，直接返回
    if OCR_AVAILABLE:
        return True
    
    # 如果已经尝试过但失败，不再重复尝试
    if _ocr_init_attempted:
        return False
    
    _ocr_init_attempted = True
    
    try:
        from paddleocr import PaddleOCR
        from pdf2image import convert_from_path
        # 使用新版本参数
        _paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='ch')  # 中文OCR
        OCR_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"[警告] OCR库未安装: {e}")
        print("[提示] 扫描版PDF需要OCR支持。请安装: pip install paddleocr pdf2image")
        return False
    except Exception as e:
        print(f"[警告] OCR初始化失败: {e}")
        print("[提示] OCR功能暂时不可用，将尝试使用常规方法提取文本")
        print("[提示] 如果PDF是扫描版，可能需要修复OCR依赖或使用其他OCR工具")
        return False

def get_paddle_ocr():
    """获取PaddleOCR实例"""
    if OCR_AVAILABLE:
        return _paddle_ocr
    return None


def pdf_to_xlsx(pdf_path, xlsx_path, page_ranges):
    """
    将 PDF 文件中的表格转换为 Excel xlsx 格式
    按页面范围提取数据，每个页面可以有不同的起始和终止关键词
    
    参数:
        pdf_path: PDF 文件路径
        xlsx_path: 输出的 Excel 文件路径
        page_ranges: 字典，格式为 {页码: {"start": "起始关键词", "end": "终止关键词"}}
                     页码从1开始
                     例如: {1: {"start": "姓名", "end": "薛志伟"}, 
                           2: {"start": "李静戈", "end": "何宇飞"}}
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(xlsx_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 存储所有提取的表格行
    all_rows = []
    header_row = None
    header_found = False
    
    # 打开 PDF 文件
    with pdfplumber.open(pdf_path) as pdf:
        pages_list = list(pdf.pages)  # 转换为列表
        
        # 按页码顺序处理每个页面
        for page_num in sorted(page_ranges.keys()):
            if page_num < 1 or page_num > len(pages_list):
                print(f"[警告] 页码 {page_num} 超出范围，跳过")
                continue
            
            page = pages_list[page_num - 1]
            range_config = page_ranges[page_num]
            start_keyword = range_config.get("start", "")
            end_keyword = range_config.get("end", "")
            
            print(f"\n正在处理第 {page_num} 页...")
            print(f"  起始关键词: {start_keyword}")
            print(f"  终止关键词: {end_keyword}")
            
            # 提取页面中的表格（使用与8_1_scores__pdf_to_xlsx2.py相同的方法）
            tables = page.extract_tables()
            
            # 检查表格是否有实际内容（非空单元格）
            has_content = False
            if tables:
                for table in tables:
                    if table and any(any(cell and str(cell).strip() for cell in row) for row in table):
                        has_content = True
                        break
            
            # 如果表格为空或没有内容，尝试其他方法
            if not tables or not has_content:
                # 方法1：尝试使用pdfplumber的chars提取字符
                print(f"  [信息] 表格提取为空，尝试从字符提取...")
                try:
                    chars = page.chars
                    print(f"  [调试] 字符数量: {len(chars) if chars else 0}")
                    
                    if chars and len(chars) > 0:
                        print(f"  [信息] 找到 {len(chars)} 个字符，尝试组织成表格...")
                        # 打印前几个字符用于调试
                        if len(chars) > 0:
                            print(f"  [调试] 前10个字符: {[c.get('text', '') for c in chars[:10]]}")
                        
                        # 按y坐标分组字符（近似行）
                        from collections import defaultdict
                        lines_dict = defaultdict(list)
                        for char in chars:
                            if char.get('text'):
                                y_key = round(char['top'] / 5) * 5  # 按5像素分组
                                lines_dict[y_key].append((char['x0'], char['text']))
                        
                        print(f"  [调试] 组织成 {len(lines_dict)} 个不同的行")
                        
                        # 按y坐标排序，然后按x坐标排序每行
                        table_from_chars = []
                        for y in sorted(lines_dict.keys(), reverse=True):  # 从上到下
                            line_items = sorted(lines_dict[y], key=lambda x: x[0])
                            line_texts = [item[1] for item in line_items]
                            line_str = ''.join(line_texts)
                            
                            if line_str.strip():
                                # 尝试按空格分割成列
                                import re
                                parts = re.split(r'\s{2,}', line_str.strip())
                                parts = [p.strip() for p in parts if p.strip()]
                                
                                if len(parts) > 1:
                                    table_from_chars.append(parts)
                                else:
                                    table_from_chars.append([line_str.strip()])
                        
                        if table_from_chars:
                            tables = [table_from_chars]
                            has_content = True
                            print(f"  [信息] 从字符提取到 {len(table_from_chars)} 行")
                            # 打印前几行用于调试
                            print(f"  [调试] 前5行内容: {table_from_chars[:5]}")
                        else:
                            print(f"  [警告] 字符提取后没有有效的行")
                    else:
                        print(f"  [警告] 页面没有字符数据（可能是扫描版PDF）")
                except Exception as e:
                    print(f"  [调试] 字符提取失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 方法2：如果字符提取也失败，尝试OCR
                if not has_content and init_ocr():
                    print(f"  [信息] 字符提取失败，尝试使用OCR识别...")
                    try:
                        from pdf2image import convert_from_path
                        # 使用pdf2image将指定页面转换为图像
                        # 注意：pdf2image需要poppler，page_num是从1开始的
                        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300)
                        
                        if not images:
                            print(f"  [警告] 无法将PDF页面转换为图像（可能需要安装poppler）")
                            continue
                        
                        pil_image = images[0]
                        
                        # 使用PaddleOCR识别文本（支持中文）
                        print(f"  [信息] 正在使用PaddleOCR识别文本（这可能需要一些时间）...")
                        ocr = get_paddle_ocr()
                        if ocr is None:
                            print(f"  [错误] OCR初始化失败")
                            continue
                        result = ocr.ocr(np.array(pil_image), cls=True)
                        
                        # 提取文本并按行组织
                        # PaddleOCR返回格式: [[[坐标], (文本, 置信度)], ...]
                        lines_dict = {}  # {y坐标: [文本列表]}
                        if result and result[0]:
                            for line_info in result[0]:
                                if line_info:
                                    bbox = line_info[0]  # 边界框坐标
                                    text_info = line_info[1]  # (文本, 置信度)
                                    text = text_info[0] if text_info else ""
                                    
                                    if text.strip():
                                        # 使用边界框的y坐标中心点作为行标识
                                        y_center = (bbox[0][1] + bbox[2][1]) / 2
                                        y_key = round(y_center / 10) * 10  # 四舍五入到10的倍数
                                        
                                        if y_key not in lines_dict:
                                            lines_dict[y_key] = []
                                        lines_dict[y_key].append((bbox[0][0], text))  # (x坐标, 文本)
                            
                            # 按y坐标排序，然后按x坐标排序每行
                            table_from_ocr = []
                            for y in sorted(lines_dict.keys()):
                                line_items = sorted(lines_dict[y], key=lambda x: x[0])
                                line_texts = [item[1] for item in line_items]
                                # 将一行中的文本组合，尝试按空格分割成列
                                line_str = ' '.join(line_texts)
                                # 尝试按多个空格分割
                                import re
                                parts = re.split(r'\s{2,}', line_str)
                                parts = [p.strip() for p in parts if p.strip()]
                                
                                if len(parts) > 1:
                                    table_from_ocr.append(parts)
                                else:
                                    table_from_ocr.append([line_str])
                            
                            if table_from_ocr:
                                tables = [table_from_ocr]
                                has_content = True
                                print(f"  [信息] OCR转换为表格，共 {len(table_from_ocr)} 行")
                                # 打印前几行用于调试
                                print(f"  [调试] 前3行内容: {table_from_ocr[:3]}")
                            else:
                                print(f"  [警告] OCR未识别到文本")
                        else:
                            print(f"  [警告] OCR未识别到内容")
                    except Exception as e:
                        print(f"  [错误] OCR处理失败: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    print(f"  [警告] 第 {page_num} 页未找到表格且OCR不可用")
                    print(f"  [提示] 如果这是扫描版PDF，请:")
                    print(f"    1. 检查OCR依赖是否正确安装")
                    print(f"    2. 或尝试使用其他OCR工具（如Tesseract OCR）")
                    print(f"    3. 或先将PDF转换为可搜索的文本版PDF")
                    continue
            
            if not tables or not has_content:
                print(f"  [警告] 第 {page_num} 页无法提取数据")
                print(f"  [提示] 可能原因:")
                print(f"    1. PDF是扫描版图片，需要OCR但OCR不可用")
                print(f"    2. PDF表格结构特殊，无法自动识别")
                continue
            
            # 处理该页面的每个表格
            for table_num, table in enumerate(tables, start=1):
                if not table or len(table) == 0:
                    continue
                
                # 查找起始行
                start_row_idx = None
                for i, row in enumerate(table):
                    # 检查行中是否包含起始关键词（与8_1_scores__pdf_to_xlsx2.py相同的方法）
                    row_str = ' '.join([str(cell) if cell else '' for cell in row])
                    if start_keyword in row_str:
                        start_row_idx = i
                        # 如果是第一页且包含"姓名"，作为表头
                        if page_num == min(page_ranges.keys()) and start_keyword == "姓名" and not header_found:
                            header_row = row
                            header_found = True
                            print(f"  找到表头行（包含'{start_keyword}'）: 表格 {table_num}，第 {i+1} 行")
                        else:
                            print(f"  找到起始行（包含'{start_keyword}'）: 表格 {table_num}，第 {i+1} 行")
                        break
                
                if start_row_idx is None:
                    print(f"  [警告] 第 {page_num} 页表格 {table_num} 未找到起始关键词 '{start_keyword}'")
                    continue
                
                # 确定数据收集的起始位置
                if page_num == min(page_ranges.keys()) and start_keyword == "姓名":
                    # 第一页且起始关键词是"姓名"，表头行不作为数据，从下一行开始
                    data_start_idx = start_row_idx + 1
                else:
                    # 其他页或非"姓名"起始行，从起始行开始（包含起始行本身）
                    data_start_idx = start_row_idx
                
                # 查找终止行并收集数据
                end_found = False
                for i in range(data_start_idx, len(table)):
                    row = table[i]
                    row_str = ' '.join([str(cell) if cell else '' for cell in row])
                    
                    # 检查行中是否包含终止关键词
                    found_end = end_keyword in row_str
                    
                    # 添加当前行（包含终止行本身）
                    all_rows.append(row)
                    
                    # 如果找到终止行，停止收集
                    if found_end:
                        end_found = True
                        print(f"  找到终止行（包含'{end_keyword}'）: 表格 {table_num}，第 {i+1} 行")
                        break
                
                if not end_found:
                    print(f"  [警告] 第 {page_num} 页表格 {table_num} 未找到终止关键词 '{end_keyword}'，已提取到表格末尾")
                
                # 一个表格处理完后，停止处理该页面的其他表格，继续处理下一页
                break
    
    # 检查是否找到了表头
    if not header_found:
        print(f"\n[错误] 未找到表头行（应包含'姓名'）")
        return False
    
    # 处理表头：确保唯一性，处理空值和重复
    if header_row:
        unique_headers = []
        header_count = {}
        for h in header_row:
            if h is None or h == '':
                h = f'列{len(unique_headers) + 1}'
            # 处理重复列名
            if h in header_count:
                header_count[h] += 1
                h = f"{h}_{header_count[h]}"
            else:
                header_count[h] = 0
            unique_headers.append(h)
    else:
        print(f"\n[错误] 表头行为空")
        return False
    
    # 创建 DataFrame
    if all_rows:
        # 确保所有行的列数与表头一致
        max_cols = len(unique_headers)
        normalized_rows = []
        for row in all_rows:
            # 补齐或截断行，使其与表头列数一致
            normalized_row = list(row[:max_cols]) if len(row) >= max_cols else list(row) + [None] * (max_cols - len(row))
            normalized_rows.append(normalized_row)
        
        df = pd.DataFrame(normalized_rows, columns=unique_headers)
        
        # 保存为 Excel 文件
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='成绩单', index=False)
        
        print(f"\n[成功] 转换完成！")
        print(f"  输入文件: {pdf_path}")
        print(f"  输出文件: {xlsx_path}")
        print(f"  数据行数: {len(df)}")
        print(f"  处理的页面: {sorted(page_ranges.keys())}")
        return True
    else:
        print(f"\n[错误] 未找到数据行")
        return False


if __name__ == '__main__':
    # 设置路径
    base_dir = Path(__file__).parent
    
    # 测试：先用"何凤霞"的PDF验证代码逻辑（已知可以正常提取）
    # pdf_file = base_dir / 'scores1_raw' / 'grd21yr23__何凤霞_2022-2023 (2) 《概率论与数理统计B》课程-学生成绩单.pdf'
    # xlsx_file = base_dir / 'scores3_xlsx' / 'grd21yr23__何凤霞3_test.xlsx'
    # page_ranges = {
    #     1: {"start": "姓名", "end": "熊天乐"}
    # }
    
    # 张金平的PDF（扫描版，需要OCR）
    pdf_file = base_dir / 'scores1_raw' / 'grd21yr23__张金平_成绩单2023spring-signed.pdf'
    xlsx_file = base_dir / 'scores3_xlsx' / 'grd21yr23__张金平.xlsx'
    
    # 检查 PDF 文件是否存在
    if not pdf_file.exists():
        print(f"[错误] 找不到 PDF 文件: {pdf_file}")
        exit(1)
    
    # 定义页面范围：字典格式 {页码: {"start": "起始关键词", "end": "终止关键词"}}
    page_ranges = {
        1: {"start": "姓名", "end": "薛志伟"},
        2: {"start": "李静戈", "end": "何宇飞"},
        3: {"start": "林明浩", "end": "许文旭"}
    }
    
    # 执行转换
    result = pdf_to_xlsx(str(pdf_file), str(xlsx_file), page_ranges)
    if result:
        print(f"\n转换完成，文件已保存到 {xlsx_file}")
    else:
        print(f"\n转换失败，请检查PDF文件格式和OCR依赖")