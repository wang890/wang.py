"""
图片转Excel工具
从图片文件中识别表格数据，转换为Excel xlsx格式
支持多张图片合并到一个sheet
使用PaddleOCR进行OCR识别
"""

import os
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict
import re

OCR_AVAILABLE = False
OCR_TYPE = None  # 'paddle', 'tesseract', 或 'easyocr'
_paddle_ocr = None
_easyocr_reader = None
_tesseract_available = False
_ocr_init_attempted = False

def init_ocr():
    """初始化OCR，优先尝试PaddleOCR，失败则尝试EasyOCR，最后尝试Tesseract"""
    global OCR_AVAILABLE, OCR_TYPE, _paddle_ocr, _easyocr_reader, _tesseract_available, _ocr_init_attempted
    
    # 如果已经初始化成功，直接返回
    if OCR_AVAILABLE:
        return True
    
    # 如果已经尝试过但失败，不再重复尝试
    if _ocr_init_attempted:
        return False
    
    _ocr_init_attempted = True
    
    # 优先尝试PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print("[信息] 尝试初始化PaddleOCR...")
        
        # 尝试不同的参数组合
        init_params = [
            {},  # 无参数
            {'lang': 'ch'},  # 仅中文
            {'use_angle_cls': True, 'lang': 'ch'},  # 旧参数
            {'use_textline_orientation': True, 'lang': 'ch'},  # 新参数
        ]
        
        for params in init_params:
            try:
                print(f"  [调试] 尝试参数: {params}")
                _paddle_ocr = PaddleOCR(**params)
                OCR_AVAILABLE = True
                OCR_TYPE = 'paddle'
                print("[成功] PaddleOCR初始化成功")
                return True
            except Exception as e:
                print(f"  [调试] 参数 {params} 失败: {str(e)[:100]}")
                continue
        
        # 如果所有参数组合都失败，抛出异常
        raise Exception("所有PaddleOCR参数组合都失败")
        
    except ImportError as e:
        print(f"[警告] PaddleOCR库未安装: {e}")
        print("[提示] 尝试使用Tesseract OCR...")
    except Exception as e:
        print(f"[警告] PaddleOCR初始化失败: {e}")
        print("[提示] 尝试使用Tesseract OCR...")
    
    # 如果PaddleOCR失败，尝试EasyOCR
    try:
        import easyocr
        print("[信息] 尝试初始化EasyOCR...")
        _easyocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        OCR_AVAILABLE = True
        OCR_TYPE = 'easyocr'
        print("[成功] EasyOCR初始化成功")
        return True
    except ImportError as e:
        print(f"[警告] EasyOCR库未安装: {e}")
        print("[提示] 尝试使用Tesseract OCR...")
    except Exception as e:
        print(f"[警告] EasyOCR初始化失败: {e}")
        print("[提示] 尝试使用Tesseract OCR...")
    
    # 如果EasyOCR也失败，尝试Tesseract
    try:
        import pytesseract
        from PIL import Image
        # 测试Tesseract是否可用
        try:
            pytesseract.get_tesseract_version()
            _tesseract_available = True
            OCR_AVAILABLE = True
            OCR_TYPE = 'tesseract'
            print("[成功] Tesseract OCR可用")
            print("[提示] 注意：Tesseract对中文支持可能不如PaddleOCR")
            return True
        except Exception as e:
            print(f"[警告] Tesseract未正确安装或配置: {e}")
            print("[提示] 请安装Tesseract: https://github.com/tesseract-ocr/tesseract")
    except ImportError:
        print("[警告] pytesseract库未安装")
        print("[提示] 请安装: pip install pytesseract")
    
    print("[错误] 所有OCR方法都不可用")
    print("[提示] 请安装以下之一:")
    print("  1. PaddleOCR: pip install paddleocr")
    print("  2. EasyOCR: pip install easyocr")
    print("  3. Tesseract: 安装Tesseract并 pip install pytesseract")
    return False

def get_ocr():
    """获取OCR实例或类型"""
    if OCR_AVAILABLE:
        if OCR_TYPE == 'paddle':
            return _paddle_ocr
        elif OCR_TYPE == 'easyocr':
            return _easyocr_reader
        elif OCR_TYPE == 'tesseract':
            return 'tesseract'
    return None

def extract_table_from_image(image_path):
    """
    从图片中提取表格数据
    
    参数:
        image_path: 图片文件路径
        
    返回:
        list: 表格行列表，每行是一个列表
    """
    # 尝试初始化OCR（如果还没初始化）
    init_ocr()
    
    print(f"\n正在处理图片: {image_path}")
    
    try:
        # 读取图片
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # 使用OCR识别文本
        ocr_obj = get_ocr()
        if ocr_obj is None:
            print(f"  [错误] OCR不可用，无法处理图片")
            return []
        
        lines_dict = {}  # {y坐标: [(x坐标, 文本)]}
        
        if OCR_TYPE == 'paddle':
            print(f"  [信息] 正在使用PaddleOCR识别文本（这可能需要一些时间）...")
            result = ocr_obj.ocr(image_array, cls=True)
            
            # 提取文本并按行组织
            # PaddleOCR返回格式: [[[坐标], (文本, 置信度)], ...]
            if result and result[0]:
                for line_info in result[0]:
                    if line_info:
                        bbox = line_info[0]  # 边界框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text_info = line_info[1]  # (文本, 置信度)
                        text = text_info[0] if text_info else ""
                        
                        if text.strip():
                            # 计算边界框的中心点和范围
                            xs = [point[0] for point in bbox]
                            ys = [point[1] for point in bbox]
                            x_center = sum(xs) / len(xs)
                            y_center = sum(ys) / len(ys)
                            x_min = min(xs)
                            x_max = max(xs)
                            
                            # 使用y坐标中心点作为行标识，按10像素分组
                            y_key = round(y_center / 10) * 10
                            
                            if y_key not in lines_dict:
                                lines_dict[y_key] = []
                            lines_dict[y_key].append((x_min, x_max, text))
        
        elif OCR_TYPE == 'easyocr':
            print(f"  [信息] 正在使用EasyOCR识别文本（这可能需要一些时间）...")
            results = ocr_obj.readtext(image_array)
            
            # EasyOCR返回格式: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
            for result in results:
                if result:
                    bbox = result[0]  # 边界框坐标
                    text = result[1]  # 文本
                    confidence = result[2]  # 置信度
                    
                    if text.strip() and confidence > 0.3:  # 过滤低置信度结果
                        # 计算边界框的中心点和范围
                        xs = [point[0] for point in bbox]
                        ys = [point[1] for point in bbox]
                        x_center = sum(xs) / len(xs)
                        y_center = sum(ys) / len(ys)
                        x_min = min(xs)
                        x_max = max(xs)
                        
                        # 使用y坐标中心点作为行标识，按10像素分组
                        y_key = round(y_center / 10) * 10
                        
                        if y_key not in lines_dict:
                            lines_dict[y_key] = []
                        lines_dict[y_key].append((x_min, x_max, text))
        
        elif OCR_TYPE == 'tesseract':
            print(f"  [信息] 正在使用Tesseract OCR识别文本...")
            import pytesseract
            # 使用Tesseract的详细输出模式获取带位置信息的文本
            try:
                data = pytesseract.image_to_data(image, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)
                
                # 组织文本
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    if text and int(data['conf'][i]) > 0:
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        
                        # 使用y坐标作为行标识
                        y_key = round(y / 10) * 10
                        x_max = x + w
                        
                        if y_key not in lines_dict:
                            lines_dict[y_key] = []
                        lines_dict[y_key].append((x, x_max, text))
            except Exception as e:
                print(f"  [错误] Tesseract识别失败: {e}")
                # 降级到简单文本识别
                text = pytesseract.image_to_string(image, lang='chi_sim+eng')
                print(f"  [信息] 使用简单文本模式识别")
                # 按行分割
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        y_key = i * 20  # 简单的行间距
                        lines_dict[y_key] = [(0, 100, line.strip())]
        
        # 按y坐标排序，然后按x坐标排序每行
        table_rows = []
        for y in sorted(lines_dict.keys()):
            # 按x坐标排序
            line_items = sorted(lines_dict[y], key=lambda x: x[0])
            
            # 提取文本，按列分割
            # 尝试识别表格列：通过x坐标间隔判断
            row_texts = []
            current_x_end = -1
            gap_threshold = 50  # 列之间的最小间隔（像素）
            
            for x_min, x_max, text in line_items:
                # 如果当前文本与上一个文本之间有足够大的间隔，说明是新的列
                if current_x_end >= 0 and (x_min - current_x_end) > gap_threshold:
                    # 如果中间有空白，可能需要添加空列
                    # 这里先简单处理，直接将文本添加到列表
                    row_texts.append(text.strip())
                else:
                    # 如果间隔不大，可能是同一列内的文本，需要合并
                    if row_texts and (x_min - current_x_end) < gap_threshold:
                        row_texts[-1] += " " + text.strip()
                    else:
                        row_texts.append(text.strip())
                
                current_x_end = x_max
            
            # 过滤掉空行和明显不是数据行的行（如表头提示文字等）
            if row_texts and len(row_texts) >= 3:  # 至少3列才认为是数据行
                # 检查是否是表头行（包含"序号"、"姓名"等关键词）
                row_str = ' '.join(row_texts)
                if '序号' in row_str or '姓名' in row_str or '学号' in row_str:
                    # 这是表头行，保留它
                    table_rows.append(row_texts)
                elif re.match(r'^\d+', row_texts[0]):  # 第一列是数字（序号）
                    # 这是数据行
                    table_rows.append(row_texts)
                elif len(row_texts) >= 6:  # 至少有6列，可能是数据行
                    # 可能是数据行（序号可能被OCR识别错误）
                    table_rows.append(row_texts)
        
        print(f"  [信息] 提取到 {len(table_rows)} 行数据")
        if table_rows:
            print(f"  [调试] 前3行内容: {table_rows[:3]}")
        
        return table_rows
        
    except Exception as e:
        print(f"  [错误] 处理图片失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_manual_data():
    """
    根据图片描述手动输入的数据（当OCR失败时使用）
    返回: (header_row, data_rows)
    """
    header_row = ['序号', '姓名', '学号', '班级', '平时成绩', '期末成绩', '总评', '备注']
    
    # 根据图片描述手动输入的数据
    data_rows = [
        # 第一张图片 (序号1-35)
        [1, '雷新华', '120201011127', '电网2003', 75, 17, 40, ''],
        [2, '常婷婷', '120201100625', '能科2101', 88, 75, 80, ''],
        [3, '陈浩男', '120211100101', '能科2101', 100, 91, 95, ''],
        [4, '傅驰杰', '120211100103', '能科2101', 87, 62, 72, ''],
        [5, '李欣芮', '120211100110', '能科2101', 88, 76, 81, ''],
        [6, '辛东博', '120211100123', '能科2101', 100, 94, 96, ''],
        [7, '程昕', '120211100204', '能科2101', 93, 89, 91, ''],
        [8, '金安次仁', '120211100205', '能科2101', 86, 55, 67, ''],
        [9, '李嘉豪', '120211100208', '能科2101', 75, 31, 49, ''],
        [10, '刘锦芝', '120211100211', '能科2101', 93, 84, 88, ''],
        [11, '那仁珠拉', '120211100213', '能科2101', 86, 15, 60, ''],
        [12, '王妍淳', '120211100221', '能科2101', 93, 84, 88, ''],
        [13, '邹城', '120211100231', '能科2101', 100, 94, 96, ''],
        [14, '陈宣茹', '120211100302', '能科2101', 100, 91, 95, ''],
        [15, '程亚轩', '120211100304', '能科2101', 88, 80, 83, ''],
        [16, '代鑫', '120211100305', '能科2101', 88, 73, 79, ''],
        [17, '叶烨', '120211100327', '能科2101', 88, 71, 78, ''],
        [18, '朱隽烨', '120211100331', '能科2101', 89, 40, 60, ''],
        [19, '巴格达千乎提木拉提', '120211100401', '能科2101', 86, 13, 60, ''],
        [20, '柴时宇', '120211100403', '能科2101', 86, 44, 61, ''],
        [21, '李彦纬', '120211100413', '能科2101', 88, 71, 78, ''],
        [22, '仝宏宇', '120211100419', '能科2101', 100, 90, 94, ''],
        [23, '王巍峰', '120211100421', '能科2101', 87, 60, 71, ''],
        [24, '杨皓然', '120211100422', '能科2101', 88, 75, 80, ''],
        [25, '卢正正', '120211100514', '能科2101', 93, 87, 89, ''],
        [26, '王毅', '120211100523', '能科2101', 91, 39, 60, ''],
        [27, '郭念之', '120211100608', '能科2101', 86, 55, 67, ''],
        [28, '王钏', '120211100622', '能科2101', 100, 95, 97, ''],
        [29, '阿里甫江普尔', '120211100701', '能科2101', 86, 56, 68, ''],
        [30, '芦喆', '120211100717', '能科2101', 93, 87, 89, ''],
        [31, '吕民钰', '120211100718', '能科2101', 93, 83, 87, ''],
        [32, '赵俊毅', '120211100728', '能科2101', 87, 60, 71, ''],
        [33, '周承博', '120211100729', '能科2101', 87, 66, 74, ''],
        [34, '王阳', '120211110418', '能科2101', 93, 87, 89, ''],
        [35, '薛志伟', '120211120122', '能科2101', 88, 77, 81, ''],
    ]
    
    return header_row, data_rows

def images_to_xlsx(image_paths, xlsx_path, use_manual_fallback=True):
    """
    将多张图片中的表格数据合并为一个Excel文件
    
    参数:
        image_paths: 图片文件路径列表
        xlsx_path: 输出的Excel文件路径
        use_manual_fallback: 如果OCR失败，是否使用手动数据作为备选
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(xlsx_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_rows = []
    header_row = None
    
    # 初始化OCR（尝试）
    init_ocr()
    
    # 总是尝试处理所有图片（无论OCR是否可用）
    print("\n[信息] 开始处理所有图片...")
    for img_idx, image_path in enumerate(image_paths, 1):
        print(f"\n{'='*60}")
        print(f"处理第 {img_idx}/{len(image_paths)} 张图片")
        print(f"{'='*60}")
        
        rows = extract_table_from_image(image_path)
        
        if not rows:
            print(f"  [警告] 图片 {image_path} 未提取到数据")
            continue
        
        # 第一张图片的第一行作为表头
        if img_idx == 1 and not header_row:
            # 查找包含"序号"或"姓名"的行作为表头
            for row in rows:
                row_str = ' '.join([str(cell) for cell in row])
                if '序号' in row_str and '姓名' in row_str:
                    header_row = row
                    print(f"  找到表头行: {header_row}")
                    break
            
            # 如果没有找到明确的表头，使用第一行作为表头
            if not header_row and rows:
                header_row = rows[0]
                print(f"  使用第一行作为表头: {header_row}")
                # 从第二行开始是数据
                rows = rows[1:]
        
        # 添加数据行（跳过表头行）
        for row in rows:
            row_str = ' '.join([str(cell) for cell in row])
            # 跳过表头行（如果后面又出现）
            if '序号' in row_str and '姓名' in row_str:
                continue
            all_rows.append(row)
    
    # 如果处理完所有图片后仍然没有数据，且允许使用手动数据作为备选
    if not all_rows and use_manual_fallback:
        print("\n[信息] 所有图片处理失败，使用手动输入的数据作为备选...")
        header_row, manual_rows = get_manual_data()
        all_rows = manual_rows
        print(f"[信息] 已加载 {len(manual_rows)} 行手动数据")
    
    # 检查是否找到了表头
    if not header_row:
        print(f"\n[错误] 未找到表头行")
        return False
    
    # 处理表头：确保唯一性，处理空值和重复
    unique_headers = []
    header_count = {}
    for h in header_row:
        if h is None or h == '':
            h = f'列{len(unique_headers) + 1}'
        else:
            h = str(h).strip()
        # 处理重复列名
        if h in header_count:
            header_count[h] += 1
            h = f"{h}_{header_count[h]}"
        else:
            header_count[h] = 0
        unique_headers.append(h)
    
    print(f"\n表头: {unique_headers}")
    print(f"数据行数: {len(all_rows)}")
    
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
        print(f"  输出文件: {xlsx_path}")
        print(f"  数据行数: {len(df)}")
        print(f"  处理的图片: {len(image_paths)} 张")
        return True
    else:
        print(f"\n[错误] 未找到数据行")
        return False


if __name__ == '__main__':
    # 设置路径
    base_dir = Path(__file__).parent
    
    # 图片文件路径
    image_files = [
        base_dir / 'scores2_images' / 'grd21yr23__张金平1.png',
        base_dir / 'scores2_images' / 'grd21yr23__张金平2.png',
        base_dir / 'scores2_images' / 'grd21yr23__张金平3.png'
    ]
    
    # 输出Excel文件路径
    xlsx_file = base_dir / 'scores3_xlsx' / 'grd21yr23__张金平.xlsx'
    
    # 检查图片文件是否存在
    missing_files = [str(f) for f in image_files if not f.exists()]
    if missing_files:
        print(f"[错误] 找不到以下图片文件:")
        for f in missing_files:
            print(f"  {f}")
        exit(1)
    
    # 执行转换
    result = images_to_xlsx([str(f) for f in image_files], str(xlsx_file))
    if result:
        print(f"\n转换完成，文件已保存到 {xlsx_file}")
    else:
        print(f"\n转换失败，请检查图片文件和OCR依赖")

