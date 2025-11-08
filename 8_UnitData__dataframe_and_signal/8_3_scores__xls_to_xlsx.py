"""
XLS 转 XLSX 工具
将 xls 文件转换为 xlsx 格式，保存到 scores3_xlsx 文件夹
xlsx 文件名为 xls 文件名的前6个字符
"""

import os
import pandas as pd
from pathlib import Path


def xls_to_xlsx(xls_path, xlsx_dir):
    """
    将 xls 文件转换为 xlsx 格式
    
    参数:
        xls_path: 输入的 xls 文件路径
        xlsx_dir: 输出的 xlsx 文件目录
    """
    # 确保输出目录存在
    if not os.path.exists(xlsx_dir):
        os.makedirs(xlsx_dir)
    
    # 获取xls文件名（不含扩展名）
    xls_filename = os.path.basename(xls_path)
    xls_name_without_ext = os.path.splitext(xls_filename)[0]
    
    # 取前6个字符作为新文件名
    new_filename = xls_name_without_ext[:14] + '.xlsx'
    xlsx_path = os.path.join(xlsx_dir, new_filename)
    
    print(f"\n[信息] 正在处理: {xls_filename}")
    print(f"  输出文件: {new_filename}")
    
    try:
        # 直接使用 xlrd 读取 .xls 文件（绕过 pandas 的版本检查）
        import xlrd
        
        # 打开工作簿
        workbook = xlrd.open_workbook(xls_path)
        
        # 读取第一个工作表
        sheet = workbook.sheet_by_index(0)
        
        # 提取数据
        data = []
        for row_idx in range(sheet.nrows):
            row = []
            for col_idx in range(sheet.ncols):
                cell_value = sheet.cell_value(row_idx, col_idx)
                row.append(cell_value)
            data.append(row)
        
        # 转换为 DataFrame
        if data:
            # 第一行作为表头
            if len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
            else:
                df = pd.DataFrame(data)
        else:
            df = pd.DataFrame()
        
        # 保存为 xlsx 文件
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
        
        print(f"  [成功] 转换完成！")
        print(f"  数据行数: {len(df)}")
        print(f"  数据列数: {len(df.columns)}")
        return True
        
    except Exception as e:
        print(f"  [错误] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_convert_xls_to_xlsx(xls_dir, xlsx_dir):
    """
    批量将指定目录下的所有 xls 文件转换为 xlsx 格式
    
    参数:
        xls_dir: 包含 xls 文件的目录
        xlsx_dir: 输出的 xlsx 文件目录
    """
    xls_dir = Path(xls_dir)
    
    # 查找所有 xls 文件
    xls_files = list(xls_dir.glob('*.xls'))
    
    if not xls_files:
        print(f"[警告] 在 {xls_dir} 中未找到 xls 文件")
        return
    
    print(f"[信息] 找到 {len(xls_files)} 个 xls 文件")
    
    success_count = 0
    fail_count = 0
    
    for xls_file in xls_files:
        if xls_to_xlsx(str(xls_file), xlsx_dir):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n[完成] 批量转换完成！")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {fail_count} 个文件")


if __name__ == '__main__':
    # 设置路径
    base_dir = Path(__file__).parent
    
    # xls 文件所在目录（scores1_raw）
    xls_dir = base_dir / 'scores1_raw'
    
    # 输出的 xlsx 文件目录
    xlsx_dir = base_dir / 'scores3_xlsx'
    
    # 检查 xls 目录是否存在
    if not xls_dir.exists():
        print(f"[错误] 找不到目录: {xls_dir}")
        exit(1)
    
    # 执行批量转换
    batch_convert_xls_to_xlsx(str(xls_dir), str(xlsx_dir))

    print("转换完成")

