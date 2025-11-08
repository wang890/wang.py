import pandas as pd
import sys

try:
    df = pd.read_excel('scores3_xlsx/grd21yr23__张金平2.xlsx', engine='openpyxl')
    print(f'总行数: {len(df)}')
    print(f'列数: {len(df.columns)}')
    print('\n列名:')
    print(df.columns.tolist())
    print('\n前15行数据:')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    print(df.head(15).to_string())
    print('\n最后5行数据:')
    print(df.tail(5).to_string())
except Exception as e:
    print(f'错误: {e}')
    import traceback
    traceback.print_exc()







