import pandas as pd
import sys
sys.path.insert(0, '.')

df = pd.read_excel('scores3_xlsx/grd21yr23__张金平2.xlsx', engine='openpyxl')
print(f'行数: {len(df)}')
print(f'列数: {len(df.columns)}')
print('\n列名:')
print(df.columns.tolist())
print('\n前15行:')
print(df.head(15).to_string())







