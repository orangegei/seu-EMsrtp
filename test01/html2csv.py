from bs4 import BeautifulSoup
import pandas as pd

# 假设 `html_content` 是你从HTML文件中读取的内容
# 例如，使用 open('Financial_Stress_Index.htm', 'r', encoding='utf-8').read()
html_content = open('./data/Financial_Stress_Index.htm', 'r', encoding='utf-8').read()  # 你的HTML内容

# 使用BeautifulSoup解析HTML
soup = BeautifulSoup(html_content, 'html.parser')

# 查找HTML中的表格数据
table = soup.find('table')  # 如果有多个表格，可能需要更精确地定位
rows = table.find_all('tr')

# 提取数据
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append(cols)

# 转换为DataFrame
df = pd.DataFrame(data)

# 假设第一行是列名
df.columns = df.iloc[0]
df = df[1:]

# 保存为CSV
df.to_csv('Financial_Stress_Index.csv', index=False)
