import os
import csv
from datetime import datetime


# 根据日期创建一个表格，将所有api调用的结果保存到表格中
def create_csv(date):
    # 创建CSV文件名，格式为：api_calls_YYYY-MM-DD.csv
    csv_filename = f"api_calls_{date}.csv"

    # 定义CSV表头
    header = ["timestamp", "API URL", "Request data", "Response Body", "time cost"]

    # 创建CSV文件，并写入表头
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        
def save_api_call_to_csv(date, api_url, request_query, request_body, response_body):
    # 创建CSV文件名，格式为：api_calls_YYYY-MM-DD.csv
    csv_filename = f"api_calls_{date}.csv"
    if not os.path.exists(csv_filename):
        create_csv(date)

    # 打开CSV文件，并写入API调用信息
    with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        date_second = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        csv_writer.writerow([date_second, api_url, request_query, request_body, response_body])