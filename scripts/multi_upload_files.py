import os
import sys
import aiohttp
import asyncio
import time
import re

file_folder = sys.argv[1]
kb_id = sys.argv[2]
support_end = ('.md', '.txt', '.pdf', '.jpg', '.jpeg', '.png', '.docx', '.xlsx', '.xls', '.eml', '.csv', '.pdf')
files = []


def remove_full_width_characters(s):
    # 匹配全角字符的正则表达式
    pattern = re.compile(r'[\uFF00-\uFFEF]')
    # 替换字符串中的全角字符为空字符串
    return pattern.sub('', s)


for root, dirs, file_names in os.walk(file_folder):
    for file_name in file_names:
        # print(file_name)
        if file_name.endswith(support_end):
            file_path = os.path.join(root, file_name)
            # if '新高考' in file_name:
            #     continue
            # if len(file_path) < 50:
            # print(file_path)
            files.append(file_path)
print(len(files))
response_times = []

# 设置超时时间
timeout = aiohttp.ClientTimeout(total=300)


async def send_request(round_, files):
    # if round_ != 18:
    #     return
    print(len(files))
    # url = 'http://0.0.0.0:8777/api/local_doc_qa/upload_files'
    url = 'https://qanything-local-test-265.site.youdao.com/api/local_doc_qa/upload_files'
    data = aiohttp.FormData()
    data.add_field('user_id', 'liujx_265')
    data.add_field('kb_id', kb_id)
    data.add_field('mode', 'soft')

    total_size = 0
    for file_path in files:
        # print(file_path)
        file_size = os.path.getsize(file_path)
        total_size += file_size
        data.add_field('files', open(file_path, 'rb'))
    print('size:', total_size / (1024 * 1024))
    for _ in range(1):
        try:
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=data) as response:
                    end_time = time.time()
                    response_times.append(end_time - start_time)
                    print(f"round_:{round_}, 响应状态码: {response.status}, 响应时间: {end_time - start_time}秒")
                    if response.status != 200:
                        # await asyncio.sleep(60)
                        continue
                # print(await response.json())
        except Exception as e:
            print(f"请求发送失败: {e}")


async def send_request_with_semaphore(semaphore, round_, files):
    async with semaphore:
        # semaphore.acquire() 已通过上下文管理器 async with 自动调用
        await send_request(round_, files)
        # semaphore.release() 也会在退出 async with 块时自动调用


async def create_tasks_by_size_limit(files, size_limit_mb, max_concurrent_tasks=4):
    tasks = []
    size_limit = size_limit_mb * 1024 * 1024  # 转换MB到字节
    current_batch = []
    current_size = 0

    semaphore = asyncio.Semaphore(max_concurrent_tasks)  # 创建 Semaphore 对象

    round_ = 0
    for file in files:
        file_size = os.path.getsize(file)  # 获取文件大小
        if current_size + file_size > size_limit and current_batch:
            # 当前批次添加文件后会超出大小限制, 发送当前批次
            task = asyncio.create_task(send_request_with_semaphore(semaphore, round_, current_batch))
            round_ += 1
            tasks.append(task)
            current_batch = []  # 重置批次
            current_size = 0  # 重置累计大小
        current_batch.append(file)
        current_size += file_size

    if current_batch:
        # 发送最后一批次，如果有的话
        task = asyncio.create_task(send_request_with_semaphore(semaphore, round_, current_batch))
        tasks.append(task)

    await asyncio.gather(*tasks)


async def main():
    start_time = time.time()
    await create_tasks_by_size_limit(files, 200)
    # round_ = 0
    # r_files = files[:500]
    # tasks = []
    # task = asyncio.create_task(send_request(round_, r_files))
    # tasks.append(task)
    # await asyncio.gather(*tasks)

    print(f"请求完成")
    end_time = time.time()
    # total_requests = len(response_times)
    total_time = end_time - start_time
    # qps = total_requests / total_time
    print(f"total_time:{total_time}")
    # print(f"QPS: {qps}")

    # if response_times:
    #    p90 = statistics.quantiles(response_times, n=100, method='inclusive')[-9]
    #    p95 = statistics.quantiles(response_times, n=100, method='inclusive')[-4]
    #    p99 = statistics.quantiles(response_times, n=100, method='inclusive')[-1]
    #    print(f"P90: {p90}秒")
    #    print(f"P95: {p95}秒")
    #    print(f"P99: {p99}秒")
    # else:
    #    print("没有足够的数据点来计算分位数")


if __name__ == '__main__':
    asyncio.run(main())
