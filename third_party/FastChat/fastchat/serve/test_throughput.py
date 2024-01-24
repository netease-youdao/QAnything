"""Benchmarking script to test the throughput of serving workers."""
import argparse
import json

import requests
import threading
import time

from fastchat.conversation import get_conv_template


def main():
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": args.model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    # conv = get_conv_template("vicuna_v1.1")
    conv = get_conv_template("qwen-7b-chat")
    prompt = """参考信息：
第二盘比赛中，士气正盛的白卓璇6-1大胜，总比分2-0赢得生涯大满贯首胜。 值得一提的是，这是她生涯第一次参加大满贯正赛，成为继郑钦文、吴易昺、商竣程之后第四位首次参加大满贯正赛就能闯过首轮关的中国球员。这也是她生涯中首次击败世界排名前100的球员。 第二轮，白卓璇的对手是本届赛事6号种子，去年温网亚军、突尼斯名将贾巴尔。 王曦雨在2023年温网比赛中 另一场中国球员参加的女单首轮比赛中，世界排名第65位的中国球员王曦雨对阵通过资格赛晋级的世界排名第102位的安德列娃。
至此，白卓璇结束温网4连胜，首次征战大满贯就能通过资格赛考验，并且拿到正赛首胜，中国00后已经创造历史。至此，中国金花结束本届温网单打之旅，接下来的女双和混双，期待姑娘们的出色发挥。
其三，万卓索娃夺得生涯单打第二冠，上一冠还是2017年夺得WTA250比尔赛冠军，且万卓索娃终结过往生涯女单决赛4连败。 其四，万卓索娃以世界排名第42名征战温网，夺冠后将升至世界第10名，成为她生涯首次跻身世界前十，创造生涯排名新高纪录。 其五，万卓索娃成为2021年法网夺冠的克雷吉茨科娃后，最近两年首位夺得大满贯女单冠军的捷克球员。 其六，万卓索娃成为2011年与2014年两次夺得温网冠军的科维托娃之后，最近9年首位夺得温网冠军的捷克球员。
2023年温布尔登网球锦标赛女子单打比赛，是2023年温布尔登网球锦标赛的其中一个比赛项目。伊莲娜·莱巴金娜是卫冕冠军，[1]但她在1/4决赛中输给本届亚军昂丝·加博。  马尔凯塔·万卓索娃是本届比赛的冠军，也是公开赛年代以来首位打入温网女单决赛的非种子选手，也是她继2019年法网后再次进入大满贯决赛，世界排名第42位的她也是自2018年温布顿网球锦标赛的塞雷娜·威廉姆斯（第181位）以来打入温网女单决赛排名最低的选手。
其七，万卓索娃温网夺冠后，温网连续6年迎来新的女单冠军。 其八，万卓索娃连克五位种子，成为公开赛年代第一位夺得温网女单冠军的非种子选手。  6号种子贾巴尔是上届温网亚军，她今年在晋级决赛之旅，她有3场比赛是打满三盘胜出，其中1/4决赛三盘淘汰3号种子莱巴金娜，半决赛三盘淘汰2号种子萨巴伦卡。非种子选手万卓索娃晋级决赛之旅，她有两场比赛打满三盘胜出，其中半决赛两盘横扫斯维托丽娜，后者此前淘汰世界第一的斯瓦泰克。 

---
我的问题或指令：
2023年温网女单亚军？ 介绍下万卓索娃的战绩
---
请依照所给历史对话和参考信息回答问题或执行指令，筛选出有关内容，忽略不相关信息。若信息不足，回答“对不起，我不知道”，切勿随意编造。
你的回复（用中文）："""
    conv.append_message(conv.roles[0], "Tell me a story with more than 1000 words")
    # conv.append_message(conv.roles[0], prompt)
    prompt_template = conv.get_prompt()
    prompts = [prompt_template for _ in range(args.n_thread)]

    print(f"[debug] prompts = {prompts} \n---------\n")

    headers = {"User-Agent": "fastchat Client"}
    ploads = [
        {
            "model": args.model_name,
            "prompt": prompts[i],
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.5,
            # "stop": conv.sep,
        }
        for i in range(len(prompts))
    ]

    def send_request(results, i):
        if args.test_dispatch:
            ret = requests.post(
                controller_addr + "/get_worker_address", json={"model": args.model_name}
            )
            thread_worker_addr = ret.json()["address"]
        else:
            thread_worker_addr = worker_addr
        print(f"thread {i} goes to {thread_worker_addr}")
        response = requests.post(
            thread_worker_addr + "/worker_generate_stream",
            headers=headers,
            json=ploads[i],
            stream=False,
        )
        k = list(
            response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0")
        )
        print(k)

        print(f"==============")
        
        response_new_words = json.loads(k[-2].decode("utf-8"))["text"]
        error_code = json.loads(k[-2].decode("utf-8"))["error_code"]
        print(f"[debug] response_new_words = {response_new_words}")
        # print(f"=== Thread {i} ===, words: {1}, error code: {error_code}")
        results[i] = len(response_new_words.split(" ")) - len(prompts[i].split(" "))

    # use N threads to prompt the backend
    tik = time.time()
    threads = []
    results = [None] * args.n_thread
    for i in range(args.n_thread):
        t = threading.Thread(target=send_request, args=(results, i))
        t.start()
        # time.sleep(0.5)
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[debug] results = {results}")
    print(f"Time (POST): {time.time() - tik} s")
    
    # n_words = 0
    # for i, response in enumerate(results):
    #     # print(prompt[i].replace(conv.sep, "\n"), end="")
    #     # make sure the streaming finishes at EOS or stopping criteria
    #     k = list(response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"))
    #     response_new_words = json.loads(k[-2].decode("utf-8"))["text"]
    #     # print(response_new_words)
    #     n_words += len(response_new_words.split(" ")) - len(prompts[i].split(" "))
    n_words = sum(results)
    time_seconds = time.time() - tik
    print(
        f"Time (Completion): {time_seconds}, n threads: {args.n_thread}, "
        f"throughput: {n_words / time_seconds} words/s."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default="Qwen-7B-Chat-8K")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--n-thread", type=int, default=1)
    parser.add_argument("--test-dispatch", action="store_true")
    args = parser.parse_args()

    main()
