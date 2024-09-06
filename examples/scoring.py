import os
from openai import OpenAI, AsyncOpenAI

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_API_URL = os.getenv("DOUBAO_API_URL")

answers_file = "answers.txt"

client = OpenAI(
    api_key=DOUBAO_API_KEY,
    base_url=DOUBAO_API_URL
)

with open(answers_file) as f:
    answers_txt = f.read()

content = f"{answers_txt}\n请对上面的参考答案和知识库答案进行打分，语义上只要知识库答案涵盖了参考答案则得高分，输出格式如下：\n问题：\n打分：\n打分依据："

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "user", "content": content},
    ],
    model="ep-20240721110948-mdv29",
)

print(chat_completion.choices[0].message.content)
