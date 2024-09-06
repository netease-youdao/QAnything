# 测试样例及打分

## 1. 用法

```bash
git clone https://github.com/NascentCore/QAnything.git
cd QAnything/examples
pip install pandas requests openpyxl openai
python doc_chat.py | tee answers.txt
```

## 2. 效果
```bash
问题: 前后座椅H点参考距离是多少？
参考答案:
    L50=926mm
来源:
    《前后排座椅H点距离要求》
知识库答案:
    如图所示为前后座椅H点距离参考：L50 = 926mm，L51 = 989mm，L51 = 989mm，L51 = 989mm，L50 = 926mm
知识库来源:
    6. 前后排座椅H点距离要求 P1.pptx    68.60%
    4. 前后排座椅间距定义 P1.pptx    62.01%
==================================================
问题: 椅背板与旁侧版的保留运动间隙是多少？
参考答案:
    推荐值≥5mm
来源:
    《back panel 与环境匹配工程校核》
知识库答案:
    椅背板与旁侧版的保留运动间隙≥5mm
知识库来源:
    5. back panel 与环境匹配工程校核_P2.pptx    77.34%
    1. 座椅STO间隙 P1.pptx    55.91%
==================================================
```

## 3. 打分
```bash
export DOUBAO_API_KEY={your_api_key}
export DOUBAO_API_URL={your_api_url}
python scoring.py
```
