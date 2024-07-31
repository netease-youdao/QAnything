from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI


class RewriteQuestionChain:
    def __init__(self, model_name, openai_api_key, openai_api_base):
        self.chat_model = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key, openai_api_base=openai_api_base,
                                     temperature=0, model_kwargs={"top_p": 0.01, "seed": 1234})
        self.condense_q_system_prompt = """
假设你是极其专业的英语和汉语语言专家。你的任务是：给定一个聊天历史记录和一个可能涉及此聊天历史的用户最新的问题(新问题)，请构造一个不需要聊天历史就能理解的独立且语义完整的问题。

你可以假设这个问题是在用户与聊天机器人对话的背景下。

instructions:
- 请始终记住，你的任务是生成独立问题，而不是直接回答新问题！
- 根据用户的新问题和聊天历史记录，判断新问题是否已经是独立且语义完整的。如果新问题已经独立且完整，直接输出新问题，无需任何改动；否则，你需要对新问题进行改写，使其成为独立问题。
- 确保问题在重新构造前后语种保持一致。
- 确保问题在重新构造前后意思保持一致。
- 在构建独立问题时，尽可能将代词（如"她"、"他们"、"它"等）替换为聊天历史记录中对应的具体的名词或实体引用，以提高问题的明确性和易理解性。

```
Example input:
HumanMessage: `北京明天出门需要带伞吗？`
AIMessage: `今天北京的天气是全天阴，气温19摄氏度到27摄氏度，因此不需要带伞噢。`
新问题: `那后天呢？`  # 问题与上文有关，不独立且语义不完整，需要改写
Example output: `北京后天出门需要带伞吗？`  # 根据聊天历史改写新问题，使其独立

Example input:
HumanMessage: `明天北京的天气是多云转晴，适合出门野炊吗？`
AIMessage: `当然可以，这样的天气非常适合出门野炊呢！不过在出门前最好还是要做好防晒措施噢~`
新问题: `那北京哪里适合野炊呢？`  # 问题已经是独立且语义完整的，不需要改写
Example output: `那北京哪里适合野炊呢？` # 直接返回新问题，不需要改写
```

"""
        self.condense_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.condense_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "新问题:{question}\n请构造不需要聊天历史就能理解的独立且语义完整的问题。\n独立问题:"),
            ]
        )

        self.condense_q_chain = self.condense_q_prompt | self.chat_model | StrOutputParser()

