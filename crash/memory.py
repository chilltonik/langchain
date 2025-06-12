import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import \
    UpstashRedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

load_dotenv(dotenv_path=".env")

history = UpstashRedisChatMessageHistory(
    url=os.getenv("UPSTASH_URL"), token=os.getenv("UPSTASH_TOKEN"), session_id="chat1"
)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=float(os.getenv("LLM_TEMPERATURE")),
    max_tokens=os.getenv("LLM_MAX_TOKENS"),
    verbose=True,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly AI assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, chat_memory=history
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
)
# chain = prompt | llm

# msg1 = {
#     "input": "My name is Toni"
# }
#
# rsp1 = chain.invoke(msg1)
# print(rsp1)

msg2 = {"input": "What is my name?"}

rsp2 = chain.invoke(msg2)
print(rsp2)
