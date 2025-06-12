import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv(dotenv_path=".env")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=float(os.getenv("LLM_TEMPERATURE")),
    max_tokens=os.getenv("LLM_MAX_TOKENS"),
    verbose=True,
)

"""
messages = [
    SystemMessage(content="You are a helpful translator. Translate the user sentence to French."),
    HumanMessage(content="Where is the nearest train station?")
]
"""
# or
messages = [
    {
        "role": "system",
        "content": "You are a helpful translator. Translate the user sentence to French.",
    },
    {"role": "user", "content": "Where is the nearest train station?"},
]

# response = llm.invoke(input=messages)
# print(response.content)
#
# response = llm.batch(["Hello, how are you?", "Tell me about quantum physics"])
# print(response)

response = llm.stream("Write a poem about AI")
for chunk in response:
    print(chunk.content, end="", flush=True)
