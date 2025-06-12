import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv(dotenv_path=".env")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=float(os.getenv("LLM_TEMPERATURE")),
    max_tokens=os.getenv("LLM_MAX_TOKENS"),
    verbose=True,
)

# Prompt template
prompt = ChatPromptTemplate.from_template("Tell me a joke about {subject}")

# Create LLM Chain
chain = prompt | llm

response = chain.invoke({"subject": "dog"})
print(response)


# Template from messages
prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": "You are a helpful translator. Translate the user sentence to French.",
        },
        {"role": "user", "content": "{input}"},
    ]
)

chain = prompt | llm

response = chain.invoke({"input": "I love programming"})
print(response.content)
