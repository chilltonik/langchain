import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=".env")


# Create Retriever
loader = WebBaseLoader("https://python.langchain.com/docs/expression_language/")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splitDocs = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL_NAME"))
vector_store = FAISS.from_documents(docs, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=float(os.getenv("LLM_TEMPERATURE")),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS")),
    verbose=True,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly assistant called Max."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

search = TavilySearchResults()
retriever_tools = create_retriever_tool(
    retriever,
    "lcel_search",
    "Use this tool when searching for information about Langchain Expression Language (LCEL).",
)
tools = [search, retriever_tools]

agent = initialize_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agentExecutor = AgentExecutor(agent=agent, tools=tools)


def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({"input": user_input, "chat_history": chat_history})
    return response["output"]


if __name__ == "__main__":
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = process_chat(agentExecutor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant:", response)
