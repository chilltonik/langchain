import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(dotenv_path="../.env")


def get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)

    splitDocs = splitter.split_documents(docs)
    return splitDocs


def create_db(docs):
    embedding_model = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_NAME")
    )
    vector_store = FAISS.from_documents(docs, embedding=embedding_model)
    return vector_store


def create_chain(vector_store):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("LLM_MODEL_NAME"),
        temperature=float(os.getenv("LLM_TEMPERATURE")),
        max_tokens=os.getenv("LLM_MAX_TOKENS"),
        verbose=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's questions based on the context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # chain = prompt | llm
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=retriever_prompt
    )

    retriever_chain = create_retrieval_chain(
        # retriever,
        history_aware_retriever,
        chain,
    )

    return retriever_chain


def process_chat(chain, question, chat_history):
    response = chain.invoke({"input": question, "chat_history": chat_history})

    return response["answer"].strip()


if __name__ == "__main__":
    docs = get_document_from_web("https://python.langchain.com/docs/concepts/lcel/")
    vector_store = create_db(docs)
    chain = create_chain(vector_store)

    chat_history = []

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        response = process_chat(chain, question, chat_history)

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))

        print("Assistant: ", response)
