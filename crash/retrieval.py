import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(dotenv_path="../.env")


def get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    splitDocs = splitter.split_documents(docs)
    # print(len(splitDocs))
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

    prompt = ChatPromptTemplate.from_template(
        """
    Answer the user question:
    Context: {context}
    Question: {input}
    """
    )

    # chain = prompt | llm
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retriever_chain = create_retrieval_chain(
        retriever,
        chain,
    )

    return retriever_chain


docs = get_document_from_web("https://python.langchain.com/docs/concepts/lcel/")
vector_store = create_db(docs)


chain = create_chain(vector_store)
response = chain.invoke(
    {
        "input": "What is the LCEL?",
    }
)

print(response["answer"])
