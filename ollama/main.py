from langchain_ollama.chat_models import ChatOllama

model_name = "qwen3:8b"

# initialize one LLM with temperature 0.0, this makes the LLM more deterministic
llm = ChatOllama(temperature=0.0, model=model_name)
print(llm.invoke("Tell me a short story about cat's adventure").content)