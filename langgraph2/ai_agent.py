from typing import List, TypedDict

from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = OllamaLLM(model="mistral:7b", temperature=0)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
