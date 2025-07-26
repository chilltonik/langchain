from typing import List, TypedDict, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = OllamaLLM(model="mistral:7b", temperature=0)


def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""

    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response))
    print(f"\nAI: {response}")

    print("CURRENT STATE: ", state["messages"])

    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]

    user_input = input("Enter: ")


with open("db/logging.txt", "w") as f:
    f.write("Your conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("End of conversation\n")
print("Conversion Complete")
