import operator
import os
from typing import Annotated, TypedDict, Union

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_ollama import OllamaLLM

from langgraph.graph import END, StateGraph

load_dotenv(dotenv_path=".env")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


llm = OllamaLLM(model="mistral:7b", temperature=0.0, verbose=True, streaming=True)
tools_list = [TavilySearchResults(max_results=1)]
tools_map = {tool.name: tool for tool in tools_list}


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


def run_agent(state: AgentState):
    invoke_args = {
        "input": state["input"],
        **({"chat_history": state["chat_history"]} if state["chat_history"] else {}),
    }
    outcome = agent.invoke(invoke_args)
    new_hist = state["chat_history"] + [outcome]
    return {
        "agent_outcome": outcome,
        "chat_history": new_hist,
    }


def execute_tools(state: AgentState):
    action = state["agent_outcome"]
    if isinstance(action, AgentAction) and action.tool in tools_map:
        tool = tools_map[action.tool]
        result = tool.run(action.tool_input)
        tool_msg = ToolMessage(content=str(result), name=action.tool)
        new_hist = state["chat_history"] + [tool_msg]
        step = (action, str(result))
        return {
            "chat_history": new_hist,
            "intermediate_steps": state.get("intermediate_steps", []) + [step],
        }
    else:
        return {
            "chat_history": state["chat_history"],
            "intermediate_steps": state.get("intermediate_steps", []),
        }


def should_continue(state: AgentState):
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    return "continue"


template = """You are a smart AI agent. You can use tools to answer questions.

Format when using a tool:
Thought: reason about what to do
Action: {tool}
Action Input: "search query"

If you know the answer directly:
Final Answer: your answer

Question: {input}
"""
prompt = PromptTemplate.from_template(template)

agent = initialize_agent(
    tools=tools_list,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prompt=prompt,
    verbose=True,
)

workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)
workflow.add_edge("action", "agent")

app = workflow.compile()

inputs = {"input": "what is the weather in San Francisco", "chat_history": []}
print("Agent is thinking...\n")
for step in app.stream(inputs):
    print(list(step.values())[0])
    print("-----")
