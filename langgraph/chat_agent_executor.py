from typing import TypedDict, Union

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.agents import AgentAction, AgentFinish
from langchain_ollama import OllamaLLM

from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv(".env")

# Initialize Ollama LLM with streaming enabled
llm = OllamaLLM(model="deepseek-r1:8b", streaming=True, temperature=0.0)

# Prepare tool registry
tools = [TavilySearchResults(max_results=1)]
tools_map = {tool.name: tool for tool in tools}
# Use dynamic tool name to ensure consistency
tool_name = tools[0].name


# Define the shared agent state
class AgentState(TypedDict):
    input: str
    messages: list[str]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: list[str]


# Node: Invoke agent (LLM or tool decision)
def run_agent(state: AgentState):
    prompt = state.get("input", "")
    messages = state.get("messages", [])
    intermediates = state.get("intermediate_steps", [])

    if prompt.startswith("search:"):
        query = prompt.split("search:", 1)[1].strip()
        # Plan the tool action using dynamic tool_name
        action = AgentAction(
            tool=tool_name, tool_input=query, log=f"Search query: {query}"
        )
        # Add interim log
        intermediates.append(f"Planned action: {tool_name} with input '{query}'")
        return {
            "agent_outcome": action,
            "messages": messages,
            "intermediate_steps": intermediates,
        }

    # Otherwise, call the LLM and stream the result
    response = ""
    for chunk in llm.complete(prompt, stream=True):
        print(chunk, end="")
        response += chunk
    # Log the LLM response in messages
    messages.append(f"LLM: {response}")
    intermediates.append("LLM response complete")

    finish = AgentFinish(
        return_values={"output": response}, log="LLM response complete"
    )
    return {
        "agent_outcome": finish,
        "messages": messages,
        "intermediate_steps": intermediates,
    }


# Node: Execute tool and finish
def execute_tools(state: AgentState):
    outcome = state.get("agent_outcome")
    messages = state.get("messages", [])
    intermediates = state.get("intermediate_steps", [])

    if isinstance(outcome, AgentAction) and outcome.tool in tools_map:
        result = tools_map[outcome.tool].run(outcome.tool_input)
        message = f"[tool: {outcome.tool}] -> {result}"
        messages.append(message)
        intermediates.append(f"Tool {outcome.tool} executed with result")

        finish = AgentFinish(
            return_values={"output": result}, log=f"Tool {outcome.tool} executed"
        )
        return {
            "agent_outcome": finish,
            "messages": messages,
            "intermediate_steps": intermediates,
        }

    # No action, finish with default
    finish = AgentFinish(
        return_values={"output": "No action performed"}, log="No action performed"
    )
    return {
        "agent_outcome": finish,
        "messages": messages,
        "intermediate_steps": intermediates,
    }


# Build and compile the state graph
graph = StateGraph(AgentState)
graph.set_entry_point("agent")
# Agent -> Action if AgentAction, else end
graph.add_node("agent", run_agent)
graph.add_conditional_edges(
    "agent",
    lambda s: "action" if isinstance(s.get("agent_outcome"), AgentAction) else END,
    {"action": "action", END: END},
)
# Action -> Agent if not finish, else end
graph.add_node("action", execute_tools)
graph.add_conditional_edges(
    "action",
    lambda s: "agent" if not isinstance(s.get("agent_outcome"), AgentFinish) else END,
    {"agent": "agent", END: END},
)

app = graph.compile()

# Run the agent with an example input
initial_state = {
    "input": "search: latest AI trends",
    "messages": [],
    "agent_outcome": None,
    "intermediate_steps": [],
}

print("--- Agent execution started ---")
# Use a mutable state and merge updates
tate = initial_state.copy()
for delta in app.stream(tate):
    for k, v in delta.items():
        tate[k] = v
final_state = tate
print("--- Execution finished ---")

# Display the collected messages and final result
print("Collected messages:", final_state.get("messages"))
outcome = final_state.get("agent_outcome")
if hasattr(outcome, "return_values"):
    print("Final output:", outcome.return_values.get("output"))
