from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from agents.index_tools.index_match import IndexMatchTool
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    execution_count: int = 0  # 添加执行次数计数


# 配置搜索工具，增加超时时间
web_search = DuckDuckGoSearchResults(
    name="WebSearch",
    timeout=30,  # 增加超时时间到30秒
)
index_match = IndexMatchTool()
tools = [web_search, calculator, index_match]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

MAX_EXECUTION_COUNT = 1  # 设置最大执行次数

current_date = datetime.now().strftime("%B %d, %Y")
# 更新 instructions 字符串，添加新工具的说明
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    Available tools and when to use them:
    1. Weather - Use this tool for any weather-related queries, including future weather forecasts. Format: city name or location.
    2. WebSearch - Use this for general web searches when other specific tools don't apply.
    3. Calculator - Use this for mathematical calculations.
    4. IndexMatch - Use this tool to query index matching results for stock codes. Input format: ['600519.SH']

    Tool Selection Guidelines:
    - For weather queries, ALWAYS use the Weather tool first, not WebSearch
    - Only use WebSearch if the Weather tool doesn't provide the needed information
    - For future weather queries, explain that accurate predictions are not possible, but you can provide historical weather patterns using WebSearch
    - For stock index matching queries, use the IndexMatch tool with the correct stock code format

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    try:
        # 增加执行次数计数
        state["execution_count"] = state.get("execution_count", 0) + 1
        
        # 检查是否超过最大执行次数
        if state["execution_count"] > MAX_EXECUTION_COUNT:
            return {
                "messages": [
                    AIMessage(
                        content="已达到最大执行次数限制，请重新开始对话或简化您的查询。"
                    )
                ]
            }

        m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
        model_runnable = wrap_model(m)
        response = await model_runnable.ainvoke(state, config)

        # Run llama guard check here to avoid returning the message if it's unsafe
        llama_guard = LlamaGuard()
        safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
        if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
            return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

        if state["remaining_steps"] < 2 and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        return {"messages": [response]}
    except Exception as e:
        error_message = f"抱歉，搜索服务暂时不可用，请稍后再试。错误信息：{str(e)}"
        return {"messages": [AIMessage(content=error_message)]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# 修改工具调用条件
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    
    # 检查是否超过最大执行次数
    if state.get("execution_count", 0) >= MAX_EXECUTION_COUNT:
        return "done"
        
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

research_assistant = agent.compile(checkpointer=MemorySaver())
