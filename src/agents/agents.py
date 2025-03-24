from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.command_agent import command_agent
from agents.interrupt_agent import interrupt_agent
from agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from agents.research_assistant import research_assistant
from schema import AgentInfo

DEFAULT_AGENT = "research-assistant"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot.", graph=chatbot),
    "research-assistant": Agent(
        description="A research assistant with web search and calculator.", graph=research_assistant
    ),
    "command-agent": Agent(description="A command agent.", graph=command_agent),
    "bg-task-agent": Agent(description="A background task agent.", graph=bg_task_agent),
    "langgraph-supervisor-agent": Agent(
        description="A langgraph supervisor agent", graph=langgraph_supervisor_agent
    ),
    "interrupt-agent": Agent(description="An agent the uses interrupts.", graph=interrupt_agent),
}


import logging

logger = logging.getLogger(__name__)

def get_agent(agent_id: str) -> CompiledStateGraph:
    try:
        save_graph_visualization(agents[agent_id].graph, "png/"+agent_id + ".png")
    except Exception as e:
        logger.warning(f"Failed to save graph visualization for {agent_id}: {str(e)}")
    return agents[agent_id].graph

def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    try:
        # 尝试获取图的可视化表示
        if hasattr(graph, "get_graph"):
            with open(filename, "wb") as f:
                f.write(graph.get_graph().draw_mermaid_png())
            logger.info(f"Graph visualization saved as {filename}")
        else:
            logger.warning(f"Graph visualization not supported for this type of graph")
    except Exception as e:
        logger.warning(f"Failed to save graph visualization: {str(e)}")


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
