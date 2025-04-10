import os
import json
# import requests # Remove direct requests import
from typing import TypedDict, Annotated, List, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage # Import BaseMessage & AIMessage
# from langchain_core.messages import message_to_dict # No longer needed if using BaseMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama # Re-add Ollama import
from langchain_core.prompts import ChatPromptTemplate # Import prompt template
from langchain.chains import LLMChain # Correct import for LLMChain if needed, but we use LCEL

# Load environment variables from .env file
load_dotenv()

# Get Ollama config from environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip('/')
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.3")
try:
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
except ValueError:
    OLLAMA_TEMPERATURE = 0.0

# Removed OLLAMA_CHAT_ENDPOINT as it's handled by ChatOllama

# 1. Define the State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: The list of messages exchanged so far (using BaseMessage).
        next_node: The next node to route to ('answer', 'retrieve', 'human').
        result: The final result from a node.
    """
    messages: Annotated[List[BaseMessage], lambda x, y: x + y] # Use BaseMessage and restore Annotated
    next_node: str | None
    result: str | None

# --- Pydantic Model for Router Decision (Moved Before Usage) ---
class RouterDecision(BaseModel):
    """Pydantic model for the router's structured decision."""
    datasource: Literal['answer', 'retrieve', 'human'] = Field(
        description="The next node to route to. Must be one of 'answer', 'retrieve', or 'human'."
    )

# Removed format_messages_for_ollama as ChatOllama handles formatting

# --- Define Router Logic Separately ---
def get_router_chain():
    """Creates and returns the LangChain runnable for the router decision."""
    # Define the prompt for the router LLM
    router_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert router.
                Based on the user's question below, decide whether the question can be answered directly by an AI assistant, requires retrieving information from a database, or needs human intervention.
                Respond only with the JSON structure specified.
                for general questions, greetings, the questins you think you can answer without any additional information, respond with 'answer'
                for questions, creative tasks like writing stories/poems/jokes, respond with 'answer'
                for questions about specific information, such as standards, policies, procedures respond with 'retrieve'
                for other questions, respond with 'human'

                """,
            ),
            ("human", "User Question:\n{user_question}"),
        ]
    )

    # Initialize ChatOllama with structured output
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1, format="json")
    structured_llm = llm.with_structured_output(RouterDecision)

    # Create the chain (LangChain Expression Language)
    router_chain = router_prompt_template | structured_llm
    return router_chain

# Instantiate the router chain once
router_chain = get_router_chain()

# 2. Define Nodes

# --- Router Node (using ChatOllama with structured output) ---
def router_node(state: GraphState):
    """
    Analyzes the latest user message using ChatOllama and decides the next step.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the decision ('next_node').
    """
    print("--- Router Node (LangChain) ---")
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        # Handle cases where the last message isn't from the user (shouldn't happen in this flow)
        print("Warning: Last message is not HumanMessage, defaulting route to 'answer'")
        return {"next_node": "answer"}

    user_question_content = last_message.content

    # --- Use the pre-defined router_chain --- 
    # Define the prompt for the router LLM
    # router_prompt_template = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             """You are an expert router. 
    #             Based on the user's question below, decide whether the question can be answered directly by an AI assistant, requires retrieving information from a database, or needs human intervention. 
    #             Respond only with the JSON structure specified.
    #             for general questions, greetings, the questins you think you can answer without any additional information, respond with 'answer'
    #             for questions, creative tasks like writing stories/poems/jokes, respond with 'answer'
    #             for questions about specific information, such as standards, policies, procedures respond with 'retrieve'
    #             for other questions, respond with 'human'
    #             
    #             """,
    #         ),
    #         ("human", "User Question:\n{user_question}"),
    #     ]
    # )

    # Initialize ChatOllama with structured output
    # Use a slightly higher temperature for routing flexibility
    # llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1, format="json")
    # structured_llm = llm.with_structured_output(RouterDecision)

    # Create the chain
    # router_chain = router_prompt_template | structured_llm

    print(f"Router executing chain for question: '{user_question_content}'")
    decision = "answer" # Default in case of error

    try:
        # Invoke the chain
        router_output: RouterDecision = router_chain.invoke({"user_question": user_question_content})
        decision = router_output.datasource.lower()
        print(f"Router LLM Decision: '{decision}'")
        if decision not in ['answer', 'retrieve', 'human']:
             print(f"Warning: Router LLM returned unexpected decision value '{decision}'. Defaulting to 'answer'.")
             decision = "answer"

    except Exception as e:
        print(f"An unexpected error occurred in router_node (LangChain): {type(e).__name__}: {e}")
        decision = "answer" # Fallback

    print(f"Decision: Route to '{decision}'")
    return {"next_node": decision}


# --- Answer Node (using direct Ollama API call - keep as is for now) ---
def answer_node(state: GraphState):
    """
    Generates a direct answer using a direct Ollama API call.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the AI's answer ('result').
    """
    print("--- Answer Node ---")
    # Format messages for Ollama API
    # Include System Prompt + User Question for context
    # Need to convert BaseMessage back to dict for manual API call format
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        print("Warning: Last message in answer_node is not HumanMessage.")
        return {"result": "Error: Invalid message state."}

    ollama_messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question directly."},
        {"role": "user", "content": last_message.content} # Directly use content
    ]

    print(f"Answer Messages for Ollama API: {ollama_messages}")

    # Prepare payload for Ollama API
    payload = {
        "model": OLLAMA_MODEL,
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE
        }
    }

    result = "Sorry, I couldn't generate an answer due to an error." # Default result

    try:
        # Need requests import back if answer_node keeps using it
        import requests
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes

        response_data = response.json()
        print(f"Ollama Answer Raw Response: {response_data}")

        # Extract the message content
        result = response_data.get("message", {}).get("content", result)

    except requests.exceptions.RequestException as req_e:
        print(f"Error calling Ollama API in answer_node: {req_e}")
    except Exception as e:
        print(f"An unexpected error occurred in answer_node: {e}")

    print(f"Answer Response: {result}")
    return {"result": result}

# --- Retrieve Node (Simulation - needs update for BaseMessage) ---
def retrieve_node(state: GraphState):
    """
    Simulates retrieving information from a source.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the simulated retrieval result ('result').
    """
    print("--- Retrieve Node (Simulation) ---")
    # Get content from the last message (should be BaseMessage)
    last_message = state["messages"][-1] if state["messages"] else None
    user_question = last_message.content if isinstance(last_message, HumanMessage) else "Unknown query"

    # In a real app, you'd query a vector DB or other source here.
    simulated_data = f"Simulated data related to '{user_question[:30]}...': Found relevant policy document XYZ."
    print(f"Retrieval Result: {simulated_data}")
    # Here you might feed the retrieved data + question back to an LLM for synthesis
    # For simplicity, we just return the simulated data directly.
    return {"result": simulated_data}

# --- Human Node (Simulation - no change needed) ---
def human_node(state: GraphState):
    """
    Simulates handing off to a human.

    Args:
        state: The current graph state.

    Returns:
        A dictionary indicating human intervention ('result').
    """
    print("--- Human Node (Simulation) ---")
    result = "The user's query requires human attention. Please assign to an agent."
    print(f"Human Handoff: {result}")
    return {"result": result}


# 3. Define Conditional Edges
def decide_next_node(state: GraphState) -> str:
    """
    Determines the next node based on the router's decision.

    Args:
        state: The current graph state.

    Returns:
        The name of the next node to execute.
    """
    print("--- Deciding Next Node ---")
    decision = state.get("next_node")
    print(f"Routing based on decision: {decision}")
    if decision == "answer":
        return "answer_node"
    elif decision == "retrieve":
        return "retrieve_node"
    elif decision == "human":
        return "human_node"
    else:
        print("Warning: No valid decision found, ending graph.")
        return END # Should not happen if router works correctly


# 4. Build the Graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("answer_node", answer_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("human_node", human_node)

# Set the entrypoint
workflow.set_entry_point("router")

# Add conditional edges
workflow.add_conditional_edges(
    "router",          # Start node
    decide_next_node,  # Function to decide the next path
    {                  # Mapping decisions to target nodes
        "answer_node": "answer_node",
        "retrieve_node": "retrieve_node",
        "human_node": "human_node",
    }
)

# Add edges from worker nodes to the end
workflow.add_edge("answer_node", END)
workflow.add_edge("retrieve_node", END)
workflow.add_edge("human_node", END)

# Compile the graph
app = workflow.compile()

# 5. Run the Graph (Interactive Mode - needs update for BaseMessage)
# Remove the entire 'if __name__ == "__main__":' block below 