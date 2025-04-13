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
# --- Import the RAG/SQL application ---
try:
    from graph_logic.graph_flow import app as rag_sql_app
    from graph_logic.graph_define import MAX_RETRIES # Import MAX_RETRIES for checking failure
    RAG_APP_AVAILABLE = True
    print("Successfully imported RAG/SQL app (rag_sql_app).")
except ImportError as e:
    print(f"Warning: Could not import RAG/SQL app from graph_logic.graph_flow: {e}. Retrieval will be simulated.")
    rag_sql_app = None
    MAX_RETRIES = 1 # Define a default if import fails
    RAG_APP_AVAILABLE = False
# --- End RAG/SQL import ---

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
        chat_history: The history of the conversation prior to the current message.
    """
    messages: Annotated[List[BaseMessage], lambda x, y: x + y] # Use BaseMessage and restore Annotated
    chat_history: List[BaseMessage] | None # Add chat_history field
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
                """You are an expert router. Your task is to classify the user's question and decide the next step based on the following rules.
Respond ONLY with the specified JSON structure: {{"datasource": "decision"}} where 'decision' must be one of 'answer', 'retrieve', or 'human'.

Routing Rules:

1.  **Route to 'answer' if:**
    *   The question is general (e.g., greetings).
    *   You believe you can answer directly without needing specific external information.
    *   The question involves creative tasks (e.g., writing stories, poems, jokes).

2.  **Route to 'retrieve' if:**
    *   The question asks for specific internal information, such as standards, policies, or procedures.
    *   The question involves specific identifiers like:
        *   Phone numbers (DIDs, extensions, service providers, telco names).
        *   Store names, statuses, retail store identifiers (e.g., R001, r002, store IDs).
        *   Internal phone extensions/numbers (e.g., 7-digit 8001XXX, 601, 621) or rules about them (mandatory, optional, sharing).
    *   The user mentions RXXX store numbers or asks about store-specific details.

3.  **Route to 'human' if:**
    *   The question does not fall into the 'answer' or 'retrieve' categories.

Example User Question Input:
User Question:
{{user_question}}

Your JSON Response:
"""
            ),
            ("human", "User Question:\\n{user_question}"),
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

# --- Retrieve Node (Updated to call RAG/SQL app) ---
def retrieve_node(state: GraphState):
    """
    Processes the user query using the RAG/SQL application, passing chat history.

    Args:
        state: The current graph state, expected to contain 'messages' and 'chat_history'.

    Returns:
        A dictionary with the final result from the RAG/SQL app ('result').
    """
    print("--- Retrieve Node ---")
    if not RAG_APP_AVAILABLE:
        print("Warning: RAG/SQL app not available. Returning simulated retrieval.")
        return {"result": "Simulated retrieval: Could not find specific information."}

    # Ensure messages list is not empty and the last message is HumanMessage
    if not state.get("messages") or not isinstance(state["messages"][-1], HumanMessage):
        print("Warning: Last message in retrieve_node is not a valid HumanMessage.")
        return {"result": "Error: Invalid message state for retrieval."}

    last_message = state["messages"][-1]
    user_query = last_message.content
    print(f"Retrieve Node processing query: '{user_query}'")

    # Prepare input for the RAG/SQL graph
    # The RAG/SQL graph expects 'question' and 'chat_history'
    rag_input = {"question": user_query}

    # --- Pass chat_history if available in the state ---
    current_chat_history = state.get("chat_history")
    if current_chat_history:
        # Ensure chat_history is in the expected format (list of BaseMessage)
        # Convert if necessary, though it should be correct from slack_app
        rag_input["chat_history"] = current_chat_history
        print(f"Retrieve Node: Passing {len(current_chat_history)} messages from chat_history to RAG/SQL app.")
    else:
        print("Retrieve Node: No chat_history found in state.")
        rag_input["chat_history"] = [] # Pass empty list if none

    try:
        print(f"Invoking RAG/SQL app with input keys: {list(rag_input.keys())}")
        # The RAG/SQL app ('rag_sql_app') should return a state dictionary
        # We expect the final answer to be in a key like 'generation' or 'answer'
        # Let's assume the RAG app's final state has a 'generation' key
        rag_result_state = rag_sql_app.invoke(rag_input, {"recursion_limit": 10}) # Increase recursion limit slightly

        print(f"RAG/SQL app returned state: {rag_result_state}") # Log the full returned state for debugging

        # Check for failure condition based on retry counts in the sub-graph
        final_generation = None
        if isinstance(rag_result_state, dict):
            # Check retries first (assuming 'retries' key exists in rag_sql_app state)
            if rag_result_state.get('retries', 0) >= MAX_RETRIES:
                print(f"Warning: RAG/SQL app hit max retries ({MAX_RETRIES}).")
                # Provide a user-friendly message indicating failure after retries
                final_generation = "I tried searching, but encountered some issues retrieving the specific information. Could you try rephrasing your question?"
            else:
                # Try to extract the final generation, might be nested
                # Prioritize 'result', then 'generation', then 'answer'
                if 'result' in rag_result_state and rag_result_state['result']:
                    final_generation = rag_result_state['result']
                elif 'generation' in rag_result_state and rag_result_state['generation']:
                    final_generation = rag_result_state['generation']
                elif 'answer' in rag_result_state and rag_result_state['answer']: # Alternative key
                    final_generation = rag_result_state['answer']

                # If still None, check the messages list in the sub-graph state
                elif 'messages' in rag_result_state and isinstance(rag_result_state['messages'], list) and rag_result_state['messages']:
                    last_rag_message = rag_result_state['messages'][-1]
                    # Ensure it's an AIMessage and not the input HumanMessage
                    if isinstance(last_rag_message, AIMessage):
                        final_generation = last_rag_message.content
                    elif len(rag_result_state['messages']) > 1 and isinstance(rag_result_state['messages'][-2], AIMessage):
                         # Sometimes the state might include the human input as the last message
                         final_generation = rag_result_state['messages'][-2].content


            if final_generation is None:
                 print("Warning: Could not extract final generation from RAG/SQL app state keys (result, generation, answer, messages).")
                 final_generation = "I wasn't able to find a specific answer for that."


        else:
            # If rag_result_state is not a dict (e.g., just a string), use it directly
            print(f"Warning: RAG/SQL app returned a non-dictionary state: {type(rag_result_state)}. Using it directly.")
            final_generation = str(rag_result_state) if rag_result_state else "Received an unexpected empty response."


    except Exception as e:
        print(f"Error invoking RAG/SQL app: {e}", exc_info=True) # Add exc_info for traceback
        # Provide a generic error message if the RAG app fails unexpectedly
        final_generation = "Sorry, I encountered an error while trying to retrieve the information."

    print(f"Retrieve Node Result: {final_generation}")
    # Update the main graph's state with the final result
    return {"result": final_generation} # Store result in the main graph's state

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