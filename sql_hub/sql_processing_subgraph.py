import sys
import os
from typing import TypedDict, Annotated, List, Optional
import operator
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate

# Add graph_logic to path to import tools and llm
# Assuming this subgraph file is in 'sql_hub' which is a sibling of 'graph_logic'
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import tools and llm from graph_logic
# Ensure these are correctly defined and accessible in database_tools.py
try:
    # from graph_logic.database_tools import llm, list_tables_tool, info_sql_tool, query_sql_tool # Old import
    from .database_tools import llm, list_tables_tool, info_sql_tool, query_sql_tool, query_checker_tool # New import from current package
except ImportError as e:
    print(f"Error importing from .database_tools: {e}")
    print("Please ensure sql_hub/database_tools.py exists and defines required tools including query_checker_tool.")
    # Define dummy implementations or raise error if preferred
    raise

# State for the SQL Agent Subgraph
class SqlAgentSubgraphState(TypedDict):
    """
    Represents the state for the SQL Agent subgraph.

    Attributes:
        question: The user's question to be answered using the database.
        max_iterations: Maximum number of LLM query generations allowed.
        iterations: Current iteration count.
        tables: List of available tables.
        schema: Schema information for relevant tables.
        history_query: The last executed SQL query.
        history_result: The result of the last executed SQL query.
        accumulated_results: List of results from all executed queries.
        final_db_results: The final accumulated results list (might be deprecated).
        final_answer: The synthesized final answer string.
        error_message: Optional error message if something goes wrong.
    """
    question: str
    max_iterations: int
    iterations: Annotated[int, operator.add] # Use operator.add to increment
    tables: Optional[str]
    schema: Optional[str]
    history_query: Optional[str]
    history_result: Optional[str]
    accumulated_results: Annotated[List[str], operator.add]
    final_db_results: Optional[List[str]]
    final_answer: Optional[str]
    error_message: Optional[str]

# --- Subgraph Nodes ---

def get_initial_db_info(state: SqlAgentSubgraphState):
    """Fetches the list of tables and the full schema for those tables."""
    print("--- SQL Subgraph: Getting Initial DB Info ---")
    try:
        tables = list_tables_tool.run("")
        schema_info = info_sql_tool.run(tables)
        print(f"--- SQL Subgraph: Available Tables: {tables} ---")
        print("--- SQL Subgraph: Schema info obtained --- ")
        return {
            "tables": tables,
            "schema": schema_info,
            "iterations": 0,
            "history_query": None,
            "history_result": "N/A",
            "accumulated_results": [],
            "final_answer": None,
            "error_message": None
        }
    except Exception as e:
        print(f"--- SQL Subgraph ERROR: Failed to get initial DB info: {e} ---")
        return {"error_message": f"Failed to get initial DB info: {e}", "iterations": 0, "accumulated_results": [], "final_answer": None}

def generate_next_query(state: SqlAgentSubgraphState):
    """
    Asks the LLM to generate the *next single* SQL query needed, based on history.
    Focuses ONLY on query generation, not deciding completion.
    Updates history_query if a new query is generated.
    """
    print(f"--- SQL Subgraph: Generate Query - Iteration {state['iterations'] + 1}/{state['max_iterations']} ---")

    # Check iteration limit before generation
    if state["iterations"] >= state["max_iterations"]:
        print(f"--- SQL Subgraph: Max iterations ({state['max_iterations']}) reached before generation. Skipping. ---")
        # Signal that no query should be executed, leading to answer synthesis attempt
        return {"history_query": None}

    # Simplified prompt focusing only on the next query
    generation_template = """
    You are an expert SQL agent working with a SQLite database. Your task is to generate the single best SQLite SQL query to help answer the user's question, based on the history and accumulated results.

    Follow these steps:
    1.  Analyze the 'User Question' and 'Accumulated Results' to identify the *primary missing information* needed.
    2.  If the question asks for *both* a count and a list, prioritize getting the *complete list* first (e.g., `SELECT item FROM ...`).
    3.  If *all* necessary information seems present in the 'Accumulated Results', respond with "NONE".
    4.  Otherwise, respond ONLY with the *single* best SQLite SQL query to fetch the primary missing information. Do not include explanations or comments.

    Here are some examples:

    --- Example 1: Simple Lookup ---
    User Question: What is the phone number for store R002?
    Available Tables: retail_store, phone_number, providers
    Database Schema: 
    CREATE TABLE retail_store (store_number TEXT PRIMARY KEY, address TEXT)
    CREATE TABLE phone_number (id INTEGER PRIMARY KEY, store_number TEXT, phone_number TEXT, provider_id INTEGER)
    CREATE TABLE providers (provider_id INTEGER PRIMARY KEY, name TEXT)
    --- History ---
    Previous Query: N/A
    Previous Result: N/A
    --- End History ---
    Accumulated Results So Far (for context):
    
    --- End Accumulated ---
    Your Response: SELECT phone_number FROM phone_number WHERE store_number = 'R002'
    --- End Example 1 ---

    --- Example 2: Count and List (Step 1: Get List) ---
    User Question: How many providers are there, and what are their names?
    Available Tables: retail_store, phone_number, providers
    Database Schema: 
    CREATE TABLE retail_store (store_number TEXT PRIMARY KEY, address TEXT)
    CREATE TABLE phone_number (id INTEGER PRIMARY KEY, store_number TEXT, phone_number TEXT, provider_id INTEGER)
    CREATE TABLE providers (provider_id INTEGER PRIMARY KEY, name TEXT)
    --- History ---
    Previous Query: N/A
    Previous Result: N/A
    --- End History ---
    Accumulated Results So Far (for context):
    
    --- End Accumulated ---
    Your Response: SELECT name FROM providers
    --- End Example 2 ---
    
    --- Example 3: Count and List (Step 2: Info Gathered, Done) ---
    User Question: How many providers are there, and what are their names?
    Available Tables: retail_store, phone_number, providers
    Database Schema: 
    CREATE TABLE retail_store (store_number TEXT PRIMARY KEY, address TEXT)
    CREATE TABLE phone_number (id INTEGER PRIMARY KEY, store_number TEXT, phone_number TEXT, provider_id INTEGER)
    CREATE TABLE providers (provider_id INTEGER PRIMARY KEY, name TEXT)
    --- History ---
    Previous Query: SELECT name FROM providers
    Previous Result: [('AT&T',), ('Verizon',), ('T-Mobile',)]
    --- End History ---
    Accumulated Results So Far (for context):
    [('AT&T',), ('Verizon',), ('T-Mobile',)]
    --- End Accumulated ---
    Your Response: NONE
    --- End Example 3 ---
    
    --- Example 4: Multi-step Join (Step 1: Find store number) ---
    User Question: What are the phone numbers at the store located at '123 Main St'?
    Available Tables: retail_store, phone_number, providers
    Database Schema: 
    CREATE TABLE retail_store (store_number TEXT PRIMARY KEY, address TEXT)
    CREATE TABLE phone_number (id INTEGER PRIMARY KEY, store_number TEXT, phone_number TEXT, provider_id INTEGER)
    CREATE TABLE providers (provider_id INTEGER PRIMARY KEY, name TEXT)
    --- History ---
    Previous Query: N/A
    Previous Result: N/A
    --- End History ---
    Accumulated Results So Far (for context):
    
    --- End Accumulated ---
    Your Response: SELECT store_number FROM retail_store WHERE address = '123 Main St'
    --- End Example 4 ---

    --- Example 5: Multi-step Join (Step 2: Use store number to find phone numbers) ---
    User Question: What are the phone numbers at the store located at '123 Main St'?
    Available Tables: retail_store, phone_number, providers
    Database Schema: 
    CREATE TABLE retail_store (store_number TEXT PRIMARY KEY, address TEXT)
    CREATE TABLE phone_number (id INTEGER PRIMARY KEY, store_number TEXT, phone_number TEXT, provider_id INTEGER)
    CREATE TABLE providers (provider_id INTEGER PRIMARY KEY, name TEXT)
    --- History ---
    Previous Query: SELECT store_number FROM retail_store WHERE address = '123 Main St'
    Previous Result: [('R005',)]
    --- End History ---
    Accumulated Results So Far (for context):
    [('R005',)]
    --- End Accumulated ---
    Your Response: SELECT phone_number FROM phone_number WHERE store_number = 'R005'
    --- End Example 5 ---

    --- Now, the actual task ---
    User Question: {question}
    Available Tables: {tables}
    Database Schema: {schema}
    --- History ---
    Previous Query: {last_query}
    Previous Result: {last_result}
    --- End History ---
    Accumulated Results So Far (for context):
    {accumulated}
    --- End Accumulated ---

    Your Response (Provide the next single SQL query or the word "NONE"):
    """
    # Keep the template string
    # generation_prompt = ChatPromptTemplate.from_template(generation_template) # Remove template creation
    # generation_chain = generation_prompt | llm # Remove chain creation

    try:
        # Manually format the prompt
        input_dict = {
            "question": state["question"],
            "schema": state["schema"],
            "tables": state["tables"],
            "last_query": state.get("history_query", "N/A"),
            "last_result": state.get("history_result", "N/A"),
            "accumulated": "\n".join(map(str, state.get("accumulated_results", [])))
        }
        formatted_prompt = generation_template.format(**input_dict)

        # Directly invoke the LLM
        llm_response_obj = llm.invoke(formatted_prompt)
        llm_response = getattr(llm_response_obj, 'content', str(llm_response_obj)).strip()

        if llm_response.upper() == "NONE":
            print("--- SQL Subgraph: LLM indicated no further query needed or possible. ---")
            # Set history_query to None to signal no execution needed
            return {"history_query": None, "iterations": 1}

        # Assume it's a SQL query
        generated_sql = llm_response
        # Basic check (optional but good)
        if not generated_sql.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE")):
            print(f"--- SQL Subgraph WARNING: LLM response ('{generated_sql}') doesn't look like SQL or 'NONE'. Treating as NONE. ---")
            return {"history_query": None, "iterations": 1}

        print(f"--- SQL Subgraph: Generated SQL: {generated_sql} ---")
        # Update history_query for the *next* step (execute_query) and increment iterations
        return {"history_query": generated_sql, "iterations": 1}

    except Exception as e:
        print(f"--- SQL Subgraph ERROR: Failed during LLM generation: {e} ---")
        error_msg = f"Failed during LLM generation: {e}"
        # Store error and signal no query execution
        return {"error_message": error_msg, "history_query": None, "iterations": 1}

def execute_query(state: SqlAgentSubgraphState):
    """
    Executes the SQL query stored in history_query using the direct SQL Tool.
    Updates accumulated_results and history_result.
    """
    sql_to_execute = state.get("history_query")

    if not sql_to_execute:
         print("--- SQL Subgraph: No query to execute. Skipping execution. ---")
         return {"history_result": "N/A"}

    print(f"--- SQL Subgraph: Executing Query directly: {sql_to_execute} ---")
    try:
        # Revert back to using the direct query_sql_tool
        execution_result = query_sql_tool.run(sql_to_execute)
        result_str = str(execution_result)
        print(f"--- SQL Subgraph: Execution Result: {result_str} ---")
        return {"accumulated_results": [result_str], "history_result": result_str, "error_message": None}
    except Exception as e:
        # Use the original error message format
        error_msg = f"Error executing query '{sql_to_execute}': {e}"
        print(f"--- SQL Subgraph ERROR: {error_msg} ---")
        return {"accumulated_results": [error_msg], "history_result": error_msg, "error_message": error_msg}

def synthesize_final_answer(state: SqlAgentSubgraphState):
    """
    Synthesizes a final natural language answer from accumulated results.
    This node is reached when query generation stops (either completed or max iterations).
    """
    print("--- SQL Subgraph: Synthesize Final Answer ---")

    # If error occurred previously, end immediately with the error
    if state.get("error_message"):
         print("--- SQL Subgraph: Ending due to previous error before final synthesis. ---")
         # Populate final_answer and final_db_results with error info
         error_msg = f"Process ended due to error: {state['error_message']}"
         return {"final_answer": error_msg, "final_db_results": [error_msg]}

    # Check if max iterations were reached (for logging purposes mainly)
    if state["iterations"] >= state["max_iterations"]:
        print(f"--- SQL Subgraph: Max iterations ({state['max_iterations']}) reached. Synthesizing best possible answer from results. ---")

    # Simplified prompt: Always synthesize the best answer.
    synthesis_prompt_template = """
    Given the original user question and the results accumulated from database queries, synthesize a concise, final natural language answer.

    Base your answer *only* on the provided information. If the results are empty, state that no information was found.

    Original User Question: {question}

    Accumulated Database Results:
    {accumulated}

    Your Response (The final synthesized answer):
    """
    # Keep the template string
    # synthesis_prompt = ChatPromptTemplate.from_template(synthesis_prompt_template) # Remove template creation
    # synthesis_chain = synthesis_prompt | llm # Remove chain creation

    try:
        accumulated_str = "\n---\n".join(map(str, state.get("accumulated_results", ["No results found."])))
        # Manually format the prompt
        formatted_prompt = synthesis_prompt_template.format(
             question=state["question"],
             accumulated=accumulated_str
        )
        # Directly invoke the LLM
        response_obj = llm.invoke(formatted_prompt)
        response = getattr(response_obj, 'content', str(response_obj)).strip()

        # Always treat the LLM response as the final answer now.
        print(f"--- SQL Subgraph: Final answer synthesized: {response} ---")
        # Return the synthesized answer for final_answer and as the only item in final_db_results
        return {"final_answer": response, "final_db_results": [response]}

    except Exception as e:
        print(f"--- SQL Subgraph ERROR: Failed during final answer synthesis: {e} ---")
        # End with error message
        error_msg = f"Error during final answer synthesis: {e}"
        return {"final_answer": error_msg, "error_message": str(e), "final_db_results": [error_msg]}

# --- Conditional Edges ---

def route_after_generation(state: SqlAgentSubgraphState):
    """Routes execution after the generate_next_query node."""
    if state.get("history_query"):
        print("--- SQL Subgraph: Routing to Execute Query ---")
        return "execute_query"
    else:
        print("--- SQL Subgraph: Routing to Synthesize Answer (No Query Generated) ---")
        return "synthesize_final_answer"

def route_after_execution(state: SqlAgentSubgraphState):
    """Routes execution after the execute_query node."""
    # Always go to synthesis after execution (unless error occurred during execution itself)
    if state.get("error_message"):
         print("--- SQL Subgraph: Routing to Synthesize Answer (Error during Execution) ---")
    else:
         print("--- SQL Subgraph: Routing to Synthesize Answer (After Query Execution) ---")
    return "synthesize_final_answer"

def route_after_synthesis(state: SqlAgentSubgraphState):
    """Routes execution after the synthesize_final_answer node."""
    if state.get("final_answer") is None and not state.get("error_message"):
        print("--- SQL Subgraph: Routing back to Generate Query ---")
        return "generate_query"
    else:
        print("--- SQL Subgraph: Routing to END ---")
        return END

# --- Build the Subgraph ---
sql_agent_subgraph_workflow = StateGraph(SqlAgentSubgraphState)

# Add nodes
sql_agent_subgraph_workflow.add_node("get_initial_db_info", get_initial_db_info)
sql_agent_subgraph_workflow.add_node("generate_query", generate_next_query)
sql_agent_subgraph_workflow.add_node("execute_query", execute_query)
sql_agent_subgraph_workflow.add_node("synthesize_final_answer", synthesize_final_answer)

# Set entry point
sql_agent_subgraph_workflow.set_entry_point("get_initial_db_info")

# Define edges
# If initial info fails, go straight to final synthesis
sql_agent_subgraph_workflow.add_conditional_edges(
    "get_initial_db_info",
    lambda state: "generate_query" if not state.get("error_message") else "synthesize_final_answer"
)

# After generating a query (or not), decide whether to execute or synthesize
sql_agent_subgraph_workflow.add_conditional_edges(
    "generate_query",
    route_after_generation, # This function still correctly routes based on history_query
    {
        "execute_query": "execute_query",
        "synthesize_final_answer": "synthesize_final_answer" # Route to final synthesis if no query
    }
)

# After executing query, always attempt final synthesis
sql_agent_subgraph_workflow.add_edge("execute_query", "synthesize_final_answer")

# After final synthesis, always end
sql_agent_subgraph_workflow.add_edge("synthesize_final_answer", END)

# Compile the subgraph
sql_agent_subgraph_app = sql_agent_subgraph_workflow.compile()

# Example invocation (for testing this file directly)
if __name__ == "__main__":
    # When running with `python -m sql_hub.sql_processing_subgraph`,
    # the top-level relative import `from .database_tools import ...` works.
    # We don't need the extra import logic here anymore.
    # try:
    #     from database_tools import llm, list_tables_tool, info_sql_tool, query_sql_tool
    # except ImportError:
    #     print("Could not import database_tools directly for testing. Ensure you run this from the project root or that sql_hub is accessible.")
    #     sys.exit(1)

    print("--- Testing SQL Agent Subgraph ---")
    # Ensure your environment has the necessary setup for database_tools (e.g., API keys, DB connection)
    initial_state = {
        "question": "who is service provider for store R002?",
        "max_iterations": 5 # Keep max iterations at 5 for now
    }
    try:
        final_state = sql_agent_subgraph_app.invoke(initial_state)
        print("--- SQL Agent Subgraph Final State ---")
        import json
        # Print final_answer if it exists
        if final_state.get("final_answer"):
             print(f"Final Answer: {final_state['final_answer']}")
        else:
             print("No final answer synthesized.")
        # Optionally print raw results too
        # print("\\nRaw Accumulated Results:")
        # print(json.dumps(final_state.get("final_db_results", []), indent=2))
        print("--- Full Final State ---")
        print(json.dumps(final_state, indent=2)) # Print full state for debugging
    except Exception as e:
        print("--- SQL Agent Subgraph Invocation Error ---")
        print(e) 