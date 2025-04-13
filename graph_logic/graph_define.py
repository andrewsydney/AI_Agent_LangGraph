from typing import List, Optional, Annotated
from pprint import pprint
import json

from typing_extensions import TypedDict

from langchain.schema import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
# Import StateGraph, END, START from langgraph
from langgraph.graph import StateGraph, END, START
from pydantic.v1 import BaseModel

# Import question_router from query_analysis.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph_logic.query_analysis import route_question, RouteQuery
from graph_logic.transform_query import rewrite_question_manual
from graph_logic.retrieve_generate import rag_chain
from index_doc import retriever
from graph_logic.retrieve_grader import retrieval_grader
from graph_logic.Hallucination_check import hallucination_grader
from graph_logic.answer_verify import answer_grader
# Import the new prompt
from graph_logic.graph_system_prompt import REWRITE_ANSWER_SYSTEM_PROMPT
# Remove the incorrect import
# from graph_logic.database_tools import llm as rewrite_llm

# Add imports needed for defining the rewrite LLM
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import subprocess

# Load environment variables for defining the rewrite LLM
load_dotenv()

# Define the LLM instance for the rewrite node
ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.3")
ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", 0.0))
rewrite_llm = ChatOllama(model=ollama_model, temperature=ollama_temperature, base_url=ollama_base_url)

# Import the compiled SQL agent subgraph
# Ensure sql_hub has an __init__.py if needed, or adjust path if necessary
try:
    # Assuming sql_hub is a sibling directory to graph_logic
    from sql_hub.sql_processing_subgraph import sql_agent_subgraph_app
except ImportError as e:
    print(f"Error importing SQL subgraph: {e}")
    print("Please ensure 'sql_hub' is importable (e.g., has __init__.py or correct path setup).")
    # Depending on requirements, might want to raise e or provide a fallback
    sql_agent_subgraph_app = None # Placeholder to avoid NameError, but graph will fail

# Define MAX_RETRIES constant
MAX_RETRIES = 1
# Define max iterations for the SQL subgraph
SQL_SUBGRAPH_MAX_ITERATIONS = 3

# Reducer function to always take the *last* value received in a step
def take_last(_, second):
    return second

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Uses Annotated with take_last for potential conflicts.
    Relies on combine_results node for merging documents and db_results.

    Attributes:
        question: The current question.
        original_question: The initial question.
        generation: LLM generation.
        documents: List of documents from retriever (before merge).
        db_results: List of results (strings) from database query.
        chat_history: Optional list of BaseMessages representing the conversation history.
        retry_count: number of retries.
        routing_decision: Optional[str]
        grading_decision: Optional[str]
    """

    # Use take_last for potential implicit conflicts, including for keys
    # explicitly updated by only one parallel branch when others return **state.
    question: Annotated[str, take_last]
    original_question: Annotated[Optional[str], take_last]
    generation: Annotated[str, take_last]
    retry_count: Annotated[Optional[int], take_last]

    # Fields updated by parallel branches, to be merged by combine_results
    documents: Annotated[List[Document], take_last] # Apply take_last
    db_results: Annotated[List[str], take_last]     # Apply take_last
    chat_history: Annotated[Optional[List[BaseMessage]], take_last] # Add chat_history field
    # Add a field to store the routing decision temporarily
    routing_decision: Optional[str]
    # Add field for grading decision
    grading_decision: Optional[str]

def retrieve(state):
    """
    Retrieve documents and update the 'documents' field.
    Returns the full state.
    """
    print("---RETRIEVE---")
    current_question_input = state["question"] # Get the input which might be a string or RewriteQuery

    # Extract the question string if it's a RewriteQuery object
    # Check if the input has the 'rewritten_question' attribute
    if hasattr(current_question_input, 'rewritten_question') and isinstance(getattr(current_question_input, 'rewritten_question'), str):
        question_to_retrieve = current_question_input.rewritten_question
        print(f"--- Using rewritten question for retrieval: {question_to_retrieve} ---")
    elif isinstance(current_question_input, str):
        question_to_retrieve = current_question_input # Assume it's already a string
        print(f"--- Using provided question for retrieval: {question_to_retrieve} ---")
    else:
        # Fallback or error handling if the question is neither a string nor the expected object
        print(f"--- ERROR: Unexpected question format: {type(current_question_input)}. Using original question if available. ---")
        # Attempt to use original_question as a fallback
        question_to_retrieve = state.get("original_question", "")
        if not question_to_retrieve:
             # If original_question is also not available or empty, handle appropriately
             print("--- ERROR: Could not determine a valid question for retrieval. Returning empty documents. ---")
             return {**state, "documents": []}


    print(f"---RETRIEVING DOCUMENTS FOR QUESTION: {question_to_retrieve}---")


    # Ensure retriever returns List[Document]
    # Pass the extracted string to the retriever
    try:
        fetched_documents: List[Document] = retriever.invoke(question_to_retrieve)
    except Exception as e:
        print(f"--- ERROR during retriever invocation: {e} ---")
        # Handle the error appropriately, e.g., return empty documents or re-raise
        return {**state, "documents": []}


    print(f"---RETRIEVED {len(fetched_documents)} DOCUMENTS---")
    if fetched_documents:
        # Optional: Print retrieved content
        for i, doc in enumerate(fetched_documents):
            print(f"--- RETRIEVED DOC {i+1} CONTENT START ---")
            try:
                # Check if doc is a Document object and has page_content
                if hasattr(doc, 'page_content'):
                    print(doc.page_content)
                else:
                    print(f"Document {i+1} has no page_content attribute.")
            except AttributeError:
                print("Could not access page_content.")
            print(f"--- RETRIEVED DOC {i+1} CONTENT END ---")
    else:
        print("---NO DOCUMENTS RETRIEVED---")

    # Update the 'documents' field. Return only the updated field.
    # return {**state, "documents": fetched_documents}
    return {"documents": fetched_documents}


# --- NODE: Call SQL Agent Subgraph ---
def call_sql_subgraph(state: GraphState):
    """
    Invokes the SQL agent subgraph to query the database based on the current question.
    Updates the 'db_results' field in the main graph state.
    """
    print("--- CALLING SQL AGENT SUBGRAPH ---")
    current_question_input = state["question"] # Get the input which might be a string or RewriteQuery

    # Extract the question string if it's a RewriteQuery object
    if hasattr(current_question_input, 'rewritten_question') and isinstance(getattr(current_question_input, 'rewritten_question'), str):
        question_for_subgraph = current_question_input.rewritten_question
        print(f"--- Using rewritten question for SQL subgraph: {question_for_subgraph} ---")
    elif isinstance(current_question_input, str):
        question_for_subgraph = current_question_input # Assume it's already a string
        print(f"--- Using provided question for SQL subgraph: {question_for_subgraph} ---")
    else:
        # Fallback or error handling if the question is neither a string nor the expected object
        print(f"--- ERROR: Unexpected question format for SQL subgraph: {type(current_question_input)}. Using original question if available. ---")
        question_for_subgraph = state.get("original_question", "")
        if not question_for_subgraph:
             print("--- ERROR: Could not determine a valid question for SQL subgraph. Passing empty string. ---")
             question_for_subgraph = "" # Pass empty string to avoid crashing subgraph

    if sql_agent_subgraph_app is None:
        print("--- ERROR: SQL Subgraph App not imported correctly. Cannot proceed. ---")
        # Return only the updated field with error
        return {"db_results": ["Error: SQL Subgraph could not be loaded."]}

    # Prepare input for the subgraph using the extracted question string
    subgraph_input_state = {
        "question": question_for_subgraph,
        "max_iterations": SQL_SUBGRAPH_MAX_ITERATIONS
    }

    try:
        # Invoke the subgraph
        # The subgraph manages its own internal state and loop
        final_subgraph_state = sql_agent_subgraph_app.invoke(subgraph_input_state)

        # Extract the final results from the subgraph's state
        subgraph_results = final_subgraph_state.get("final_db_results", [])
        print(f"--- SUBGRAPH FINAL RESULTS ({len(subgraph_results)} items): {subgraph_results} ---")

        # Update the main graph's db_results field
        # The 'take_last' annotation on db_results in GraphState handles the update
        # Return only the updated field
        # return {**state, "db_results": subgraph_results}
        return {"db_results": subgraph_results}

    except Exception as e:
        print(f"--- ERROR invoking SQL subgraph: {e} ---")
        # Update db_results with the error message. Return only this field.
        # return {**state, "db_results": [f"Error during SQL subgraph execution: {e}"]}
        return {"db_results": [f"Error during SQL subgraph execution: {e}"]}


# --- NODE: Combine Results (Restored) ---
def combine_results(state):
    """
    Combines documents retrieved from vector store (in 'documents')
    and results from database query (in 'db_results').
    Formats DB results as Langchain Documents.
    Updates 'documents' field with the combined list.
    Clears 'db_results'.
    """
    print("---COMBINE RESULTS---")
    retrieved_docs = state.get("documents", [])
    db_data = state.get("db_results", [])

    print(f"---Combining {len(retrieved_docs)} retrieved docs and {len(db_data)} DB results---")

    # Format DB results as Document objects
    formatted_db_docs: List[Document] = []
    for i, result in enumerate(db_data):
        doc = Document(page_content=f"Database Result {i+1}: {result}", metadata={"source": "database"})
        formatted_db_docs.append(doc)

    # Combine the lists
    combined_docs = retrieved_docs + formatted_db_docs
    print(f"---Total combined documents: {len(combined_docs)}---")

    # --- START: Added print statement for combined documents ---
    print("---Combined Documents Content (Input to Generate Node):---")
    if combined_docs:
        for i, doc in enumerate(combined_docs):
            print(f"--- Document {i+1} Start ---")
            try:
                print(f"Source: {doc.metadata.get('source', 'N/A')}") # Print source if available
                print(f"Content: {doc.page_content}")
            except AttributeError:
                print(f"Could not access content/metadata for document {i+1}")
            print(f"--- Document {i+1} End ---")
    else:
        print("--- No combined documents to display. ---")
    # --- END: Added print statement ---

    # Update the documents field, clear db_results, and pass other state through
    return {**state, "documents": combined_docs, "db_results": []}


# --- NODE: Agent ---
def agent(state):
    """
    Determines whether to retrieve documents, query database, or both,
    based on the initial routing decision. It also performs question rewriting using chat history.
    Updates the 'question' field with the rewritten question (if any).
    Returns the full state.
    """
    print("---AGENT: REWRITING QUESTION (with history)---") # Updated log
    original_question = state.get("original_question") or state["question"] # Use original if available
    chat_history = state.get("chat_history", []) # <<< GET CHAT HISTORY
    retry_count = state.get("retry_count", 0)

    if retry_count >= MAX_RETRIES:
        print("---AGENT: MAX RETRIES REACHED. ENDING.--")
        # Ensure 'generation' is updated to signal the end reason
        return {**state, "documents": [], "db_results": [], "generation": "Max retries reached, cannot answer question."}

    # Perform question rewriting before deciding on retrieval/DB query
    try:
        print("---REWRITING QUESTION (Agent Node)---") # Clarify log source
        # Create the dictionary expected by the updated rewrite_question_manual
        rewrite_input = {
            "question": original_question,
            "chat_history": chat_history # <<< PASS CHAT HISTORY
        }
        # Correctly CALL the rewrite function
        rewritten_question_str = rewrite_question_manual(rewrite_input) # Pass the dictionary

        # Check if rewriting produced a non-empty string
        if isinstance(rewritten_question_str, str) and rewritten_question_str:
            print(f"---REWRITTEN QUESTION: {rewritten_question_str}---")
            # Update the question field for subsequent nodes
            state["question"] = rewritten_question_str
        else:
            # Handle cases where rewriting failed or returned empty string
            print(f"--- WARNING during question rewriting (agent node): Rewriting returned empty or invalid. Using original question. ---")
            # Ensure state["question"] remains the original or previous question
            state["question"] = original_question # Fallback to original
    except Exception as e:
        # Catch any other unexpected errors during the rewrite call
        print(f"--- ERROR during question rewriting function call (agent node): {e} ---")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        # Fallback to original question
        state["question"] = original_question


    # Update retry count *after* potential rewriting
    # Note: Retry logic might need refinement based on graph goals
    updated_retry_count = retry_count + 1
    state["retry_count"] = updated_retry_count

    print(f"---AGENT: Updated state after rewrite (Retry {updated_retry_count})---")
    # Return only updated fields handled by take_last or explicitly modified here
    # The graph flow handles routing based on edges from 'agent'
    return {"question": state["question"], "retry_count": updated_retry_count}


def generate(state):
    """
    Generate answer based on current question and documents.
    Updates the 'generation' field in the state.
    """
    print("---GENERATE---")
    current_question = state["question"]
    documents = state["documents"]

    # Call the imported chain
    try:
        generation_result = rag_chain.invoke({"context": documents, "question": current_question})
    except Exception as e:
        print(f"--- ERROR during answer generation: {e} ---")
        generation_result = f"Error generating answer: {e}" # Provide error message as generation

    # Update generation field and pass other state through
    # Note: rag_chain already returns a string due to StrOutputParser, so no need for getattr
    return {**state, "generation": generation_result}


def grade_documents(state):
    """
    Determines whether retrieved documents are relevant to the current question.
    Updates the 'documents' field in the state with filtered list.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    current_question = state["question"]
    # original_question = state.get("original_question") # No longer needed
    documents = state["documents"]
    # retry_count = state.get("retry_count", 0) # No longer needed

    filtered_docs = []
    # Make sure documents is not None and is iterable
    if documents:
        for d in documents:
            # Ensure 'd' has 'page_content' attribute
            page_content = getattr(d, 'page_content', None)
            if page_content is None:
                print(f"---WARNING: Document object missing page_content: {d}---")
                continue # Skip this document
            
            # Restore original invocation using the imported grader
            score = retrieval_grader.invoke(
                {"question": current_question, "document": page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT--- ")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # No need for continue here, loop naturally continues
    else:
        print("---WARNING: No documents found in state for grading.---")

    # Update documents field with filtered list and pass other state through
    return {**state, "documents": filtered_docs}


def transform_query(state):
    """
    Transform the query to produce a better question.
    Retrieves chat_history from the state.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    original_question = state.get("original_question", question) # Use original if available
    retry_count = state.get("retry_count", 0) + 1 # Increment retry count
    chat_history = state.get("chat_history", []) # Get chat history from state

    print(f"--- Transforming query (Retry {retry_count}): {original_question} ---")
    print(f"--- Using history length: {len(chat_history)} ---")

    # Check if max retries reached
    if retry_count > MAX_RETRIES:
        print(f"--- Max retries ({MAX_RETRIES}) reached. Ending run. ---")
        # Decide how to end - maybe return a specific state or marker
        # For now, returning state which might lead to END via conditional edge
        return {**state, "grading_decision": "end_no_docs"} # Signal to end

    # Assuming rewrite_question_manual is imported and handles the actual rewrite
    # We need to ensure it accepts chat_history
    # The output of rewrite_question_manual might be just the string or a structured object
    try:
        # Pass chat_history to the rewrite function
        rewritten_question_result = rewrite_question_manual.invoke({
            "question": original_question,
            "chat_history": chat_history
        })
        print(f"--- Rewritten Question Result: {rewritten_question_result} ---")
        # The actual rewritten string might be inside the result object
        # Adjust based on what rewrite_question_manual actually returns
        # Assuming it returns a string directly for now:
        new_question = rewritten_question_result if isinstance(rewritten_question_result, str) else original_question

    except Exception as e:
        print(f"--- ERROR during query transformation: {e}. Using original question. ---")
        new_question = original_question # Fallback to original on error

    # Update state with the new question and incremented retry count
    # Return only the updated fields
    # return {**state, "question": new_question, "retry_count": retry_count}
    return {"question": new_question, "retry_count": retry_count}


def human(state):
    """
    Pass the current question to human_action.py script for processing.
    Passes along original_question.
    """
    print("---PASSING QUESTION TO HUMAN---")
    current_question = state["question"]
    original_question = state.get("original_question")
    retry_count = state.get("retry_count", 0)

    try:
        command = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "human_action.py"),
            "--question",
            current_question
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Output from human_action.py:")
        print(result.stdout)
        if result.stderr:
            print("Error output from human_action.py:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error calling human_action.py: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: human_action.py script not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return {**state, "original_question": original_question, "retry_count": retry_count}


### Edges ###

# New node for state initialization
def initialize_state(state: GraphState):
    """
    Initializes the state for the RAG/SQL graph run.
    Sets retry_count to 0 and copies original_question.
    Also copies chat_history from the input state.
    """
    print("---INITIALIZING STATE---")
    # state is the input dictionary passed to invoke, containing {"question": ..., "chat_history": ...}
    initial_state = GraphState(
        question=state['question'],
        original_question=state['question'], # Store the initial question
        generation="",
        documents=[],
        db_results=[],
        retry_count=0,
        routing_decision=None,
        grading_decision=None,
        chat_history=state.get('chat_history', []) # Get chat_history from input
    )
    print(f"--- Initial State Created with History Length: {len(initial_state['chat_history'])} ---")
    return initial_state

def decide_to_generate(state):
    """
    Determines whether to generate an answer, re-generate a question, or end due to retries.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    if not filtered_documents:
        print(f"---NO RELEVANT DOCUMENTS FOUND (Retry Count: {retry_count})---")
        if retry_count >= MAX_RETRIES:
            print(f"---MAX RETRIES ({MAX_RETRIES}) REACHED. ENDING PROCESS.---")
            return "end_no_docs"
        else:
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded, answers question, or max retries reached.
    Updates state with grading decision and returns it in a dictionary.
    Uses original_question for answer grading.
    """
    print("---CHECK HALLUCINATIONS---")
    # Get the potentially rewritten question for context if needed, but use original for grading
    # current_question = state["question"]
    original_question = state.get("original_question")
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0) # Get retry count

    # Ensure original_question is available
    if not original_question:
        print("---ERROR: original_question not found in state for grading! Using current question as fallback.---")
        original_question = state["question"] # Fallback, though this shouldn't happen in normal flow
    
    # Restore original hallucination check invocation
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    routing_decision = "" # Initialize routing decision variable

    if grade == "yes": # No hallucination
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs ORIGINAL QUESTION---")
        # Use original_question for the answer grader
        # Restore original answer grader invocation
        score = answer_grader.invoke({"question": original_question, "generation": generation})
        grade = score.binary_score
        if grade == "yes": # Grounded and useful
            print("---DECISION: GENERATION ADDRESSES ORIGINAL QUESTION---")
            # Use 'rewrite' as the key for the next step
            routing_decision = "rewrite"
        else: # Grounded but not useful
            print("---DECISION: GENERATION DOES NOT ADDRESS ORIGINAL QUESTION---")
            if retry_count >= MAX_RETRIES:
                print(f"---MAX RETRIES ({MAX_RETRIES}) REACHED (via not useful). ENDING PROCESS.---")
                routing_decision = "end_no_docs"
            else:
                routing_decision = "not useful"
    else: # Hallucination detected
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY--- (Hallucination)")
        if retry_count >= MAX_RETRIES:
            print(f"---MAX RETRIES ({MAX_RETRIES}) REACHED (via hallucination). ENDING PROCESS.---")
            routing_decision = "end_no_docs"
        else:
            routing_decision = "not supported"

    # Print the routing decision before returning
    print(f"--- GRADING ROUTING DECISION: {routing_decision} ---")
    # Return the decision in a dictionary to update state
    return {"grading_decision": routing_decision}

# --- NODE: Rewrite Final Answer ---
def rewrite_final_answer(state: GraphState):
    """
    Rewrites the final generated answer for conciseness and clarity using a dedicated prompt.
    """
    print("--- REWRITING FINAL ANSWER ---")
    original_question = state.get("original_question")
    generation_to_rewrite = state.get("generation")

    # --- START: Added print statements for debugging ---
    print(f"--- Rewrite Input - Original Question: {original_question} ---")
    print(f"--- Rewrite Input - Generation to Rewrite: {generation_to_rewrite} ---")
    # --- END: Added print statements ---

    if not generation_to_rewrite:
        print("--- WARNING: No generation found in state to rewrite. Skipping. ---")
        return {"generation": ""} # Return empty string explicitly if no input

    if not original_question:
        print("--- WARNING: Original question not found for rewrite context. Proceeding without it. ---")
        original_question = "Not available"

    try:
        # --- Modification: Use direct invoke for Ollama rewrite ---
        # # Removed Restore Original LangChain Setup:
        # rewrite_prompt = ChatPromptTemplate.from_messages([...])
        # rewrite_chain = rewrite_prompt | rewrite_llm
        # invoke_input = {...}
        # rewritten_answer_obj = rewrite_chain.invoke(invoke_input)

        # Assume rewrite_llm is an initialized ChatOllama instance
        # Assume REWRITE_ANSWER_SYSTEM_PROMPT is the template string imported

        # Manually format the prompt string
        formatted_prompt = REWRITE_ANSWER_SYSTEM_PROMPT.format(
            original_question=original_question,
            generation=generation_to_rewrite
        )
        
        # Directly invoke the Ollama LLM
        rewritten_answer_obj = rewrite_llm.invoke(formatted_prompt)
        # --- End Modification ---

        rewritten_answer = getattr(rewritten_answer_obj, 'content', str(rewritten_answer_obj)).strip()

        print(f"--- REWRITTEN ANSWER (direct invoke): {rewritten_answer} ---") # Update log message
        return {"generation": rewritten_answer} # Return only the updated generation

    except Exception as e:
        print(f"--- ERROR during final answer rewriting: {e} ---")
        # Return the original generation in case of error
        return {"generation": generation_to_rewrite}

# Define the function that determines the initial route
def determine_initial_route(state: GraphState):
    """
    Determines the initial route based on the 'routing_decision' key in the state,
    which is now always set to 'agent' by the modified initialize_state.
    """
    print(f"--- Determining initial route based on state decision: {state.get('routing_decision')} ---")
    if state.get("routing_decision") == "human":
        print("--- Routing to: human ---")
        return "human"
    else:
        # Default to agent if decision is 'agent' or missing/unexpected
        print("--- Routing to: agent ---")
        return "agent"