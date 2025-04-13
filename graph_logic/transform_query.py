### Question Re-writer
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import os # Import os module
from dotenv import load_dotenv # Import dotenv
from .graph_system_prompt import TRANSFORM_QUERY_SYSTEM_PROMPT as system
from pydantic.v1 import BaseModel, Field # Import BaseModel and Field
import json # Import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # Add BaseMessage imports
from typing import List # Import List
# import re # Keep regex import commented out

# Load environment variables from .env file
load_dotenv()

# Get the base URL from environment variable, using OLLAMA_HOST and providing a default
ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Get the model name from environment variable, providing a default
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.3")
# Get the temperature from environment variable, providing a default
ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", 0.0))

# Data model for the rewritten query
class RewriteQuery(BaseModel):
    """Transform the query to produce a better question."""
    rewritten_question: str = Field(
        description="Improved question based on the original."
    )

# LLM - Use the base LLM instance, format="json" is not strictly needed when using with_structured_output
# llm = ChatOllama(model=ollama_model, temperature=ollama_temperature, base_url=ollama_base_url, format="json")
llm = ChatOllama(model=ollama_model, temperature=ollama_temperature, base_url=ollama_base_url)

# Use structured output with the Pydantic model
structured_llm_rewriter = llm.with_structured_output(RewriteQuery)

# Helper function to format chat history
def format_history_for_prompt(history: List[BaseMessage]) -> str:
    """Formats a list of BaseMessage objects into a string for the prompt."""
    if not history:
        return "No previous conversation history."
    formatted_lines = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_lines.append(f"AI: {msg.content}")
        else: # Handle other message types if necessary
            formatted_lines.append(f"{msg.type.capitalize()}: {msg.content}")
    return "\n".join(formatted_lines)

# Create the prompt template using the imported system prompt
# No need to manually add JSON instructions here
rewrite_prompt_template = ChatPromptTemplate.from_messages([
    ("system", system), 
    ("human", "Chat History:\n{chat_history}\n\nQuestion:\n{question}") # Use curly braces directly
])

# Define the rewriter chain
rewriter_chain = rewrite_prompt_template | structured_llm_rewriter

# Rewriting function using the structured output chain
def rewrite_question_manual(input_dict: dict) -> str: # Expects dict with 'question' and 'chat_history'
    """
    Rewrites the user question using chat history via a structured output chain.
    Args: input_dict: A dictionary containing 'question' (str) and 'chat_history' (List[BaseMessage]).
    Returns: A string containing the rewritten question.
    """
    question = input_dict.get("question")
    chat_history = input_dict.get("chat_history", [])

    if not question:
        print("--- ERROR: No question provided to rewrite_question_manual. Returning empty string. ---")
        return "" # Or handle error appropriately

    # Format history here, although the chain expects the list
    # Let's pass the formatted string AND the original question to the chain input dictionary
    formatted_history_str = format_history_for_prompt(chat_history)

    chain_input = {
        "question": question,
        "chat_history": formatted_history_str # Pass formatted string based on template
    }

    print(f"--- DEBUG: Invoking structured rewriter chain with input keys: {list(chain_input.keys())} ---")

    try:
        # Invoke the structured output chain
        result: RewriteQuery = rewriter_chain.invoke(chain_input)
        rewritten = result.rewritten_question
        print(f"--- Structured Rewriter Output: {rewritten} ---")
        return rewritten

    except Exception as e:
        print(f"--- ERROR: Structured rewriter chain failed: {e}. Falling back to original question. ---")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return question # Fallback string

# ... (Keep the rest of the file as is, removing old manual parsing logic if desired, but not strictly necessary) ...