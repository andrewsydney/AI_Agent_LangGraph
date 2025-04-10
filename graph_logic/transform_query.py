### Question Re-writer
from langchain_core.prompts import ChatPromptTemplate 
from langchain_ollama import ChatOllama
import os # Import os module
from dotenv import load_dotenv # Import dotenv
from .graph_system_prompt import TRANSFORM_QUERY_SYSTEM_PROMPT as system
from pydantic.v1 import BaseModel, Field # Import BaseModel and Field
import json # Import json

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

# LLM
llm = ChatOllama(model=ollama_model, temperature=ollama_temperature, base_url=ollama_base_url, format="json")
structured_llm_rewriter = llm.with_structured_output(RewriteQuery)

# Update system template to explicitly ask for JSON
system_template = system + "\n\nReturn your rewritten question as a JSON object with a single key 'rewritten_question'. Example: {{\"rewritten_question\": \"What specific phone numbers are assigned to store r001?\"}}"

# Rewriting function using manual JSON parsing
def rewrite_question_manual(question: str) -> RewriteQuery:
    """
    Rewrites the user question by asking the LLM for JSON and parsing it manually.
    Args: question: The user's question to rewrite.
    Returns: A RewriteQuery object containing the rewritten question.
    """
    formatted_prompt = system_template.format(question=question)
    print(f"--- DEBUG: Rewriting Prompt Sent: {formatted_prompt[:500]}... ---")
    try:
        # Directly invoke the base LLM
        response = llm.invoke(formatted_prompt)
        response_content = getattr(response, 'content', str(response)).strip()
        print(f"--- DEBUG: Raw LLM Rewriting Response: {response_content} ---")

        # Attempt to parse the JSON response
        try:
            parsed_json = json.loads(response_content)
            rewritten = parsed_json.get("rewritten_question")
            if isinstance(rewritten, str) and rewritten:
                print(f"--- Parsed Rewritten Question: {rewritten} ---")
                return RewriteQuery(rewritten_question=rewritten)
            else:
                print(f"--- ERROR: Invalid/missing 'rewritten_question' value in JSON: {rewritten}. Falling back to original. ---")
                return RewriteQuery(rewritten_question=question) # Fallback
        except json.JSONDecodeError as json_e:
            print(f"--- ERROR: Failed to decode LLM JSON response for rewrite: {json_e}. Raw: {response_content}. Falling back to original. ---")
            return RewriteQuery(rewritten_question=question) # Fallback
        except Exception as parse_e:
            print(f"--- ERROR: Error processing LLM rewrite response: {parse_e}. Falling back to original. ---")
            return RewriteQuery(rewritten_question=question) # Fallback
            
    except Exception as invoke_e:
        print(f"--- ERROR: LLM invocation failed during rewrite: {invoke_e}. Falling back to original. ---")
        return RewriteQuery(rewritten_question=question) # Fallback

# Removed Prompt template string
# system_template = """..."""

# Restore ChatPromptTemplate setup
# re_write_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system),
#         (
#             "human",
#             "Here is the initial question: \n\n {question} \n Formulate an improved question based on the original.",
#         ),
#     ]
# )

# Restore chain setup
# question_rewriter = re_write_prompt | structured_llm_rewriter

# Remove the rewriting function
# def rewrite_question(question: str) -> RewriteQuery:
#     """...
#     """
#     ...
#     return result

# Remove comments related to manual formatting/direct invoke
# # --- Example of how to call (adjust based on actual usage) ---
# ... (removed example call comments) ...

# Comment out or remove the old StrOutputParser version if not needed
# # question_rewriter = re_write_prompt | llm | StrOutputParser()