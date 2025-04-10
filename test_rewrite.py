# test_rewrite.py
import os
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI # Remove OpenAI import
from langchain_ollama import ChatOllama  # Import ChatOllama
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# # Ensure the OpenAI API key is set # Remove OpenAI check
# if "OPENAI_API_KEY" not in os.environ:
#     print("Error: OPENAI_API_KEY environment variable not set.")
#     exit()

# Read Ollama configuration from environment variables
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # Default if not set
ollama_model = os.getenv("OLLAMA_MODEL", "llama3") # Default model if not set
ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", 0.0)) # Default temperature

# System prompt from graph_logic/graph_system_prompt.py
REWRITE_ANSWER_SYSTEM_PROMPT_TEMPLATE = """From the 'Answer to Rewrite' below, rewrite the answer to be more concise and clear. rewirte to more natural and conversational language.
only rewrite the answer, do not include any other text. don't include other information not related to the original question.

Original User Question:
{original_question}

Answer to Rewrite:
{generation}
""" # Keep the original prompt for reference, rename slightly for template usage

# Simple test prompt (Keep commented out)
# SIMPLE_TEST_PROMPT = "Tell me a short joke."

# Simplified rewrite prompt (Keep commented out)
# SIMPLIFIED_REWRITE_PROMPT = """...""

# Prompt to simply repeat the input text (Keep commented out)
# REPEAT_INPUT_PROMPT_TEMPLATE = "Repeat the following text exactly as it is given:\n\n{generation}"

# Input text provided by the user (Restore the original long text)
generation_to_rewrite = "The primary contact number for r001 is not explicitly specified. However, the additional or alternative phone numbers associated with r001 are +861082345001, +861082345002, +861082345003, +861082345004, +861082345005, +861082345006, +861082345007, +861082345008, +861082345009, and +861082345010. These numbers can be considered as related to r001."

# Example original question (Restore)
original_question = "What are the phone numbers for store r001?"

# Initialize the LLM (using ChatOllama)
rewrite_llm = ChatOllama(
    base_url=ollama_base_url,
    model=ollama_model,
    temperature=ollama_temperature
)

# --- Bypassing ChatPromptTemplate and Chain ---

# Invoke the chain to get the rewritten answer
try:
    # Manually format the original complex prompt string
    formatted_prompt = REWRITE_ANSWER_SYSTEM_PROMPT_TEMPLATE.format(
        original_question=original_question,
        generation=generation_to_rewrite
    )
    print(f"\n--- MANUALLY FORMATTED PROMPT ---\n{formatted_prompt}\n")

    # Directly invoke the LLM with the formatted prompt string
    response_obj = rewrite_llm.invoke(formatted_prompt)

    # Print the raw response object for debugging
    print("\n--- RAW RESPONSE OBJECT ---")
    print(response_obj)

    # Extract content from the response object (works for AIMessage)
    rewritten_answer = getattr(response_obj, 'content', str(response_obj)).strip()

    print("\n--- ORIGINAL GENERATION ---") # Restore original print section
    print(generation_to_rewrite)
    print("\n--- REWRITTEN ANSWER (direct invoke) ---") # Update label
    print(rewritten_answer)

except Exception as e:
    print(f"An error occurred: {e}") 