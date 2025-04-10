### Router

from typing import Literal
import os # Import os module
from dotenv import load_dotenv
import json # Import json module

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic.v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Import the system prompt
from .graph_system_prompt import QUERY_ANALYSIS_SYSTEM_PROMPT # Re-add import

# Load environment variables from .env file
load_dotenv()

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["agent", "human"] = Field(
        ...,
        description="Given a user question choose to route it to agent or human.",
    )


# Get the base URL from environment variable, using OLLAMA_HOST and providing a default
ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Get the model name from environment variable, providing a default
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.3")
# Get the temperature from environment variable, providing a default
ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", 0.0))

# LLM with function call
# Initialize with the base URL from the environment
llm = ChatOllama(model=ollama_model, temperature=ollama_temperature, base_url=ollama_base_url, format="json")
# structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt template string
system_template = QUERY_ANALYSIS_SYSTEM_PROMPT + "\n\nReturn your decision as a JSON object with a single key 'datasource' and a value of either 'agent' or 'human'. Example: {{\"datasource\": \"agent\"}}"

# Define the routing function (using direct invoke)
def route_question(question: str) -> RouteQuery:
    """
    Routes the user question by asking the LLM for JSON and parsing it manually.
    Args: question: The user's question.
    Returns: A RouteQuery object indicating the datasource.
    """
    formatted_prompt = system_template.format(question=question)
    print(f"--- DEBUG: Routing Prompt Sent: {formatted_prompt[:500]}... ---")
    try:
        # Directly invoke the base LLM
        response = llm.invoke(formatted_prompt)
        response_content = getattr(response, 'content', str(response)).strip()
        print(f"--- DEBUG: Raw LLM Routing Response: {response_content} ---")

        # Attempt to parse the JSON response
        try:
            parsed_json = json.loads(response_content)
            datasource = parsed_json.get("datasource")
            if datasource in ["agent", "human"]:
                print(f"--- Parsed Datasource: {datasource} ---")
                return RouteQuery(datasource=datasource)
            else:
                print(f"--- ERROR: Invalid datasource value in JSON: {datasource}. Defaulting to 'agent'. ---")
                return RouteQuery(datasource='agent')
        except json.JSONDecodeError as json_e:
            print(f"--- ERROR: Failed to decode LLM JSON response: {json_e}. Raw response: {response_content}. Defaulting to 'agent'. ---")
            # Fallback: Check if raw string is just 'agent' or 'human'
            if response_content.lower() == 'agent':
                 return RouteQuery(datasource='agent')
            if response_content.lower() == 'human':
                 return RouteQuery(datasource='human')
            return RouteQuery(datasource='agent')
        except Exception as parse_e: # Catch other potential errors during parsing/access
            print(f"--- ERROR: Error processing LLM response: {parse_e}. Defaulting to 'agent'. ---")
            return RouteQuery(datasource='agent')
            
    except Exception as invoke_e:
        print(f"--- ERROR: LLM invocation failed during routing: {invoke_e}. Defaulting to 'agent'. ---")
        return RouteQuery(datasource='agent')

# If run as the main module, execute the test code (using the function)
if __name__ == "__main__":
    print("Please enter your question")
    print("\nEnter 'q' to exit the program")
    
    while True:
        user_input = input("\nPlease enter your question: ")
        if user_input.lower() == 'q':
            print("Thank you for using, goodbye!")
            break
        
        if not user_input.strip():
            print("Question cannot be empty, please re-enter.")
            continue
            
        print(f"User question: {user_input}")
        # Use the route_question function
        try:
            result = route_question(user_input)
            print(f"Routing result: {result.datasource}")
            
            if result.datasource == "human":
                print("This question will be answered by a human.")
            elif result.datasource == "agent":
                print("This question will be answered by an AI agent.")
        except Exception as e:
            print(f"An error occurred during routing: {e}")