### Hallucination Grader
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import os # Import os module
from dotenv import load_dotenv # Import dotenv
from typing_extensions import TypedDict
from typing import List

# Import the system prompt
from .graph_system_prompt import HALLUCINATION_CHECK_SYSTEM_PROMPT

# Load environment variables from .env file
load_dotenv()

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# Get the base URL from environment variable, using OLLAMA_HOST and providing a default
ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Get the model name from environment variable, providing a default
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.3")
# Get the temperature from environment variable, providing a default
ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", 0.0))

# LLM with function call
llm = ChatOllama(model=ollama_model, temperature=ollama_temperature, base_url=ollama_base_url)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt template string
system = HALLUCINATION_CHECK_SYSTEM_PROMPT # Use the imported prompt

# Restore ChatPromptTemplate setup
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# Restore chain setup
hallucination_grader = hallucination_prompt | structured_llm_grader

# Comment out the calling code to avoid execution on import
# hallucination_grader.invoke({"documents": docs, "generation": generation})