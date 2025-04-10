### Retrieval Grader

from typing import Literal, List, TypedDict
import os # Import os module
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic.v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Import the system prompt
from .graph_system_prompt import RETRIEVE_GRADER_SYSTEM_PROMPT

# Load environment variables from .env file
load_dotenv()

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    documents: List[str]


# Get the base URL from environment variable, using OLLAMA_HOST and providing a default
ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Get the model name from environment variable, providing a default
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.3")
# Get the temperature from environment variable, providing a default
ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", 0.0))

# LLM Setup
llm = ChatOllama(model=ollama_model, format="json", temperature=ollama_temperature, base_url=ollama_base_url)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt template string (Keep the system prompt definition)
system = RETRIEVE_GRADER_SYSTEM_PROMPT # Use the imported prompt

# Restore ChatPromptTemplate setup
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Restore chain setup
retrieval_grader = grade_prompt | structured_llm_grader

# Remove comments related to manual formatting/direct invoke
# # --- You would call it like this (example, adjust based on actual usage) ---
# # question = "agent memory"
# ... (removed example call comments) ...

# Comment out the calling code to avoid execution on import
# question = "agent memory"
# docs = retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))