### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import os # Import os module
from dotenv import load_dotenv # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
# Get the base URL from environment variable, using OLLAMA_HOST and providing a default
ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Get the model name from environment variable, providing a default
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.3")
# Get the temperature from environment variable, providing a default
ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", 0.0))

# LLM
llm = ChatOllama(model=ollama_model, temperature=ollama_temperature, base_url=ollama_base_url)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Re-enable the original Chain setup
rag_chain = prompt | llm | StrOutputParser()

# Comment out the calling code, as it's not needed during import
# Run
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)