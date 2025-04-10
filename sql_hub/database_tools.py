from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the path to the database file relative to the project root
# Assuming this script is in graph_logic and agent.db is in the parent directory (project root)
project_root = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(project_root, 'agent.db')
db_uri = f"sqlite:///{db_path}"

# Initialize the SQLDatabase object
db = SQLDatabase.from_uri(db_uri)

# Get Ollama configuration from environment variables
ollama_base_url = os.getenv("OLLAMA_BASE_URL") # Default is http://localhost:11434
ollama_model = os.getenv("OLLAMA_MODEL", "llama3") # Default to llama3 if not set
ollama_temperature_str = os.getenv("OLLAMA_TEMPERATURE", "0.0") # Default to 0.0
try:
    ollama_temperature = float(ollama_temperature_str)
except ValueError:
    print(f"Warning: Invalid OLLAMA_TEMPERATURE '{ollama_temperature_str}', defaulting to 0.0")
    ollama_temperature = 0.0


# Initialize Ollama LLM using environment variables
# Ensure Ollama service is running
llm = ChatOllama(model=ollama_model, temperature=ollama_temperature, base_url=ollama_base_url)

# Initialize the tools
info_sql_tool = InfoSQLDatabaseTool(db=db)
list_tables_tool = ListSQLDatabaseTool(db=db)
query_sql_tool = QuerySQLDatabaseTool(db=db)
# Initialize the query checker tool using Ollama
query_checker_tool = QuerySQLCheckerTool(db=db, llm=llm)

# List of tools to be used by an agent or other parts of the application
# QuerySQLCheckerTool runs the query after checking it, so we often use it instead of the raw query_sql_tool
tools = [
    info_sql_tool,
    list_tables_tool,
    query_checker_tool, # Use the checker tool which includes execution
    # query_sql_tool, # Usually replaced by query_checker_tool
]

# Example usage section removed or commented out as the primary goal is tool export
# if __name__ == "__main__":
#     print(f"Database URI: {db_uri}")
#     # Example check and run (replace with a valid table/query)
#     # try:
#     #     result = query_checker_tool.run("SELECT name FROM sqlite_master WHERE type='table';")
#     #     print(f"Checked query result: {result}")
#     # except Exception as e:
#     #     print(f"An error occurred: {e}")
#     print("Database tools initialized with Ollama query checker.")
#     print("Available tools:", [tool.name for tool in tools]) 