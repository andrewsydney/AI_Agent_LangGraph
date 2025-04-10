# Project Summary

This project implements a RAG (Retrieval-Augmented Generation) application using LangGraph, with integrations for Slack and SQL database querying.

## Root Directory (`/`)

-   **`run.py`**: The main entry point of the application. It loads environment variables, imports the compiled LangGraph workflow (`app` from `graph_logic.graph_flow`), and runs an asynchronous loop to receive user input and invoke the graph.
-   **`requirements.txt`**: Lists the required Python packages for the project.
-   **`.env`**: Stores environment variables like API keys, Slack tokens, Ollama configuration, and database paths. Should not be committed to version control.
-   **`agent.db`**: The SQLite database file used by the SQL agent subgraph for data retrieval.
-   **`test_rewrite.py`**: Likely contains tests, potentially focused on the query/answer rewriting functionality within the graph.
-   **`venv/`**: Directory containing the Python virtual environment to isolate dependencies.
-   **`logs/`**: Directory likely used for storing application logs.
-   **`__pycache__/`**: Standard Python directory for storing bytecode cache files.
-   **`.cursor/`**: Metadata directory used by the Cursor editor.
-   **`.DS_Store`**: Metadata file created by macOS Finder.

## `slack_hub/`

-   **`slack_app.py`**: Handles the Slack integration.
    -   Uses `slack-bolt` and `Flask` to create a Slack app instance.
    -   Listens for `@app_mention` events in Slack channels.
    -   Fetches the bot's user ID.
    -   Extracts the user's query from the mention text.
    -   Acknowledges the request in the Slack thread.
    -   Contains placeholder logic to invoke the main LangGraph application (from `run.py`/`graph_logic`) with the user's query and send the result back to Slack.
    -   Provides a `/slack/events` endpoint for Slack API calls and a `/` health check endpoint.

## `graph_logic/`

This directory contains the core logic for the main LangGraph RAG workflow.

-   **`graph_define.py`**: Defines the core components of the main graph:
    -   `GraphState`: A `TypedDict` defining the state passed between nodes (question, documents, generation, etc.).
    -   Node Functions: Implements the logic for each step in the graph (e.g., `retrieve`, `generate`, `grade_documents`, `transform_query`, `rewrite_final_answer`, `call_sql_subgraph`, `combine_results`).
    -   LLM Initialization: Defines the `ChatOllama` instance used for rewriting.
    -   Imports the SQL agent subgraph (`sql_agent_subgraph_app`).
-   **`graph_flow.py`**: Builds and compiles the main LangGraph workflow (`app`).
    -   Instantiates `StateGraph` with `GraphState`.
    -   Adds nodes defined in `graph_define.py`.
    -   Defines the edges and conditional edges connecting the nodes, dictating the flow of execution based on the state.
    -   Sets the entry and end points of the graph.
    -   Compiles the graph into the `app` object, which is imported by `run.py`.
-   **`index_doc.py`**: (Likely) Handles document loading, splitting, embedding generation, and vector store setup/retrieval (defines the `retriever` used in `graph_define.py`).
-   **`query_analysis.py`**: Implements the `route_question` function to decide the initial path (e.g., agent vs. human) based on the query.
-   **`transform_query.py`**: Implements query rewriting logic (`rewrite_question_manual`) used when retrieved documents are not relevant.
-   **`retrieve_generate.py`**: Defines the core RAG chain (`rag_chain`) used within the `generate` node.
-   **`retrieve_grader.py`**: Implements `retrieval_grader` to assess the relevance of retrieved documents to the query.
-   **`Hallucination_check.py`**: Implements `hallucination_grader` to check if the generated answer is grounded in the retrieved documents.
-   **`answer_verify.py`**: Implements `answer_grader` to check if the generated answer addresses the original question.
-   **`graph_system_prompt.py`**: Contains system prompts (e.g., `REWRITE_ANSWER_SYSTEM_PROMPT`) used by LLMs in different graph nodes.
-   **`human_action.py`**: Defines a node for potential human-in-the-loop intervention.
-   **`vector_store/`**: (Likely) Directory used to store the vector database index files.
-   **`__init__.py`**: Makes the directory a Python package.

## `sql_hub/`

This directory contains the logic for a separate LangGraph subgraph focused on SQL database interaction.

-   **`sql_processing_subgraph.py`**: Defines and builds the SQL agent subgraph (`sql_agent_subgraph_app`).
    -   Defines `SqlAgentSubgraphState` for managing state within this subgraph (question, tables, schema, history, results).
    -   Defines nodes for the subgraph's workflow: `get_initial_db_info`, `generate_next_query`, `execute_query`, `synthesize_final_answer`.
    -   Uses LangChain SQL tools and an LLM (from `database_tools.py`) to interact with the database iteratively.
    -   Includes routing logic (`route_after_execution`, `route_after_synthesis`) to handle the loop and exit conditions.
    -   Compiles the subgraph into `sql_agent_subgraph_app`, which is imported by `graph_logic.graph_define.py`.
-   **`database_tools.py`**: Initializes and configures tools for database interaction.
    -   Connects to the SQLite database (`agent.db`).
    -   Initializes an Ollama LLM instance (`llm`) based on `.env` settings.
    -   Creates LangChain SQLDatabase tools: `ListSQLDatabaseTool`, `InfoSQLDatabaseTool`, `QuerySQLDatabaseTool`, and `QuerySQLCheckerTool` (using the `llm` for checking).
    -   Exports these tools for use by the SQL subgraph.
-   **`__init__.py`**: Makes the directory a Python package. Allows importing `sql_agent_subgraph_app` from `graph_logic`. 

