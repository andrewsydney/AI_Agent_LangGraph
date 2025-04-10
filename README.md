# Advanced RAG and SQL Querying Agent with LangGraph

This project implements a sophisticated question-answering agent using LangGraph. It leverages Retrieval-Augmented Generation (RAG) from a vector store and can query SQL databases to provide comprehensive answers. The agent features self-correction mechanisms, including document relevance grading, hallucination checks, and answer rewriting.

## Features

*   **Retrieval-Augmented Generation (RAG):** Retrieves relevant documents from a vector store (e.g., ChromaDB, based on file structure) to ground responses.
*   **SQL Database Querying:** Integrates a LangGraph subgraph (`sql_hub`) to query SQL databases for structured data.
*   **Conditional Routing:** Intelligently routes user queries based on initial analysis (though the main flow seems to prioritize RAG/SQL over direct answers or human handoff defined in other files).
*   **Query Transformation:** Rewrites user questions for improved retrieval accuracy.
*   **Multi-Step Reasoning:**
    *   Retrieves from vector store and/or SQL database.
    *   Combines information from different sources.
    *   Grades the relevance of retrieved documents.
    *   Generates an initial answer based on context.
*   **Answer Evaluation & Refinement:**
    *   Grades the generated answer against the retrieved documents (Hallucination Check).
    *   Grades the answer's relevance to the original question.
    *   Rewrites the final answer if deemed necessary by the grading steps.
*   **Command-Line Interface:** Interact with the agent via `run.py`.
*   **Powered by LangGraph & Ollama:** Built on the LangGraph framework using Ollama (defaults to Llama3.3) for language model tasks.

## Architecture

The core logic resides in the `graph_logic/` directory:

1.  **`graph_define.py`:** Defines the `GraphState` TypedDict and the individual node functions (retrieve, call_sql_subgraph, combine_results, grade_documents, generate, grade_generation_v_documents_and_question, transform_query, rewrite_final_answer, etc.).
2.  **`graph_flow.py`:** Defines the `StateGraph`, connects the nodes with conditional edges, and compiles the final LangGraph `app`.
3.  **`run.py`:** The main entry point. It loads the compiled LangGraph `app` and runs an asynchronous command-line loop to interact with the user.
4.  **`sql_hub/`:** Contains the subgraph logic for interacting with SQL databases.
5.  **Vector Store:** Uses a retriever (likely configured in `index_doc.py` or similar) to interact with a vector database (inferred `graph_logic/vector_store/`).

## Getting Started

### Prerequisites

*   Python 3.8+
*   Git
*   Ollama installed and running. Ensure the model specified in `.env` (default: `llama3.3`) is available.
    *   [Ollama Website](https://ollama.com/)
*   Access to the SQL database targeted by `sql_hub` (if used).
*   Potentially specific setup for the vector store used.

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  Create a `.env` file in the project root directory by copying `.env.example` (if available) or creating it manually.
2.  Add the necessary environment variables. Key variables likely include:
    *   `OLLAMA_HOST`: The base URL for your Ollama instance (e.g., `http://localhost:11434`).
    *   `OLLAMA_MODEL`: The Ollama model to use (e.g., `llama3.3`).
    *   Database connection details for the `sql_hub` (check `sql_hub/database_tools.py` or similar for required variables).
    *   Any API keys or credentials needed for vector store access.
3.  **Important:** Ensure `.env` is listed in your `.gitignore` file and is **never** committed to version control.

### Running the Application

Start the interactive command-line interface:

```bash
python run.py
```

Enter your questions when prompted. Type `quit` or `q` to exit.

## How it Works (Simplified Flow)

1.  **Initialize:** The graph starts and potentially performs initial routing based on `determine_initial_route`.
2.  **Agent Step:** The `agent` node likely prepares the question for retrieval/querying.
3.  **Parallel Retrieval/Query:** The graph runs `retrieve` (vector store) and `call_sql_subgraph` (SQL DB) in parallel.
4.  **Combine:** `combine_results` merges documents from both sources.
5.  **Grade Documents:** `grade_documents` assesses the relevance of the combined documents.
6.  **Conditional Generation:** Based on the grading (`decide_to_generate`):
    *   If documents are relevant -> `generate`.
    *   If documents are not relevant but retries remain -> `transform_query` (rewrite and loop back to agent).
    *   If no relevant documents and no retries -> `END`.
7.  **Generate:** `generate` creates an answer using the relevant documents.
8.  **Grade Generation:** `grade_generation_v_documents_and_question` checks the answer for hallucination and relevance.
9.  **Conditional Refinement:** Based on generation grading:
    *   If answer is good -> `rewrite_final_answer` (might just format or pass through) -> `END`.
    *   If answer needs improvement -> `transform_query` (potentially triggering a full loop again).
    *   If answer is unsupported/not useful -> `transform_query`.

## Contributing

Contributions are welcome! Please follow standard GitHub practices (fork, feature branch, pull request).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

*   **Your Name** - *Initial work* - [Your GitHub Profile](https://github.com/your-username)

See also the list of [contributors](https://github.com/your-username/your-project-name/contributors) who participated in this project.

## Acknowledgments

*   Hat tip to anyone whose code was used
*   Inspiration
*   etc 