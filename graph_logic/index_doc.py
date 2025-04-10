### Document Vector Indexing and Retrieval

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import time
import os
import shutil
import logging
import sys
from datetime import datetime
import chromadb # Import chromadb client
import argparse # Add argparse import

# Configure logging
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "index_doc.log")

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure vector store path - inside the graph_logic directory
# base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# vector_store_path = os.path.join(base_dir, "ai_knowledge_data")
vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
logger.info(f"Vector store path: {vector_store_path}")

# Get the base URL from environment variable
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
logger.info(f"Using Ollama base URL: {ollama_base_url}") # Log the URL being used

# Initialize embedding model with base_url
embd = OllamaEmbeddings(
    model="llama3.3",
    base_url=ollama_base_url # Add the base_url parameter
)

# --- Default Documents (for initial build) ---
default_docs_content = [
    "Each retail store has a unique identifier starting with 'R' followed by three digits (e.g., R001). This RXXX number is normally called store number.",
    "The specific store name corresponding to each 'RXXX' number is defined in the database.",
    "Retail store phone extensions are 7-digit numbers, however the 3-digit extension number can also be used for dialing within the store.",
    "The 7-digit extension format is: '8' + (3-digit store number, derived from the 'R' number) + (3-digit internal extension).",
    "For example, extension 8001601 represents internal extension 601 at store R001.",
    "Similarly, 8003626 represents internal extension 626 at store R003.",
    "Internal extensions can be referred to by their full 7-digit number system-wide, or by the 3-digit extension number within the store.",
    "Within a specific store, the 3-digit extension number (e.g., 601) can also be used for dialing.",
    "Both the 7-digit extension number and the 3-digit extension number can be used for dialing.",
    "For example, at store R001, people can dial 8001601 or 601 to reach the manager at R001 store. Or dial 8002601 to reach manager at store R002.",
    "Compulsorily, every retail store is required to have the following internal extensions assigned: 601 (Manager), 602 (Receiving 1), 621 (Genius Admin), 626 (Business), and 681 (Operator 1).",
    "The following internal extensions are optional and can be assigned as needed: 603 (Receiving 2), 608 (Flex 1), 609 (Flex 2), 622 (Genius Admin 2), 623 (Genius Admin 3), 624 (Genius Admin 4), 631 (Repair Room 1), 632 (Repair Room 2), 633 (Repair Room 3), 634 (Repair Room 4), 635 (Repair Room 5), 636 (Repair Room 6), 637 (Repair Room 7), 638 (Repair Room 8), 682 (Operator 2), 683 (Operator 3).",
    "These internal extensions can be shared across multiple physical phones: 601, 602, 603, 608, 609, 626.",
    "These internal extensions must be dedicated to a single physical phone (cannot be shared): 621, 622, 623, 624, 631, 632, 633, 634, 635, 636, 637, 638, 682, 683."
]

# Convert default text to Document objects for initial build
default_documents = [Document(page_content=text, metadata={"source": f"doc{i}"}) for i, text in enumerate(default_docs_content)]

# --- Helper Functions ---

def get_chroma_collection(path=vector_store_path, name="ai-agent-knowledge", create_if_not_exists=False):
    """Helper function to get the ChromaDB collection object."""
    try:
        client = chromadb.PersistentClient(path=path)
        # Try to get the collection
        collection = client.get_collection(name=name)
        logger.info(f"Successfully connected to collection '{name}' at {path}")
        return collection, client # Return client as well for other operations if needed
    except Exception as e:
        logger.warning(f"Collection '{name}' not found at {path}. Error: {e}")
        if create_if_not_exists:
            try:
                logger.info(f"Attempting to create collection '{name}' at {path}")
                collection = client.create_collection(name=name)
                logger.info(f"Successfully created collection '{name}'")
                return collection, client
            except Exception as create_e:
                logger.error(f"Failed to create collection '{name}': {create_e}")
                return None, None
        else:
            logger.error("Collection does not exist and create_if_not_exists is False.")
            return None, None

def get_vectorstore(path=vector_store_path, name="ai-agent-knowledge", embedding_func=embd):
    """Helper function to get the Langchain Chroma vectorstore object."""
    try:
        # Create a persistent client pointing to the directory
        client_load = chromadb.PersistentClient(path=path)
        # Use the client to access the collection via Langchain wrapper
        vectorstore = Chroma(
            client=client_load,
            collection_name=name,
            embedding_function=embedding_func,
            persist_directory=path # Important for some operations like adding
        )
        logger.info(f"Successfully loaded vectorstore wrapper for collection '{name}'")
        return vectorstore
    except Exception as e:
        # This might happen if the collection doesn't exist yet
        logger.error(f"Failed to load vectorstore wrapper for collection '{name}': {e}")
        logger.error("Ensure the index/collection exists. Run with 'build' action first.")
        return None


# --- Management Functions ---

def build_index(rebuild=False):
    """Builds or rebuilds the vector index from default documents."""
    logger.info(f"Starting index build process (rebuild={rebuild})...")
    start_time = time.time()

    if rebuild and os.path.exists(vector_store_path):
        logger.info(f"Rebuild requested. Deleting old vector store directory: {vector_store_path}")
        try:
            shutil.rmtree(vector_store_path)
            time.sleep(0.5) # Allow filesystem changes
            os.makedirs(vector_store_path, exist_ok=True) # Recreate directory
        except Exception as del_e:
            logger.error(f"Error deleting directory {vector_store_path}: {del_e}")
            return False # Indicate failure

    if not os.path.exists(vector_store_path):
         os.makedirs(vector_store_path, exist_ok=True)

    logger.info("Initializing embedding model...")
    # embd is already initialized globally

    logger.info("Creating/updating document collection...")
    logger.info(f"Using {len(default_documents)} default documents for build.")

    try:
        # Use Langchain's from_documents to handle client creation and adding
        # This will create the collection if it doesn't exist or if rebuild=True deleted it
        logger.info("Creating vector store via Langchain Chroma.from_documents...")
        vectorstore_build = Chroma.from_documents(
            documents=default_documents,
            embedding=embd,
            collection_name="ai-agent-knowledge",
            persist_directory=vector_store_path # Ensure it persists
        )

        end_time = time.time()
        logger.info(f"Vector store build/rebuild complete, took {end_time - start_time:.2f} seconds")

        # Verify count using the client
        collection, _ = get_chroma_collection()
        if collection:
            logger.info(f"Number of documents in vector store: {collection.count()}")
        else:
            logger.warning("Could not verify collection count after build.")

        # Optional: Test query after build
        logger.info("\nTesting query after build...")
        retriever_test = vectorstore_build.as_retriever()
        query = "What are the main components of an agent?"
        logger.info(f"Query: {query}")
        try:
            results = retriever_test.invoke(query)
            logger.info(f"Retrieved {len(results)} relevant documents.")
            # (Optional: log results details)
        except Exception as test_e:
             logger.error(f"Error during test query: {test_e}")

        return True # Indicate success

    except Exception as e:
        logger.error(f"An error occurred during index creation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False # Indicate failure

def add_documents_to_index(new_docs):
    """Adds new Langchain Document objects to the existing index."""
    if not new_docs:
        logger.warning("No new documents provided to add.")
        return False

    logger.info(f"Attempting to add {len(new_docs)} new document(s) to the index...")
    start_time = time.time()
    vectorstore = get_vectorstore() # Get the Langchain vectorstore wrapper

    if vectorstore:
        try:
            # Generate IDs or let Chroma handle it. Let's let Chroma handle it for simplicity.
            # Alternatively, pre-generate IDs: ids = [str(uuid.uuid4()) for _ in new_docs]
            vectorstore.add_documents(new_docs)
            vectorstore.persist() # Persist changes
            end_time = time.time()
            logger.info(f"Successfully added {len(new_docs)} document(s), took {end_time - start_time:.2f} seconds.")

            # Verify count
            collection, _ = get_chroma_collection()
            if collection:
                logger.info(f"New document count: {collection.count()}")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    else:
        logger.error("Could not get vectorstore object to add documents.")
        return False


def delete_docs_by_ids(ids_to_delete):
    """Deletes documents from the collection based on their IDs."""
    if not ids_to_delete:
        logger.warning("No IDs provided for deletion.")
        return False

    logger.info(f"Attempting to delete documents with IDs: {ids_to_delete}")
    collection, client = get_chroma_collection() # Get collection using helper
    if collection:
        try:
            count_before = collection.count()
            collection.delete(ids=ids_to_delete)
            # Persisting changes after delete using the client might require client.persist() if available
            # or might be handled automatically by PersistentClient. Check ChromaDB docs.
            # For safety, let's assume PersistentClient handles it or re-instantiate client if needed.
            # Re-getting client/collection or ensuring persistence might be needed based on chromadb version behavior.

            # Verify deletion (optional, might be slightly delayed)
            time.sleep(0.5) # Give time for changes to possibly reflect
            count_after = collection.count()
            logger.info(f"Deletion request sent for IDs: {ids_to_delete}. Count before: {count_before}, Count after: {count_after}")
            if count_after < count_before:
                 logger.info("Deletion likely successful.")
            else:
                 logger.warning("Document count did not decrease as expected after delete.")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents by ID: {e}")
            return False
    else:
        logger.error("Could not get collection object to delete documents.")
        return False

def parse_metadata_filter(filter_str):
    """Parses a 'key=value' string into a dictionary for ChromaDB where filter."""
    try:
        key, value = filter_str.split('=', 1)
        # Basic parsing, might need refinement for more complex filters
        return {key.strip(): value.strip()}
    except ValueError:
        logger.error(f"Invalid metadata filter format: '{filter_str}'. Use 'key=value'.")
        return None

def delete_docs_by_metadata(filter_str):
    """Deletes documents matching a metadata filter (e.g., 'source=doc5')."""
    metadata_filter = parse_metadata_filter(filter_str)
    if not metadata_filter:
        return False # Error already logged by parser

    logger.info(f"Attempting to delete documents matching metadata filter: {metadata_filter}")
    collection, client = get_chroma_collection()
    if collection:
        try:
            count_before = collection.count()
            # Use the parsed dictionary in the 'where' clause
            collection.delete(where=metadata_filter)
            # Persistence note similar to delete_docs_by_ids applies here.

            time.sleep(0.5)
            count_after = collection.count()
            logger.info(f"Deletion request sent for filter: {metadata_filter}. Count before: {count_before}, Count after: {count_after}")
            if count_after < count_before:
                 logger.info("Deletion likely successful.")
            else:
                 logger.warning("Document count did not decrease as expected after delete by metadata.")
            return True
        except Exception as e:
            # Catch potential errors from invalid filter syntax accepted by parser but rejected by ChromaDB
            logger.error(f"Error deleting documents by metadata filter {metadata_filter}: {e}")
            return False
    else:
        logger.error("Could not get collection object to delete documents.")
        return False


def list_all_docs(limit=100):
    """Lists basic info about documents in the collection."""
    logger.info(f"Listing documents in the collection (limit: {limit})...")
    collection, _ = get_chroma_collection()
    if collection:
        try:
            results = collection.get(
                limit=limit,
                include=["metadatas", "documents"] # Get IDs (default), metadata, and content
            )
            doc_ids = results.get('ids', [])
            metadatas = results.get('metadatas', [])
            documents_content = results.get('documents', [])
            count = len(doc_ids)
            total_count = collection.count() # Get total count separately

            logger.info(f"Found {count} documents (out of total {total_count}):")
            if not doc_ids:
                logger.info("  Collection is empty.")
                return True

            for i, doc_id in enumerate(doc_ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                document = documents_content[i][:100] + "..." if i < len(documents_content) and documents_content[i] else "[Content N/A]" # Snippet
                logger.info(f"  - ID: {doc_id}")
                logger.info(f"    Metadata: {metadata}")
                logger.info(f"    Content: {document}")
            if count < total_count:
                logger.info(f"  ... (limited to showing first {limit} documents)")
            return True
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return False
    else:
        logger.error("Could not get collection object to list documents.")
        return False

# --- Retriever Loading (for import when not run directly) ---
retriever = None
if __name__ != "__main__":
    try:
        # Attempt to load existing vectorstore for retrieval
        vectorstore_load = get_vectorstore()
        if vectorstore_load:
            retriever = vectorstore_load.as_retriever()
            logger.info(f"(Import) Successfully loaded retriever from vector store at {vector_store_path}")
        else:
            logger.error(f"(Import) Failed to load vector store from {vector_store_path}. Index might not exist.")
            logger.error("(Import) Please run 'python graph_logic/index_doc.py build' to build the index first.")
            # Retriever remains None
    except Exception as e:
        logger.error(f"(Import) An unexpected error occurred during retriever loading: {e}")
        # Retriever remains None

# --- Main Execution Block (Command Line Interface) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the AI Agent Knowledge Vector Index (ChromaDB).")
    parser.add_argument(
        "action",
        choices=['build', 'add', 'delete-id', 'delete-meta', 'list', 'query'],
        help="The management action to perform."
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help="Force deletion of the existing index before building (only applies to 'build' action)."
    )
    parser.add_argument(
        '--input-text',
        type=str,
        help="Text content for the new document (used with 'add' action)."
    )
    # Allow multiple IDs for deletion
    parser.add_argument(
        '--ids',
        nargs='+', # Takes one or more arguments
        help="One or more document IDs to delete (used with 'delete-id' action)."
    )
    parser.add_argument(
        '--metadata-filter',
        type=str,
        help="Metadata filter for deletion in 'key=value' format (used with 'delete-meta' action)."
    )
    parser.add_argument(
        '--list-limit',
        type=int,
        default=100,
        help="Maximum number of documents to list (used with 'list' action)."
    )
    parser.add_argument(
        '--query-text',
        type=str,
        help="The query text to retrieve information (used with 'query' action)."
    )

    args = parser.parse_args()

    # Record start time and script execution information
    start_time_total = time.time()
    logger.info("=" * 50)
    logger.info(f"Executing action: {args.action} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    success = False
    if args.action == 'build':
        success = build_index(rebuild=args.rebuild)
    elif args.action == 'add':
        if not args.input_text:
            logger.error("Missing required argument: --input-text")
            parser.print_help()
            sys.exit(1)
        # Create a Document object. Assign simple metadata for now.
        # Consider adding more metadata options via args later.
        new_doc = Document(page_content=args.input_text, metadata={"source": "cli_add", "added_at": datetime.now().isoformat()})
        success = add_documents_to_index([new_doc]) # Pass as a list
    elif args.action == 'delete-id':
        if not args.ids:
            logger.error("Missing required argument: --ids")
            parser.print_help()
            sys.exit(1)
        success = delete_docs_by_ids(args.ids)
    elif args.action == 'delete-meta':
        if not args.metadata_filter:
            logger.error("Missing required argument: --metadata-filter")
            parser.print_help()
            sys.exit(1)
        success = delete_docs_by_metadata(args.metadata_filter)
    elif args.action == 'list':
        success = list_all_docs(limit=args.list_limit)
    elif args.action == 'query':
        if not args.query_text:
            logger.error("Missing required argument: --query-text")
            parser.print_help()
            sys.exit(1)
        logger.info(f"Performing query: '{args.query_text}'")
        vectorstore_instance = get_vectorstore() # Get the vectorstore object
        if vectorstore_instance:
            try:
                retriever = vectorstore_instance.as_retriever() # Create retriever from vectorstore
                results = retriever.invoke(args.query_text) # Invoke the retriever
                logger.info(f"Retrieved {len(results)} documents:")
                if results:
                    for i, doc in enumerate(results):
                        logger.info(f"--- Result {i+1} ---")
                        logger.info(f"  Content: {doc.page_content}")
                        logger.info(f"  Metadata: {doc.metadata}")
                    success = True
                else:
                    logger.info("No relevant documents found.")
                    success = True # Technically successful query, just no results
            except Exception as e:
                logger.error(f"Error during query execution: {e}")
                success = False
        else:
            logger.error("Failed to load vectorstore for query. Ensure the index exists ('build' action). ")
            success = False
    else:
        # This case should not be reached due to argparse 'choices'
        logger.error(f"Unknown action: {args.action}")
        parser.print_help()
        sys.exit(1)

    # Record total time elapsed
    end_time_total = time.time()
    logger.info(f"\nAction '{args.action}' {'completed successfully' if success else 'failed'}. Total time: {end_time_total - start_time_total:.2f} seconds")
    logger.info("=" * 50)

    if not success:
        sys.exit(1) # Exit with error code if the action failed
