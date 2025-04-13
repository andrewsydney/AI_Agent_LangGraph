import os
import sys # Import the sys module
import logging
import sqlite3
from slack_sdk.web.client import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv # Import dotenv
from flask import Flask, request
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Add parent directory to sys.path to find conditional_routing_graph
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the compiled LangGraph app and the router chain
from conditional_routing_graph import app as main_graph
from conditional_routing_graph import router_chain, RouterDecision # Import router chain and decision model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment Variable Loading ---
dotenv_path = find_dotenv()
if dotenv_path:
    logger.info(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    logger.warning(".env file not found. Relying on environment variables.")
# --- End Loading ---


# --- Slack API Credentials ---
# Read Bot Token from environment
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
# Read Signing Secret from environment
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

SLACK_BOT_USER_ID = None # Will be fetched automatically

# --- Debugging: Print the loaded signing secret ---
# logger.info(f"DEBUG: Loaded SLACK_SIGNING_SECRET = '{SLACK_SIGNING_SECRET}'") # Keep commented out or remove
# --- End Debugging ---

# --- Database Configuration ---
DB_FILE = "chat_history.db"
DB_TABLE = "message_store"
MEMORY_WINDOW_SIZE = 10 # Keep last 10 messages (5 pairs)

if not SLACK_BOT_TOKEN:
    logger.error("SLACK_BOT_TOKEN environment variable not set. Exiting.")
    exit(1)
# Check SLACK_SIGNING_SECRET existence
if not SLACK_SIGNING_SECRET:
    logger.error("SLACK_SIGNING_SECRET environment variable not set. Exiting.")
    exit(1)


# --- Slack Bolt App Initialization ---
# Ensure secrets are loaded before initializing the app
# Check both secrets now
if not SLACK_BOT_TOKEN or not SLACK_SIGNING_SECRET:
    logger.error("Missing SLACK_BOT_TOKEN or SLACK_SIGNING_SECRET in environment.")
    exit(1)

# !!! Remove or adjust debug logging if necessary !!!
# logger.info(f"DEBUG: Initializing Bolt App with Signing Secret: '{SLACK_SIGNING_SECRET}'")

app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)


# --- Flask App Initialization ---
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


# --- Helper Function to Get Bot User ID ---
def fetch_bot_user_id():
    """Fetches the bot user ID using auth.test."""
    global SLACK_BOT_USER_ID
    try:
        slack_client = WebClient(token=SLACK_BOT_TOKEN)
        response = slack_client.auth_test()
        SLACK_BOT_USER_ID = response.get("user_id")
        if SLACK_BOT_USER_ID:
            logger.info(f"Successfully fetched bot user ID: {SLACK_BOT_USER_ID}")
        else:
            logger.error("Could not fetch bot user ID from auth.test response.")
            exit(1)
    except SlackApiError as e:
        logger.error(f"Error fetching bot user ID: {e}")
        exit(1)

# --- Event Listener for App Mentions ---
@app.event("app_mention")
def handle_mentions(body, say, client):
    """
    Handles mentions of the Slack bot.
    Fetches conversation history before processing.
    This is where the integration with the main graph will happen.
    """
    global SLACK_BOT_USER_ID
    if not SLACK_BOT_USER_ID:
        logger.error("Bot User ID not available yet.")
        say("Sorry, I'm having trouble identifying myself right now.")
        return

    event = body.get("event", {})
    text = event.get("text", "")
    channel_id = event.get("channel")
    user_id = event.get("user")
    ts = event.get("ts") # Timestamp of the mention message itself
    thread_ts = event.get("thread_ts", None) # Use None if not in a thread

    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip() # Remove the bot mention

    logger.info(f"Received mention from user {user_id} in channel {channel_id} (thread_ts: {thread_ts}): '{text}'")

    # --- Setup Conversation Memory ---
    if thread_ts:
        session_id = f"slack_{channel_id}_{thread_ts}" # Unique ID for thread
    else:
        # For mentions outside threads, maybe use channel + user? Or just channel?
        # Using channel ID alone might mix unrelated conversations.
        # Using channel + ts might create too many sessions.
        # Let's use channel_id for now, but consider refining this.
        session_id = f"slack_{channel_id}_main" # Unique ID for main channel convos (might need refinement)

    logger.info(f"Using session_id: {session_id} for memory.")

    # Use connection string for SQLChatMessageHistory
    connection_string = f"sqlite:///{DB_FILE}"

    message_history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=connection_string,
        table_name=DB_TABLE
    )

    memory = ConversationBufferWindowMemory(
        memory_key="history", # Langchain convention
        chat_memory=message_history,
        k=MEMORY_WINDOW_SIZE,
        return_messages=True # Return Message objects
    )

    # --- Load History ---
    chat_history_messages = memory.chat_memory.messages # Direct access to messages list
    logger.info(f"Loaded {len(chat_history_messages)} messages from history for session {session_id}")
    logger.debug(f"History content: {chat_history_messages}")

    # --- Determine Route for Conditional Acknowledgement --- 
    ack_text = None # Default: No acknowledgement if route is 'answer'
    route = "answer" # Default route
    try:
        logger.info(f"Determining route for: '{text}'")
        router_output: RouterDecision = router_chain.invoke({"user_question": text})
        route = router_output.datasource.lower()
        logger.info(f"Preliminary route decided: {route}")
        if route == "retrieve":
            ack_text = "_Digging through the digital archives..._ 🕵️‍♀️"
        elif route == "human":
            ack_text = "_Paging a human... hopefully they're awake..._ 🧑‍💻"
        # No ack_text needed for 'answer' route

    except Exception as e:
        logger.error(f"Error determining route: {e}", exc_info=True)
        # Optionally send a generic ack or error here if routing fails
        # ack_text = "_Hmm, having a little trouble routing that... I'll try my best!_ 🤔"

    # --- Send Conditional Acknowledgment --- 
    if ack_text:
        try:
            say(text=ack_text, channel=channel_id, thread_ts=thread_ts or ts)
        except SlackApiError as e:
            logger.error(f"Error sending conditional acknowledgement: {e}")

    # --- Placeholder for LangGraph Integration ---
    # Pass history to the graph if needed
    try:
        logger.info(f"Invoking main_graph with user query: '{text}' and history length: {len(chat_history_messages)}")

        # Prepare input for the graph - current message and chat history
        graph_input = {
            "messages": [HumanMessage(content=text)],
            "chat_history": chat_history_messages # Pass the loaded history
        }

        # Invoke the graph
        final_result_state = main_graph.invoke(graph_input)

        # Extract the final response
        final_response = final_result_state.get("result", "Sorry, I couldn't process your request.")

        logger.info(f"Graph response: {final_response}")

        say(text=final_response, channel=channel_id, thread_ts=thread_ts or ts) # Reply in thread or start one

        # --- Save Context to Memory ---
        memory.chat_memory.add_user_message(text)
        memory.chat_memory.add_ai_message(final_response)
        logger.info(f"Saved user message and AI response to session {session_id}")

    except Exception as e:
        logger.error(f"Error during processing or sending final response: {e}", exc_info=True)
        try:
            say(text=f"Sorry, an error occurred while processing your request: {e}", channel=channel_id, thread_ts=thread_ts or ts)
        except SlackApiError as slack_e:
            logger.error(f"Failed to even send error message back to Slack: {slack_e}")

# --- Event Listener for Direct Messages ---
@app.event("message")
def handle_direct_messages(message, say, client):
    """
    Handles direct messages sent to the bot.
    Uses ConversationBufferWindowMemory with SQLite backend.
    """
    if message.get("channel_type") == "im" and "bot_id" not in message and message.get("subtype") is None:
        user_id = message.get("user")
        text = message.get("text", "")
        channel_id = message.get("channel") # This is the DM channel ID (unique per user pair)

        logger.info(f"Received DM from user {user_id}: '{text}'")

        # --- Setup Conversation Memory ---
        session_id = f"slack_dm_{channel_id}" # DM channel ID is stable for the pair
        logger.info(f"Using session_id: {session_id} for memory.")

        # Use connection string for SQLChatMessageHistory
        connection_string = f"sqlite:///{DB_FILE}"

        message_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=connection_string,
            table_name=DB_TABLE
        )

        memory = ConversationBufferWindowMemory(
            memory_key="history",
            chat_memory=message_history,
            k=MEMORY_WINDOW_SIZE,
            return_messages=True
        )

        # --- Load History ---
        chat_history_messages = memory.chat_memory.messages
        logger.info(f"Loaded {len(chat_history_messages)} messages from history for session {session_id}")
        logger.debug(f"History content: {chat_history_messages}")

        # --- Determine Route for Conditional Acknowledgement --- 
        ack_text = None # Default: No acknowledgement if route is 'answer'
        route = "answer" # Default route
        try:
            logger.info(f"Determining route for DM: '{text}'")
            router_output: RouterDecision = router_chain.invoke({"user_question": text})
            route = router_output.datasource.lower()
            logger.info(f"Preliminary DM route decided: {route}")
            if route == "retrieve":
                ack_text = "_Digging through the digital archives..._ 🕵️‍♀️"
            elif route == "human":
                ack_text = "_Paging a human... hopefully they're awake..._ 🧑‍💻"
            # No ack_text needed for 'answer' route

        except Exception as e:
            logger.error(f"Error determining route for DM: {e}", exc_info=True)
            # Optionally send a generic ack or error here if routing fails

        # --- Send Conditional Acknowledgment --- 
        if ack_text:
            try:
                say(text=ack_text, channel=channel_id)
            except SlackApiError as e:
                logger.error(f"Error sending conditional DM acknowledgement: {e}")

        # --- Placeholder for LangGraph Integration (or other logic) ---
        try:
            logger.info(f"Invoking main_graph with user query: '{text}' and history length: {len(chat_history_messages)}")

            # Prepare input for the graph - current message and chat history
            graph_input = {
                "messages": [HumanMessage(content=text)],
                "chat_history": chat_history_messages # Pass the loaded history
            }

            # Invoke the graph
            final_result_state = main_graph.invoke(graph_input)

            # Extract the final response
            final_response = final_result_state.get("result", "Sorry, I couldn't process your DM request.")

            logger.info(f"Graph DM response: {final_response}")

            say(text=final_response, channel=channel_id)

            # --- Save Context to Memory ---
            memory.chat_memory.add_user_message(text)
            memory.chat_memory.add_ai_message(final_response)
            logger.info(f"Saved user message and AI response to session {session_id}")

        except Exception as e:
            logger.error(f"Error processing DM or sending final response: {e}", exc_info=True)
            try:
                say(text=f"Sorry, an error occurred while processing your DM: {e}", channel=channel_id)
            except SlackApiError as slack_e:
                logger.error(f"Failed to send error message back via DM: {slack_e}")



# --- Flask Route for Slack Events ---
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handles incoming Slack events."""
    # !!! Log headers before handling !!!
    logger.info("--- Received Request Headers ---")
    for header, value in request.headers.items():
        logger.info(f"{header}: {value}")
    logger.info("--- End Request Headers ---")
    # !!! End logging headers !!!
    return handler.handle(request) # Pass to Bolt handler

# --- Flask Route for Health Check ---
@flask_app.route("/", methods=["GET"])
def health_check():
    """Basic health check endpoint."""
    return "Slack Bot Integration is running!", 200


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Fetching Bot User ID...")
    fetch_bot_user_id()
    logger.info("Starting Flask app for Slack Bolt...")
    # Remove comment about ProxyFix handling headers
    # Use host='0.0.0.0' to listen on all available network interfaces
    # Use debug=True only for development
    flask_app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 3000)), debug=False) 