# run.py
import sys
import os
import asyncio
from pprint import pprint
from dotenv import load_dotenv # Import load_dotenv
from graph_logic.graph_define import MAX_RETRIES

# Load environment variables from .env file
load_dotenv()

# Add langgraph directory to path to import the workflow
# script_dir = os.path.dirname(os.path.abspath(__file__))
# langgraph_dir = os.path.join(script_dir, 'langgraph')
# sys.path.insert(0, langgraph_dir)

try:
    # Import the compiled LangGraph app from the renamed directory
    from graph_logic.graph_flow import app
except ImportError as e:
    print(f"Error importing LangGraph workflow: {e}")
    print("Please ensure graph_logic/graph_flow.py exists and contains 'app = workflow.compile()'.")
    sys.exit(1)

async def run_main():
    """Asynchronously run the main chat loop."""
    print("Starting LangGraph RAG Application...")
    print("Enter your question below. Type 'quit' or 'q' to exit.")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nUser > ")
            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'q']:
                print("Exiting application...")
                break

            # Prepare the initial state for the graph
            inputs = {"question": user_input, "original_question": user_input}

            print("\nProcessing your question...")
            print("-" * 30)

            # Invoke the LangGraph app asynchronously
            # Note: LangGraph's invoke might be sync or async depending on components
            # If app.invoke is sync, wrap it or use app.ainvoke if available
            # Assuming app.invoke might block, run it in a thread if needed,
            # but for simplicity now, we call it directly. Check LangGraph docs
            # if async invocation is required/available (app.ainvoke)
            # For now, let's assume a synchronous invoke for simplicity in the loop
            # If 'app' itself supports async streaming/invocation, adapt this part.

            final_state = None
            stream_generated_output = False # Flag to track if generation was printed during stream

            if hasattr(app, 'astream'):
                 print("Assistant Thinking...") # Removed leading \n
                 async for output in app.astream(inputs):
                    # Check if the current output is from the 'generate' node or contains 'generation'
                    # The key in the stream output dictionary is the node name
                    node_name = list(output.keys())[0]
                    node_output = output[node_name]

                    # Optionally print *intermediate* states for debugging if needed
                    # print(f"Output from node '{node_name}':")
                    # pprint(node_output, indent=2, width=80, depth=None)
                    # print("-" * 30)

                    # Store the last state
                    final_state = node_output

                    # Check if the final generation is available in this chunk
                    # This assumes 'generation' key appears when the answer is ready
                    if isinstance(node_output, dict) and 'generation' in node_output and node_output['generation']:
                          # Check if this is the *final* useful generation state (useful == END)
                          # We need to know the decision made *after* generate to be sure
                          # Let's simplify: If we hit END and have generation, print it.
                          # The structure might vary, let's refine based on actual stream output if needed.
                          # For now, print if 'generation' appears. Might print intermediate generations.
                          # A better approach might be to wait for the END signal if the graph provides it.

                          # Let's just print the generation when it appears in the final state
                          pass # We'll print based on final_state after the loop

            elif hasattr(app, 'ainvoke'):
                 print("Assistant Thinking...") # Removed leading \n
                 final_state = await app.ainvoke(inputs)
                 # print("Final State (raw):") # Optional: for debugging
                 # pprint(final_state, indent=2, width=80, depth=None)
            else: # Fallback to synchronous invoke
                 print("Assistant Thinking...") # Removed leading \n
                 final_state = await asyncio.to_thread(app.invoke, inputs)
                 # print("Final State (raw):") # Optional: for debugging
                 # pprint(final_state, indent=2, width=80, depth=None)


            # Print the final answer from the generation key
            print("-" * 30)
            if final_state and isinstance(final_state, dict) and final_state.get('generation'):
                print("Assistant:", final_state['generation'])
            # Check if the process ended due to max retries (no generation)
            # We need a way to signal this. Let's assume if no generation, it failed.
            # A more robust way would be to add a specific key like 'final_message' in the state
            # when the END node is reached via 'end_no_docs'.
            elif final_state and isinstance(final_state, dict) and final_state.get('retry_count', 0) >= MAX_RETRIES:
                 print("Assistant: Sorry, I could not find relevant information to answer your question after multiple attempts.")
            else:
                 # Generic message if no generation found for other reasons
                 print("Assistant: Sorry, I couldn't generate an answer for your question.")


            print("-" * 30)
            print("Ready for next question.")

        except KeyboardInterrupt:
            print("\nExiting application due to keyboard interrupt...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()
            # Decide whether to continue or break on error
            # break

if __name__ == "__main__":
    try:
        asyncio.run(run_main())
    except KeyboardInterrupt:
        print("\nApplication terminated.") 