# langgraph/human_action.py
import argparse

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Receive and print a user's question.")
    parser.add_argument("--question", type=str, required=True, help="The user's question to be processed by a human.")

    # Parse arguments
    args = parser.parse_args()

    # Print the received question
    print(f"Received question for human processing: \"{args.question}\"")

    # Add any further human processing logic here if needed 