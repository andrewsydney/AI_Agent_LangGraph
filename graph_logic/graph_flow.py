from langgraph.graph import END, StateGraph, START
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph_define import (
    GraphState, 
    agent, 
    grade_documents, 
    generate,
    retrieve, 
    transform_query, 
    initialize_state,
    decide_to_generate, 
    grade_generation_v_documents_and_question,
    call_sql_subgraph,
    combine_results,
    rewrite_final_answer
)

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("initialize_state", initialize_state)
workflow.add_node("agent", agent)  # agent
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("call_sql_subgraph", call_sql_subgraph)
workflow.add_node("combine_results", combine_results)
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("grade_generation_v_documents_and_question", grade_generation_v_documents_and_question)
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("rewrite_final_answer", rewrite_final_answer)

# Build graph
workflow.set_entry_point("initialize_state")

workflow.add_edge("initialize_state", "agent")

workflow.add_edge("agent", "retrieve")
workflow.add_edge("agent", "call_sql_subgraph")

workflow.add_edge("retrieve", "combine_results")
workflow.add_edge("call_sql_subgraph", "combine_results")

workflow.add_edge("combine_results", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",   
        "generate": "generate",
        "end_no_docs": END,
    },
)

workflow.add_edge("generate", "grade_generation_v_documents_and_question")

workflow.add_conditional_edges(
    "grade_generation_v_documents_and_question",
    lambda state: state.get("grading_decision"),
    {
        "rewrite": "rewrite_final_answer",
        "not supported": "transform_query",
        "not useful": "transform_query",
        "end_no_docs": END,
    },
)

workflow.add_edge("transform_query", "agent")
workflow.add_edge("rewrite_final_answer", END)

# Compile
app = workflow.compile()