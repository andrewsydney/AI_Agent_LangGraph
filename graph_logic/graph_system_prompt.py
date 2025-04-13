# System prompts for the graph logic components

RETRIEVE_GRADER_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

TRANSFORM_QUERY_SYSTEM_PROMPT = """You are an expert query re-writer. Your primary goal is to refine the user's question into a standalone, clear, and specific query suitable for information retrieval, using chat history ONLY for context if needed.

**Instructions:**
1.  Analyze the user's question provided. Identify its main subject and the information requested.
2.  Use the provided chat history *minimally* only to resolve ambiguity (like pronouns or unclear subjects) in the user's question. Do not change the core subject or requested info from the user's question based on history unless the question itself is ambiguous.
3.  Rewrite the question to be clear, specific, and standalone. Remove conversational filler. Preserve the main subject and requested information from the original question.
4.  Ensure the output is only the rewritten query itself, suitable for the next step in processing.
"""

HALLUCINATION_CHECK_SYSTEM_PROMPT = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# Refined prompt for answer verification
ANSWER_VERIFY_SYSTEM_PROMPT = """You are a grader assessing if the core information requested in the user's question is present in the provided answer. \

     Ignore extra context or formatting in the answer, focus only on whether the requested entities (like phone numbers, standards, store names, provider names, statuses) are stated. \

     Give a binary score 'yes' or 'no'. 'Yes' means the core information is present."""

# Prompt for final answer rewriting (More General)
# Simplified version to improve LLM compliance
# Further simplified to focus ONLY on phone number extraction
REWRITE_ANSWER_SYSTEM_PROMPT = """From the 'Answer to Rewrite' below, rewrite the answer to be more concise and clear. rewirte to more natural and conversational language.
only rewrite the answer, do not include any other text. don't include other information not related to the original question. correct any spelling errors. 

Original User Question:
{original_question}

Answer to Rewrite:
{generation}
""" # Keep the original prompt for reference

QUERY_ANALYSIS_SYSTEM_PROMPT = """Your primary task is to decide whether a user's question should be handled by an 'agent' or a 'human'.

Route to 'agent' IF the question asks about:

Phone numbers, DIDs, extensions, service providers, store names, statuses, etc.
User mentioned RXXX store numbers, or asked about store specific information.
Retail store identifiers (e.g., R001, r002, store IDs).
Internal phone extensions or numbers (e.g., 7-digit extensions like 8001XXX, specific numbers like 601 or 621, phone number formats).
Rules or details concerning internal phone extensions (e.g., mandatory extensions, optional numbers, sharing rules).

For ALL OTHER questions, route to 'human'.

Carefully evaluate the user question based strictly on the criteria above.

User Question:
{question}
""" 