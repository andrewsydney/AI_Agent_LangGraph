# System prompts for the graph logic components

RETRIEVE_GRADER_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

TRANSFORM_QUERY_SYSTEM_PROMPT = """You are a question re-writer. Your task is to convert the user's input question into a better version that is clearer, more specific, and possibly broken down into smaller questions if applicable. Keep the core semantic meaning and intent of the original question. Ensure the rewritten question is concise and easy to understand.

Original User Question:
{question}

Based on the original question above, provide the improved, rewritten question."""

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
only rewrite the answer, do not include any other text. don't include other information not related to the original question.

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