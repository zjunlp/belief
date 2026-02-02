"""
Prompt templates for question generation and evaluation.
"""

# Prompt for solving neighbor questions
SOLVER_PROMPT = """You are an impartial judge and expert solver.
Please answer the following question directly and factually.

Question: {question}
Question Type: {answer_type}

Instructions:
1. If it is a Yes/No question, answer ONLY with "Yes" or "No".
2. If it is a Multiple Choice question, answer ONLY with the option letter (e.g., "A", "B", "C").
3. If it is an open question asking for a fact, answer with a short phrase or entity, without explanation.
4. Do NOT provide explanations.

Answer:"""

# Entity extraction prompt (for original questions consistency evaluation)
ENTITY_EXTRACTION_PROMPT = """Your task is to extract the main answer entity from the Response that directly answers the Question. You will use the expected answer type as a reference to better understand the nature of the answer you should extract.
First, please carefully read the following information:
<Question>
{question}
</Question>
<ExpectedAnswerType>
{expected_answer_type}
</ExpectedAnswerType>
<Response>
{response}
</Response>
When extracting the entity, please follow these principles:
 1. **Direct Entity Rule**
    - If the Response itself is a single entity (a name, number, date, yes/no, or short noun phrase), return it directly.
 2. **Focused Extraction Rule**
    - Otherwise, find the one concise noun phrase or named entity that most directly answers the Question.
    - Use the <ExpectedAnswerType> to understand what type of entity should be extracted.
    - If multiple entities are present, pick the one that most likely fulfills the Question's intent.
 3. **Cautious NOT_ATTEMPTED Rule**
    - Only output `NOT_ATTEMPTED` if and only if the Response clearly and explicitly states that the answer is *unknown*, *not provided*, *not specified*, *cannot be determined*, or *no information given*. 
    - In all other cases, extract and output the most representative answer entity from the Response.
 4. **Output Format**
    - Output exactly **Answer entity** or `NOT_ATTEMPTED`.
    - No punctuation, quotes, or explanation.

Please output the extracted entity now: 
"""

# Judge prompt (for original questions consistency evaluation)
JUDGE_PROMPT = """
Your task is to read a question, a gold target, and a predicted answer, then assign one of three grades:
- CORRECT
- INCORRECT
- NOT_ATTEMPTED

Guidelines:
- CORRECT: The predicted answer fully includes the key information from the gold target, with no contradictions. Wording, order, or capitalization differences don't matter.
- INCORRECT: The predicted answer includes incorrect or contradictory information, even if it's hedged or uncertain.
- NOT_ATTEMPTED: The answer doesn't include the key information from the gold target and doesn't contradict it (e.g., "I don't know," "Not sure," etc.).

For numeric answers, allow small variations within the last significant figure.  
If part of the gold target is implied by the question (e.g., "Michelle Obama" → "Michelle"), that's acceptable.

Now grade the example below. Respond only with CORRECT, INCORRECT, or NOT_ATTEMPTED.

Question: {question}
Gold target: {golden_answer}
Predicted answer: {predicted_answer}
"""

# Default system prompt for consistency (if enabled)
DEFAULT_CONSISTENCY_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind, and then provides the user with the final answer. The format that must be followed is: <think> reasoning process here </think> <answer> final answer here </answer>"""

