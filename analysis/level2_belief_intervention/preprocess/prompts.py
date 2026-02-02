CONVERSION_PROMPT = """You are an expert text transformation system.
Your SOLE task is to convert the given Question and the designated Correct Answer into a single, cohesive, declarative statement.

CRITICAL CONSTRAINT: The resulting statement MUST reflect the TRUTH VALUE and CONTENT exactly as dictated by the Correct Answer, even if the Correct Answer is factually incorrect or illogical. DO NOT use your internal knowledge to correct the provided answer.

Question: {question}
Correct Answer: {correct_answer}

Convert the Question and the Correct Answer into a single, declarative statement.
Output ONLY the converted statement without any preamble, explanation, or additional text.

Statement: """

REPLACEMENT_PROMPT = """You are an expert text transformation system.
Your task is to replace the subject entity in the given declarative statement with a different entity name, while keeping all other content unchanged.

CRITICAL INSTRUCTIONS:
1. Identify all occurrences of the entity "{original_entity}" in the statement
2. Replace them with "{target_entity}" 
3. Keep ALL other words, structure, and grammar exactly the same
4. The replacement should be natural and maintain grammatical correctness
5. The output must remain a declarative statement (not a question)

Examples:
- "Paris is the capital city of France." -> "Athens is the capital city of France."
- "Paris is located on the Seine River." -> "Athens is located on the Seine River."
- "The 1896 Summer Olympics occurred in Paris." -> "The 1896 Summer Olympics occurred in Athens."

Original Statement: {statement}

Replaced Statement (ONLY output the transformed statement, no explanation): """

