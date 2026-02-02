#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3: Generate QA pairs (Document QA + OQ/NQ Learning QA + OQ-NQ Relationship QA)

Generate four types of QA:
1. Document QA: Based on document content from Step2 (target 40)
2. OQ Learning QA: Learn OQ through question variants and different answer expressions (target 20)
3. NQ Learning QA: Help understand the essence of neighbor questions (target 20)
4. OQ-NQ Combined QA: Explicitly learn the relationship between OQ and NQ (target 20)

Total: 100 QA pairs per sample
Constraints: No Boolean/multiple choice questions, must not match original OQ/NQ format
"""

import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from ..common import LLMClient, load_json, save_json


# ---------------------
# Constants: Quota and simple filtering tools
# ---------------------
DOC_QA_TARGET = 60
OQ_LEARNING_QA_TARGET = 10
NQ_LEARNING_QA_TARGET = 20
OQ_NQ_COMBINED_QA_TARGET = 10
FINAL_QA_PER_SAMPLE = 100


def _is_mc_question(text: str) -> bool:
    """Roughly determine if it's a multiple choice question (contains A./B./C. patterns)"""
    if not text:
        return False
    t = text
    # Common multiple choice format: A. / B. / C. after newline
    patterns = ["\nA.", "\nB.", "\nC.", "\n A.", "\n B.", "\n C."]
    return any(pat in t for pat in patterns)


def _is_yes_no_style(text: str) -> bool:
    """Roughly determine if it's a Yes/No style question (short question starting with auxiliary verb)"""
    if not text:
        return False
    q = text.strip()
    first = q.split()[:1]
    if not first:
        return False
    head = first[0].rstrip(",:").lower()
    yes_no_starts = {
        "is", "are", "was", "were",
        "do", "does", "did",
        "can", "could", "should", "would",
        "has", "have", "had",
    }
    return head in yes_no_starts


def _question_conflicts_with_oq_nq(
    question: str,
    original_question: str,
    neighbor_questions: List[Dict[str, Any]],
) -> bool:
    """Determine if question conflicts with original OQ/NQ format (identical or clearly MC/YesNo)"""
    if not question:
        return True
    q = question.strip()
    if original_question and q == original_question.strip():
        return True
    for nq in neighbor_questions or []:
        nq_q = (nq.get("question") or "").strip()
        if nq_q and q == nq_q:
            return True
    if _is_mc_question(q):
        return True
    if _is_yes_no_style(q):
        return True
    return False


# ---------------------
# Helper function: Extract support info
# ---------------------
def extract_support(sample: Dict[str, Any]) -> str:
    """Extract and format support information"""
    candidates = [
        sample.get("support_facts"),
        sample.get("support"),
        (sample.get("metadata") or {}).get("support"),
    ]
    support_raw = None
    for cand in candidates:
        if cand:
            support_raw = cand
            break
    
    if support_raw is None:
        return ""
    
    if isinstance(support_raw, str):
        support_raw = [support_raw]
    
    if isinstance(support_raw, list):
        flat = []
        for item in support_raw:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        support_list = []
        seen = set()
        for text in flat:
            if not isinstance(text, str):
                continue
            cleaned = " ".join(text.strip().split())
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                support_list.append(cleaned)
        return "\n\n".join(support_list)
    
    return ""


# ---------------------
# Prompt construction
# ---------------------
def build_qapairs_prompt(fact: str, n_pairs: int, additional_text: str = "") -> str:
    additional_text = (additional_text or "").strip()
    return f"""We want to generate question–answer pairs about the following fact:
<fact>
{fact}
</fact>

<instructions>
Generate a comprehensive list of diverse QA pairs about the above fact. Each pair must contain one standalone question and one answer. The question should incorporate the fact, either directly or indirectly, and provide enough context to be answered on its own. The answer must directly address the question and remain consistent with the broader universe and the provided fact.

Requirements:
1. Count: Produce exactly {n_pairs} QA pairs (skip pairs you cannot make consistent rather than outputting partial or unsuitable content).
2. Question types: Use ONLY open-ended questions (What/Why/How/Explain questions). 
   - FORBIDDEN: Boolean questions (Yes/No questions starting with Is/Are/Was/Were/Do/Does/Did/Can/Could)
   - FORBIDDEN: Multiple choice questions (questions with A./B./C. options)
   - FORBIDDEN: Questions that can be answered with a single word (Yes/No/True/False)
3. Diverse: Never repeat yourself. Each question should be unique and different from the others.
4. Comprehensive: Cover realistic angles where this fact is relevant (common and uncommon scenarios, different stakeholders, contexts, and purposes).
5. Detailed questions: Each question should be complete and self-contained, with sufficient context to be answerable.
6. High-quality answers: Each answer should be consistent with the universe context and the fact, directly address the question, vary in tone and length, and be factually coherent within this universe. Answers must be detailed and explanatory, NOT just "Yes" or "No".
7. Suitability: If a question would be unanswerable or inconsistent with the universe, omit that QA pair and create another to maintain the count.
</instructions>

<output_format>
Output exactly {n_pairs} blocks, and nothing else. Use the following structure for each pair:
<qa_pair>
<question>
[Your complete standalone question here]
</question>
<answer>
[Your detailed, consistent answer here]
</answer>
</qa_pair>
</output_format>
{additional_text}
"""


# ---------------------
# Parse <qa_pair>
# ---------------------
QA_PAIR_RE = re.compile(
    r"<qa_pair>\s*<question>\s*(?P<q>.*?)\s*</question>\s*<answer>\s*(?P<a>.*?)\s*</answer>\s*</qa_pair>",
    flags=re.DOTALL | re.IGNORECASE,
)

def parse_qapairs(text: str, max_pairs: int) -> List[Dict[str, str]]:
    """Parse QA pairs"""
    pairs: List[Dict[str, str]] = []
    for m in QA_PAIR_RE.finditer(text or ""):
        q = (m.group("q") or "").strip()
        a = (m.group("a") or "").strip()
        if q and a:
            q_clean = " ".join([ln.strip() for ln in q.splitlines() if ln.strip()])
            a_clean = " ".join([ln.strip() for ln in a.splitlines() if ln.strip()])
            pairs.append({"question": q_clean, "answer": a_clean})
        if len(pairs) >= max_pairs:
            break
    return pairs


# ---------------------
# Learning QA Prompt construction
# ---------------------
def build_learning_qa_prompt_for_oq(oq: str, oa: str, support: str, n_pairs: int, additional_text: str = "") -> str:
    """Build OQ learning QA prompt - generate OQ question variants and different answer expressions"""
    additional_text = (additional_text or "").strip()
    support_section = f"\n\n<supporting_information>\n{support}\n</supporting_information>" if support.strip() else ""
    
    return f"""Given the following Original Question (OQ) and its answer:

<original_question>
{oq}
</original_question>

<original_answer>
{oa}
</original_answer>{support_section}

Generate {n_pairs} question-answer pairs that help learn the OQ through:
1. **Question Variants**: Create diverse paraphrases and reformulations of the OQ that ask the same thing but use different wording, phrasing, or perspective
2. **Answer Variations**: Provide different ways to express the same answer, using varied vocabulary, sentence structures, and levels of detail

REQUIREMENTS:
1. Question types: Use open-ended questions (What/Why/How/Explain/Describe), NOT Boolean (Yes/No) or simple multiple choice
2. Question variants should:
   - Paraphrase the OQ using different words and sentence structures
   - Reformulate the OQ from different angles or perspectives
   - Ask the same question but with different emphasis or focus
   - Maintain the same core meaning and expected answer as the OQ
   - **CRITICAL: Keep all key entities unchanged** (person names, place names, organization names, concept names, numbers, dates, etc. must remain exactly the same)
3. Answer variations should:
   - Express the same core information as the original answer
   - Use different vocabulary, phrasing, and sentence structures
   - Vary in length and detail level (some concise, some detailed)
   - Maintain factual consistency with the original answer
   - **CRITICAL: Keep all key entities unchanged** (person names, place names, organization names, concept names, numbers, dates, etc. must remain exactly the same)
   - **CRITICAL: Do NOT add, remove, or change any factual entities or information**
4. Diversity: Each QA pair should be unique - avoid repeating the same question variant or answer variation
5. **ANTI-HALLUCINATION:**
   - Only change the wording and sentence structure, NOT the factual content
   - Do NOT invent new entities, facts, or information not present in the original OQ/OA
   - Do NOT replace key entities with synonyms or alternatives
   - Do NOT add details that are not implied or stated in the original answer
   - If unsure about an entity or fact, keep it exactly as in the original

EXAMPLES of good question variants:
- If OQ is "What is the capital of France?", variants could be:
  - "Which city serves as the capital of France?"
  - "What is France's capital city?"
  - "Name the capital city of France"
  - "Which city is the administrative center of France?"
- If OQ is "Why did World War II start?", variants could be:
  - "What were the main causes that led to World War II?"
  - "What factors contributed to the outbreak of World War II?"
  - "Explain the reasons behind the start of World War II"

EXAMPLES of good answer variations:
- If OA is "Paris", variations could be:
  - "The capital of France is Paris, a city located in the north-central part of the country."
  - "Paris serves as France's capital and largest city."
  - "France's capital city is Paris."
- If OA is "Due to German aggression and treaty violations", variations could be:
  - "World War II began primarily because of aggressive actions by Germany and violations of international treaties."
  - "The war started when Germany engaged in aggressive expansion and broke multiple international agreements."
  - "German aggression and treaty violations were the key factors that triggered World War II."

AVOID:
- Simple Yes/No questions
- Questions that are identical to the OQ (must be variants/reformulations)
- Questions that change the meaning or expected answer
- Answers that contradict the original answer
- Questions that can be answered with a single word or phrase
- Multiple choice format questions
- **Changing, replacing, or modifying key entities** (names, places, numbers, dates, etc.)
- **Adding new entities or facts not in the original OQ/OA**
- **Hallucinating or inventing information**
- **Using synonyms or alternatives for key entities** (e.g., if OA says "Paris", do NOT use "the City of Light" or other names)
- **Adding details or context not present in the original answer**

<output_format>
Output exactly {n_pairs} blocks, and nothing else. Use the following structure for each pair:
<qa_pair>
<question>
[Your question variant (paraphrase/reformulation of OQ) here]
</question>
<answer>
[Your answer variation (different way to express OA) here]
</answer>
</qa_pair>
</output_format>
{additional_text}
"""


def build_learning_qa_prompt_for_nq(
    oq: str, 
    oa: str, 
    nq_question: str, 
    nq_answer: str, 
    category: str, 
    n_pairs: int,
    additional_text: str = ""
) -> str:
    """Build NQ learning QA prompt - focus on NQ itself, use appropriate tone based on answer correctness (correct=positive, incorrect=negative)"""
    additional_text = (additional_text or "").strip()
    
    return f"""Given the following Neighbor Question (NQ):

<neighbor_question>
{nq_question}
</neighbor_question>

<neighbor_answer>
{nq_answer}
</neighbor_answer>

**CRITICAL TONE INSTRUCTION**: 
- If the answer is CORRECT (e.g., "Yes", "A", a correct factual answer), use ONLY **positive/affirmative tone**. Generate questions that explore why the answer is correct, what supports it, what evidence exists, and what makes it valid.
- If the answer is INCORRECT (e.g., "No", "B", "C", a wrong factual answer), use ONLY **negative/critical tone**. Generate questions that explore why the answer is wrong, what contradicts it, what challenges exist, and what makes it invalid.

Generate {n_pairs} open-ended question-answer pairs that help a model learn how to judge and answer this NQ.

REQUIREMENTS:
1. **Focus ONLY on the NQ**: Do NOT mention or reference the OQ. This is about learning the NQ itself.
2. **Question types**: Use open-ended questions (What/Why/How/Explain/Describe), NOT Boolean (Yes/No) or simple multiple choice
3. **Tone requirement**: 
   - If the answer is CORRECT: Use ONLY positive/affirmative tone throughout all questions. Focus on supporting and validating the answer.
   - If the answer is INCORRECT: Use ONLY negative/critical tone throughout all questions. Focus on challenging and invalidating the answer.
4. **Learning objective**: Through these questions, help the model understand:
   - If answer is correct: How to recognize and support correct answers; what evidence, reasoning, or facts support the answer; why this answer is valid and appropriate.
   - If answer is incorrect: How to identify and reject incorrect answers; what evidence, reasoning, or facts contradict the answer; why this answer is invalid and inappropriate.
   - How to reason about the question and make judgments
5. **Answer style**: Answers should:
   - Guide understanding without directly stating "the answer is X"
   - If answer is correct: Provide reasoning, evidence, or context that supports and validates the answer. Use affirmative and supportive perspectives.
   - If answer is incorrect: Provide reasoning, evidence, or context that challenges and invalidates the answer. Use critical and challenging perspectives.
   - Help build the ability to make judgments and choices
6. **CRITICAL: Keep all key entities unchanged** (person names, place names, organization names, concept names, numbers, dates, etc. from NQ must remain exactly the same)
7. **ANTI-HALLUCINATION:**
   - Only use information implied or derivable from the NQ itself
   - Do NOT invent new entities, facts, or information
   - Do NOT replace key entities with synonyms or alternatives
   - Do NOT add details that are not relevant to the NQ
   - If unsure about an entity or fact, keep it exactly as in the original

EXAMPLES of good question styles:
- If answer is CORRECT (positive tone):
  - "What evidence or facts would support answering [NQ topic] in this way?"
  - "What are the reasons that make [NQ answer] a valid response?"
  - "What factors confirm that [NQ answer] is correct?"
  - "How does [NQ answer] align with established facts or knowledge?"
  - "What makes [NQ answer] the appropriate response to [NQ question]?"
- If answer is INCORRECT (negative tone):
  - "What factors might challenge or contradict [NQ answer]?"
  - "What evidence would refute or invalidate [NQ answer]?"
  - "What are the reasons that make [NQ answer] an incorrect response?"
  - "How does [NQ answer] conflict with established facts or knowledge?"
  - "What makes [NQ answer] inappropriate or wrong for [NQ question]?"

AVOID:
- Simple Yes/No questions
- Questions that directly restate the NQ
- Questions that mention or reference the OQ
- Questions that explicitly state "the answer is [NQ answer]"
- Vague questions without clear focus
- **Using the wrong tone**: If answer is correct, do NOT use negative/critical tone. If answer is incorrect, do NOT use positive/affirmative tone.
- **Changing, replacing, or modifying key entities** (names, places, numbers, dates, etc. from NQ)
- **Adding new entities or facts not relevant to the NQ**
- **Hallucinating or inventing information**
- **Using synonyms or alternatives for key entities**
- **Adding details or context not relevant to the NQ**

<output_format>
Output exactly {n_pairs} blocks, and nothing else. Use the following structure for each pair:
<qa_pair>
<question>
[Your open-ended question with diverse tone/perspective about the NQ]
</question>
<answer>
[Your answer that guides understanding and judgment without directly stating the answer]
</answer>
</qa_pair>
</output_format>
{additional_text}
"""


# ---------------------
# Learning QA generation functions
# ---------------------
def generate_learning_qa_for_oq(
    sample: Dict[str, Any],
    client: LLMClient,
    n_pairs: int = 8,
    additional_text: str = "",
) -> List[Dict[str, str]]:
    """
    Generate learning QA pairs for OQ through question variants and different answer expressions
    
    Generation strategy:
    - Question variants: Generate paraphrases, synonymous expressions, different angles while keeping core semantics
    - Answer variants: Generate different expression forms of answers with varying vocabulary, structures, detail levels while keeping facts consistent
    - Goal: Help model deeply understand OQ essence and answer expression through diverse question-answer pairs
    """
    oq = sample.get("original_question", "").strip()
    oa = sample.get("original_answer", "").strip()
    
    if not oq or not oa:
        return []
    
    support = extract_support(sample)
    
    try:
        prompt = build_learning_qa_prompt_for_oq(oq, oa, support, n_pairs, additional_text)
        text = client.generate(
            prompt,
            temperature=0.3,
            top_p=0.9,
            max_tokens=4096,
            system_message="You generate diverse, open-ended question-answer pairs that help deeply understand concepts and relationships. Focus on What/Why/How/Explain questions, NOT Boolean or simple multiple choice. CRITICAL: Keep all key entities (names, places, numbers, dates) exactly unchanged. Do NOT hallucinate or invent new information."
        )
        
        if not text or not text.strip():
            return []
        
        parsed = parse_qapairs(text, max_pairs=n_pairs)
        return parsed
    except Exception as e:
        print(f"Error generating OQ learning QA: {str(e)}")
        return []


def build_oq_nq_combined_qa_prompt(
    oq: str,
    oa: str,
    nq_question: str,
    nq_answer: str,
    nq_category: str,
    n_pairs: int,
    additional_text: str = "",
) -> str:
    """Build OQ and NQ combined QA prompt"""
    additional_text = (additional_text or "").strip()
    
    category_explanation = {
        "entity_prerequisite": "Entity Prerequisite: The NQ verifies an attribute of the OQ answer entity.",
        "logical_implication": "Logical Implication: The NQ asks about a logical consequence related to the OQ.",
        "thematic_association": "Thematic Association: The NQ connects to the OQ through a broader theme or topic.",
    }
    
    explanation = category_explanation.get(nq_category, "")
    
    return f"""Given the following Original Question (OQ) and Neighbor Question (NQ):

<original_question>
{oq}
</original_question>

<original_answer>
{oa}
</original_answer>

<neighbor_question>
{nq_question}
</neighbor_question>

<neighbor_answer>
{nq_answer}
</neighbor_answer>

<relationship_type>
{nq_category} - {explanation}
</relationship_type>

Generate {n_pairs} question-answer pairs that combine or connect the OQ and NQ. These QA pairs should help a model learn the relationship between OQ and NQ.

REQUIREMENTS:
1. Question types: Use open-ended questions (What/Why/How/Explain), NOT Boolean or simple multiple choice
2. Combine OQ and NQ: Questions should either:
   - Explicitly ask about the relationship between OQ and NQ
   - Require understanding both OQ and NQ to answer correctly
   - Connect OQ and NQ to show how they relate
3. Examples of good questions:
   - "How does knowing [OQ answer] help verify [NQ question]?"
   - "What broader knowledge connects [OQ topic] and [NQ topic]?"
   - "Given [OQ fact], what can we infer about [NQ attribute]?"
   - "Explain how [OQ entity] relates to [NQ concept]"
4. Answers should demonstrate understanding of both OQ and NQ and their relationship
5. Answers should be detailed and explanatory
6. **CRITICAL: Keep all key entities unchanged** (person names, place names, organization names, concept names, numbers, dates, etc. from OQ/OA/NQ must remain exactly the same)
7. **ANTI-HALLUCINATION:**
   - Only explain the relationship between the given OQ/OA/NQ information
   - Do NOT invent new entities, facts, or information not present in the original OQ/OA/NQ
   - Do NOT replace key entities with synonyms or alternatives
   - Do NOT add details that are not implied or stated in the original answers
   - If unsure about an entity or fact, keep it exactly as in the original

AVOID:
- Questions that only ask about OQ or NQ separately
- Simple Yes/No questions
- Questions that don't connect OQ and NQ
- **Changing, replacing, or modifying key entities** (names, places, numbers, dates, etc. from OQ/OA/NQ)
- **Adding new entities or facts not in the original OQ/OA/NQ**
- **Hallucinating or inventing information**
- **Using synonyms or alternatives for key entities**
- **Adding details or context not present in the original answers**

<output_format>
Output exactly {n_pairs} blocks, and nothing else. Use the following structure for each pair:
<qa_pair>
<question>
[Your complete question that combines OQ and NQ]
</question>
<answer>
[Your detailed answer that shows the relationship]
</answer>
</qa_pair>
</output_format>
{additional_text}
"""


def generate_oq_nq_combined_qa(
    sample: Dict[str, Any],
    nq: Dict[str, Any],
    client: LLMClient,
    n_pairs: int = 10,
    additional_text: str = "",
) -> List[Dict[str, str]]:
    """
    Generate OQ and NQ combined QA pairs to help model learn their relationship
    
    Args:
        sample: Sample data containing OQ and OA
        nq: Neighbor question dict
        client: LLM client
        n_pairs: Number of QA pairs to generate
        additional_text: Additional prompt text
    """
    oq = sample.get("original_question", "").strip()
    oa = sample.get("original_answer", "").strip()
    nq_question = nq.get("question", "").strip()
    nq_answer = nq.get("correct_answer", "").strip()
    nq_category = nq.get("category", "")
    
    if not all([oq, oa, nq_question, nq_answer]):
        return []
    
    try:
        prompt = build_oq_nq_combined_qa_prompt(
            oq, oa, nq_question, nq_answer, nq_category, n_pairs, additional_text
        )
        text = client.generate(
            prompt,
            temperature=0.3,
            top_p=0.9,
            max_tokens=4096,
            system_message="You generate question-answer pairs that combine original questions and neighbor questions, helping models learn their relationships. CRITICAL: Keep all key entities (names, places, numbers, dates) exactly unchanged. Do NOT hallucinate or invent new information."
        )
        
        if not text or not text.strip():
            return []
        
        parsed = parse_qapairs(text, max_pairs=n_pairs)
        return parsed
    except Exception as e:
        print(f"Error generating OQ-NQ combined QA: {str(e)}")
        return []


def generate_learning_qa_for_nq(
    nq: Dict[str, Any],
    oq: str,
    oa: str,
    client: LLMClient,
    n_pairs: int = 5,
    additional_text: str = "",
) -> List[Dict[str, str]]:
    """
    Generate learning QA pairs for each NQ, focus on NQ itself, use appropriate tone based on answer correctness (correct=positive, incorrect=negative)
    
    Generation strategy:
    - Focus only on NQ itself, no OQ-NQ relationship
    - Use appropriate tone based on NQ answer correctness (determined by LLM in prompt):
      * If answer is correct (e.g., "Yes", "A", correct factual answer), use positive tone
      * If answer is incorrect (e.g., "No", "B", "C", wrong factual answer), use negative tone
    - Don't directly provide judgments or original questions, learn through guiding open QA
    - Goal: Help model learn NQ judgment and selection through different toned questions
    """
    nq_question = nq.get("question", "").strip()
    nq_answer = nq.get("correct_answer", "").strip()
    category = nq.get("category", "")
    
    if not nq_question or not nq_answer or not oq or not oa:
        return []
    
    try:
        prompt = build_learning_qa_prompt_for_nq(
            oq, oa, nq_question, nq_answer, category, n_pairs, additional_text
        )
        text = client.generate(
            prompt,
            temperature=0.3,
            top_p=0.9,
            max_tokens=4096,
            system_message="You generate diverse, open-ended question-answer pairs focused ONLY on the neighbor question itself. Determine if the answer is correct or incorrect (e.g., Yes/No, A/B/C, or factual answer), then use appropriate tone (positive for correct, negative for incorrect) to help models learn how to judge and answer the question. Focus on What/Why/How/Explain questions. Do NOT mention the original question. CRITICAL: Keep all key entities (names, places, numbers, dates) exactly unchanged. Do NOT hallucinate or invent new information."
        )
        
        if not text or not text.strip():
            return []
        
        parsed = parse_qapairs(text, max_pairs=n_pairs)
        return parsed
    except Exception as e:
        print(f"Error generating NQ learning QA: {str(e)}")
        return []


# ---------------------
# Single sample processing
# ---------------------
def process_sample(
    sample: Dict[str, Any],
    client: LLMClient,
    qa_pairs_per_doc: int,
    additional_text: str,
    oq_learning_qa_pairs: int = 8,
    nq_learning_qa_pairs: int = 5,
    learning_qa_additional_text: str = "",
    max_nqs_for_learning: int = 5,
    oq_nq_combined_qa_pairs: int = 10,
) -> Dict[str, Any]:
    """Process single sample, generate QA pairs for valid_docs (optional), and generate learning QA
    
    Note: This function only generates as much as possible; the final 100 quota per sample
    is controlled by upper aggregation logic; this mainly does deduplication and format filtering.
    """
    metadata = sample.get("metadata") or {}
    valid_docs = metadata.get("valid_docs", [])

    # 1. Original document QA generation logic (optional, skip if no valid_docs)
    if valid_docs:
        for i, d in enumerate(valid_docs):
            fact = (d.get("content") or "").strip()
            if not fact:
                fallback_fact = (d.get("fact") or "").strip()
                if fallback_fact:
                    fact = fallback_fact
                else:
                    d.setdefault("qa_errors", []).append("Empty content/fact for QA generation.")
                    continue

            try:
                prompt = build_qapairs_prompt(fact=fact, n_pairs=qa_pairs_per_doc, additional_text=additional_text)
                text = client.generate(
                    prompt,
                    temperature=0.3,
                    top_p=0.9,
                    max_tokens=4096,
                    system_message="You generate diverse, high-quality QA pairs using ONLY open-ended questions (What/Why/How/Explain). NEVER generate Boolean (Yes/No) or multiple choice questions."
                )
                
                if not text or not text.strip():
                    d.setdefault("qa_errors", []).append("Empty LLM response for QA generation.")
                    continue
                
                parsed = parse_qapairs(text, max_pairs=qa_pairs_per_doc)
                if not parsed or len(parsed) == 0:
                    d.setdefault("qa_errors", []).append(f"No <qa_pair> parsed (response length={len(text)}).")
                    continue

                # Merge QA pairs and deduplicate (preserve order)
                existing = d.get("qa_pairs") or []
                seen = set((p.get("question", ""), p.get("answer", "")) for p in existing if isinstance(p, dict))
                new_pairs = []
                for p in parsed:
                    key = (p["question"], p["answer"])
                    if key not in seen:
                        new_pairs.append(p)
                        seen.add(key)
                d["qa_pairs"] = existing + new_pairs

            except Exception as e:
                d.setdefault("qa_errors", []).append(f"QA generation failed: {str(e)}")
    
    # Update valid_docs in metadata
    metadata["valid_docs"] = valid_docs
    
    # 2. Generate OQ learning QA
    oq_learning_qa = generate_learning_qa_for_oq(
        sample, client, n_pairs=oq_learning_qa_pairs, additional_text=learning_qa_additional_text
    )
    if oq_learning_qa:
        # Filter out questions that conflict with original OQ/NQ or are non-compliant
        original_question = (sample.get("original_question") or "").strip()
        neighbor_questions = sample.get("neighbor_questions", [])
        filtered_oq_learning = []
        for qa in oq_learning_qa:
            q = (qa.get("question") or "").strip()
            if _question_conflicts_with_oq_nq(q, original_question, neighbor_questions):
                continue
            filtered_oq_learning.append(qa)
        if filtered_oq_learning:
            metadata["oq_learning_qa"] = filtered_oq_learning
    
    # 3. Generate learning QA for each NQ
    nq_learning_qa_list = []
    neighbor_questions = sample.get("neighbor_questions", [])
    if neighbor_questions:
        for nq_idx, nq in enumerate(neighbor_questions[:max_nqs_for_learning]):
            if not isinstance(nq, dict):
                continue
            nq_qa = generate_learning_qa_for_nq(
                nq,
                sample.get("original_question", ""),
                sample.get("original_answer", ""),
                client,
                n_pairs=nq_learning_qa_pairs,
                additional_text=learning_qa_additional_text
            )
            if nq_qa:
                original_question = (sample.get("original_question") or "").strip()
                neighbor_questions_all = neighbor_questions
                filtered_nq_qa = []
                for qa in nq_qa:
                    q = (qa.get("question") or "").strip()
                    if _question_conflicts_with_oq_nq(q, original_question, neighbor_questions_all):
                        continue
                    # Also filter out obvious Yes/No style answers
                    a = (qa.get("answer") or "").strip().lower()
                    if a in {"yes", "no", "true", "false"}:
                        continue
                    filtered_nq_qa.append(qa)
                if filtered_nq_qa:
                    nq_learning_qa_list.append({
                        "nq_index": nq_idx,
                        "nq_question": nq.get("question", ""),
                        "nq_category": nq.get("category", ""),
                        "learning_qa": filtered_nq_qa
                    })
    if nq_learning_qa_list:
        metadata["nq_learning_qa"] = nq_learning_qa_list
    
    # 4. Generate OQ and NQ combined QA pairs
    oq_nq_combined_qa_list = []
    if neighbor_questions:
        # Generate combined QA for each NQ (limit count to avoid excess)
        for nq_idx, nq in enumerate(neighbor_questions[:max_nqs_for_learning]):
            if not isinstance(nq, dict):
                continue
            combined_qa = generate_oq_nq_combined_qa(
                sample,
                nq,
                client,
                n_pairs=oq_nq_combined_qa_pairs,
                additional_text=learning_qa_additional_text
            )
            if combined_qa:
                original_question = (sample.get("original_question") or "").strip()
                neighbor_questions_all = neighbor_questions
                filtered_combined = []
                for qa in combined_qa:
                    q = (qa.get("question") or "").strip()
                    if _question_conflicts_with_oq_nq(q, original_question, neighbor_questions_all):
                        continue
                    filtered_combined.append(qa)
                if filtered_combined:
                    oq_nq_combined_qa_list.append({
                        "nq_index": nq_idx,
                        "nq_question": nq.get("question", ""),
                        "nq_category": nq.get("category", ""),
                        "combined_qa": filtered_combined
                    })
    if oq_nq_combined_qa_list:
        metadata["oq_nq_combined_qa"] = oq_nq_combined_qa_list
    
    sample["metadata"] = metadata
    return sample


# ---------------------
# Step 3 main class
# ---------------------
class Step3GenQAPairs:
    """Step 3: Generate QA pairs (Document QA + OQ/NQ Learning QA + OQ-NQ Relationship QA)"""
    
    def __init__(
        self,
        provider: str,
        api_key: str = None,
        base_url: str = None,
        model_name: str = "DeepSeek-V3.2",
        max_workers: int = 64,
        api_concurrency: int = 64,
    ):
        self.client = LLMClient(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            api_concurrency=api_concurrency,
        )
        self.max_workers = max_workers
    
    def run(
        self,
        input_path: str,
        output_path: str,
        qa_pairs_per_doc: int = 10,
        additional_text: str = "",
        oq_learning_qa_pairs: int = 8,
        nq_learning_qa_pairs: int = 5,
        learning_qa_additional_text: str = "",
        max_nqs_for_learning: int = 5,
        oq_nq_combined_qa_pairs: int = 10,
    ):
        """Execute Step 3"""
        samples = load_json(input_path)
        total = len(samples)
        print(f"Loaded {total} samples. Generating QA pairs from metadata.valid_docs and learning QA for OQ/NQ...")

        results = [None] * total
        stats = {
            "total_samples": total,
            "samples_with_errors": 0,
            "total_docs_processed": 0,
            "total_doc_qapairs_generated": 0,
            "total_doc_qapairs_selected": 0,
            "total_oq_learning_qa_generated": 0,
            "total_oq_learning_qa_selected": 0,
            "total_nq_learning_qa_generated": 0,
            "total_nq_learning_qa_selected": 0,
            "total_oq_nq_combined_qa_generated": 0,
            "total_oq_nq_combined_qa_selected": 0,
            "total_final_qas": 0,
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {}
            for idx, sample in enumerate(samples):
                future = ex.submit(
                    process_sample,
                    sample,
                    self.client,
                    qa_pairs_per_doc,
                    additional_text,
                    oq_learning_qa_pairs,
                    nq_learning_qa_pairs,
                    learning_qa_additional_text,
                    max_nqs_for_learning,
                    oq_nq_combined_qa_pairs,
                )
                future_map[future] = idx

            with tqdm(total=total, desc="Generating QA pairs", unit="sample") as pbar:
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        result = fut.result()
                        md = result.get("metadata") or {}
                        docs = md.get("docs", [])
                        
                        stats["total_docs_processed"] += len(docs)
                        # 1) Document QA
                        doc_qas = []
                        for d in docs:
                            doc_qas.extend([qa for qa in d.get("qa_pairs", []) or [] if isinstance(qa, dict)])
                        # 2) OQ Learning QA
                        oq_learning = md.get("oq_learning_qa", []) or []
                        oq_qas = list(oq_learning)
                        # 3) NQ Learning QA
                        nq_learning_list = md.get("nq_learning_qa", []) or []
                        nq_qas = []
                        for item in nq_learning_list:
                            nq_qas.extend([qa for qa in item.get("learning_qa", []) or [] if isinstance(qa, dict)])
                        # 4) OQ-NQ Combined QA
                        oq_nq_combined_list = md.get("oq_nq_combined_qa", []) or []
                        combined_qas = []
                        for item in oq_nq_combined_list:
                            combined_qas.extend([qa for qa in item.get("combined_qa", []) or [] if isinstance(qa, dict)])

                        # Count generated
                        stats["total_doc_qapairs_generated"] += len(doc_qas)
                        stats["total_oq_learning_qa_generated"] += len(oq_qas)
                        stats["total_nq_learning_qa_generated"] += len(nq_qas)
                        stats["total_oq_nq_combined_qa_generated"] += len(combined_qas)

                        def _dedup_keep_order(qas: List[Dict[str, str]]) -> List[Dict[str, str]]:
                            seen_local = set()
                            out_local: List[Dict[str, str]] = []
                            for qa in qas:
                                key = (qa.get("question", ""), qa.get("answer", ""))
                                if key in seen_local:
                                    continue
                                seen_local.add(key)
                                out_local.append(qa)
                            return out_local

                        doc_qas = _dedup_keep_order(doc_qas)
                        oq_qas = _dedup_keep_order(oq_qas)
                        nq_qas = _dedup_keep_order(nq_qas)
                        combined_qas = _dedup_keep_order(combined_qas)

                        def _take(qas: List[Dict[str, str]], n: int) -> (List[Dict[str, str]], List[Dict[str, str]]):
                            return qas[:n], qas[n:]

                        selected_doc, rest_doc = _take(doc_qas, DOC_QA_TARGET)
                        selected_oq, rest_oq = _take(oq_qas, OQ_LEARNING_QA_TARGET)
                        selected_nq, rest_nq = _take(nq_qas, NQ_LEARNING_QA_TARGET)
                        selected_combined, rest_combined = _take(combined_qas, OQ_NQ_COMBINED_QA_TARGET)

                        # Count selected
                        stats["total_doc_qapairs_selected"] += len(selected_doc)
                        stats["total_oq_learning_qa_selected"] += len(selected_oq)
                        stats["total_nq_learning_qa_selected"] += len(selected_nq)
                        stats["total_oq_nq_combined_qa_selected"] += len(selected_combined)

                        all_qas = []
                        all_qas.extend(selected_doc)
                        all_qas.extend(selected_oq)
                        all_qas.extend(selected_nq)
                        all_qas.extend(selected_combined)

                        # If less than 100, fill from remaining; if still insufficient, cycle from deduplicated pool
                        if len(all_qas) < FINAL_QA_PER_SAMPLE:
                            leftovers = rest_doc + rest_oq + rest_nq + rest_combined
                            leftovers = _dedup_keep_order(leftovers)
                            i = 0
                            while len(all_qas) < FINAL_QA_PER_SAMPLE and leftovers:
                                all_qas.append(leftovers[i % len(leftovers)])
                                i += 1

                        if len(all_qas) > FINAL_QA_PER_SAMPLE:
                            all_qas = all_qas[:FINAL_QA_PER_SAMPLE]

                        stats["total_final_qas"] += len(all_qas)

                        # Write back to sample metadata for subsequent SFT use
                        md["final_qas"] = all_qas
                        result["metadata"] = md
                        results[idx] = result
                        
                    except Exception as e:
                        s = samples[idx]
                        m = s.get("metadata") or {}
                        m.setdefault("qa_errors", []).append(f"Sample processing failed: {str(e)}")
                        s["metadata"] = m
                        results[idx] = s
                        stats["samples_with_errors"] += 1
                    finally:
                        pbar.update(1)

        # Print statistics
        print("\n" + "=" * 60)
        print("Step 3: QA Generation Statistics")
        print("=" * 60)
        print(f"Total samples processed: {stats['total_samples']}")
        print(f"Samples with errors: {stats['samples_with_errors']}")
        print(f"Total docs processed: {stats['total_docs_processed']}")
        print(f"Document QA pairs generated: {stats['total_doc_qapairs_generated']}, selected: {stats['total_doc_qapairs_selected']}")
        print(f"OQ learning QA generated: {stats['total_oq_learning_qa_generated']}, selected: {stats['total_oq_learning_qa_selected']}")
        print(f"NQ learning QA generated: {stats['total_nq_learning_qa_generated']}, selected: {stats['total_nq_learning_qa_selected']}")
        print(f"OQ-NQ combined QA generated: {stats['total_oq_nq_combined_qa_generated']}, selected: {stats['total_oq_nq_combined_qa_selected']}")
        print(f"Total QA pairs generated: {stats['total_doc_qapairs_generated'] + stats['total_oq_learning_qa_generated'] + stats['total_nq_learning_qa_generated'] + stats['total_oq_nq_combined_qa_generated']}")
        print(f"Total QA pairs selected (final): {stats['total_final_qas']}")
        print("=" * 60 + "\n")

        print(f"Saving results to {output_path} ...")
        save_json(output_path, results)
        print("Done!")


def main():
    """Command line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 3: Generate QA pairs (doc QA + OQ/NQ learning QA + OQ-NQ combined QA). Target: 100 QA pairs per sample.")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "zhipu"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--qa_pairs_per_doc", type=int, default=10)
    parser.add_argument("--additional_text", type=str, default="")
    parser.add_argument("--oq_learning_qa_pairs", type=int, default=8)
    parser.add_argument("--nq_learning_qa_pairs", type=int, default=5)
    parser.add_argument("--learning_qa_additional_text", type=str, default="")
    parser.add_argument("--max_nqs_for_learning", type=int, default=5)
    parser.add_argument("--oq_nq_combined_qa_pairs", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="DeepSeek-V3.2")
    parser.add_argument("--base_url", type=str, default="https://www.dmxapi.cn/v1")
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--api_concurrency", type=int, default=64)
    args = parser.parse_args()

    step = Step3GenQAPairs(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_workers=args.max_workers,
        api_concurrency=args.api_concurrency,
    )
    step.run(
        input_path=args.input_file,
        output_path=args.output_file,
        qa_pairs_per_doc=args.qa_pairs_per_doc,
        additional_text=args.additional_text,
        oq_learning_qa_pairs=args.oq_learning_qa_pairs,
        nq_learning_qa_pairs=args.nq_learning_qa_pairs,
        learning_qa_additional_text=args.learning_qa_additional_text,
        max_nqs_for_learning=args.max_nqs_for_learning,
        oq_nq_combined_qa_pairs=args.oq_nq_combined_qa_pairs,
    )


if __name__ == "__main__":
    main()
