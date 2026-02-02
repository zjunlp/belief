import os
import sys
import time
import socket
import subprocess
import requests
import re
import atexit
from typing import List, Optional
import torch
import socket  # Import socket module at top
# ----------------------------------------------------------------------------
# Helper class: vLLM Server process manager
# ----------------------------------------------------------------------------
class VLLMServerManager:
    """Independently manage vLLM process lifecycle"""
    def __init__(self, model_path: str, gpu_ids: str, port: int = 23456):
        self.model_path = model_path
        self.gpu_ids = gpu_ids      # e.g. "4,5,6,7"
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.process = None

    def start(self):
        # 1. Check if port is occupied
        if self._is_port_in_use():
            print(f"[vLLM Manager] Port {self.port} is already in use. Assuming service is running.")
            return

        print(f"[vLLM Manager] Starting vLLM Service...")
        print(f"  - Model: {self.model_path}")
        print(f"  - GPUs: {self.gpu_ids}")
        print(f"  - Port: {self.port}")

        # 2. Build startup command (using vllm.entrypoints.openai.api_server)
        # Calculate tensor_parallel_size
        tp_size = len(self.gpu_ids.split(','))
        log_file_path = "vllm_server.log"
        self.log_file = open(log_file_path, "w")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--served-model-name", self.model_path, # Ensure model name matches
            "--port", str(self.port),
            "--tensor-parallel-size", str(tp_size),
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.90",     # GPU memory utilization
            "--max-model-len", "4096",              # Prevent OOM
            "--disable-log-requests"                # Reduce log spam
        ]

        # 3. Set environment variables (key: CUDA_VISIBLE_DEVICES for GPU isolation)
        # 3. Set environment variables
        env = os.environ.copy()
        
        # [Key fix 1]: Set GPU isolation
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        
        # [Key fix 2]: Clean up torchrun pollution!
        # Must remove these variables, otherwise vLLM will try to join training cluster causing hang or port conflict
        vars_to_kill = [
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
            "TORCHELASTIC_RUN_ID", "NCCL_ASYNC_ERROR_HANDLING", 
            "OMP_NUM_THREADS", "UVLOOP_CONCURRENCY"
        ]
        for var in vars_to_kill:
            if var in env:
                del env[var]
        
        # 4. Start subprocess
        # stdout/stderr can be redirected to file for debugging, set to DEVNULL for simplicity
        self.process = subprocess.Popen(
            cmd, 
            env=env,
            stdout=self.log_file,    # Redirect stdout to file
            stderr=subprocess.STDOUT # Merge stderr into stdout
        )
        
        # 5. Register exit cleanup (prevent zombie vLLM process after training stops)
        atexit.register(self.stop)
        
        # 6. Wait for service ready
        self._wait_for_ready()

    def stop(self):
        if self.process:
            print("[vLLM Manager] Terminating vLLM process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def _is_port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', self.port)) == 0

    def _wait_for_ready(self, timeout=300):
        """Optimized: Fix exception capture types to avoid AttributeError"""
        health_check_url = f"{self.base_url}/models"
        print(f"[vLLM Manager] Starting health check, URL: {health_check_url}")
        print(f"[vLLM Manager] Timeout: {timeout}s, retrying every 3 seconds...")
        print(f"[vLLM Manager] vLLM subprocess PID: {self.process.pid if self.process else 'None'}")

        start_time = time.time()
        retry_count = 0

        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            retry_count += 1

            # Output progress every 12 seconds
            if retry_count % 4 == 0:
                print(f"[vLLM Manager] Waited {elapsed:.1f}s, vLLM service not ready yet... continuing (PID: {self.process.pid})")

            # Check vLLM subprocess status
            if self.process:
                pid = self.process.pid
                poll_result = self.process.poll()
                if poll_result is not None:
                    # Read subprocess error log (ensure log file is properly redirected)
                    try:
                        with open(VLLM_LOG_FILE, "r", encoding="utf-8") as f:
                            stderr = f.read()[-2000:]  # Read last 2000 chars (avoid overly long logs)
                    except Exception as e:
                        stderr = f"Unable to read log file: {str(e)}"
                    
                    raise RuntimeError(
                        f"vLLM subprocess unexpectedly exited! PID: {pid}, exit code: {poll_result}\n"
                        f"vLLM error log (last 2000 chars):\n{stderr}\n"
                        f"Troubleshooting suggestions:\n"
                        f"1. Check if model path is correct ({self.model_path});\n"
                        f"2. Check if specified GPUs ({self.gpu_ids}) are free (use nvidia-smi);\n"
                        f"3. Check if GPU memory is sufficient (32B 8bit model needs ~8-10GB/GPU);\n"
                        f"4. Check if vLLM version is compatible (recommend upgrading: pip install --upgrade vllm)"
                    )
                else:
                    # Subprocess running normally notification
                    if retry_count % 10 == 0:
                        print(f"[vLLM Manager] vLLM subprocess running normally (PID: {pid}), waiting for service ready...")

            # Access health check endpoint (fixed exception capture)
            try:
                res = requests.get(health_check_url, timeout=5)
                print(f"[vLLM Manager] Retry {retry_count}: status code {res.status_code}, response: {res.text[:100]}")
                
                if res.status_code == 200:
                    print(f"[vLLM Manager] Service ready! Total time: {elapsed:.1f}s")
                    return
                else:
                    print(f"[vLLM Manager] Service not ready: status code {res.status_code}, response: {res.text[:200]}")

            except requests.exceptions.ConnectionError:
                # Catch all connection errors (including connection refused, unreachable, etc.)
                print(f"[vLLM Manager] Retry {retry_count}: connection failed (service not started or port not listening)")
            except requests.exceptions.Timeout:
                print(f"[vLLM Manager] Retry {retry_count}: request timeout (service still starting)")
            except socket.timeout:
                # Catch underlying socket timeout (avoid missing)
                print(f"[vLLM Manager] Retry {retry_count}: underlying network timeout")
            except socket.error as e:
                # Catch other socket errors (e.g., port occupied, network unreachable)
                print(f"[vLLM Manager] Retry {retry_count}: Socket error - {str(e)}")
            except requests.exceptions.RequestException as e:
                # Catch other requests network errors
                print(f"[vLLM Manager] Retry {retry_count}: request failed - {str(e)}")
            except Exception as e:
                # Catch all other unknown errors
                print(f"[vLLM Manager] Retry {retry_count}: unknown error - {str(e)} (type: {type(e).__name__})")

            time.sleep(3)
        
        self.stop()
        raise TimeoutError(f"vLLM service timed out after {timeout}s")


# ----------------------------------------------------------------------------
# Main class: AnswerVerifier
# ----------------------------------------------------------------------------
# Global singleton reference to prevent Rank 0 from being garbage collected causing atexit hooks to fail or objects to be lost
_vllm_server_ref = None

class AnswerVerifier:
    def __init__(
        self, 
        model_name: str = "deepseek-chat", 
        api_key: str = None, 
        base_url: str = "https://api.deepseek.com",
        use_api: bool = True,
        judge_device_map: str = "4,5,6,7" # Default isolated GPU IDs
    ):
        self.model_name = model_name
        self.use_api = use_api
        
        # If not using remote API, start local vLLM
        if not self.use_api:
            vllm_port = 23456
            local_api_url = f"http://127.0.0.1:{vllm_port}/v1"
            
            # Get LOCAL_RANK (torchrun sets this environment variable automatically)
            # For single machine multi-GPU, LOCAL_RANK 0 is responsible for starting Server
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            
            if local_rank == 0:
                global _vllm_server_ref
                if _vllm_server_ref is None:
                    # Main process starts service
                    _vllm_server_ref = VLLMServerManager(model_name, judge_device_map, vllm_port)
                    _vllm_server_ref.start()
            
            # All processes (including Rank 0) wait for service ready
            # Although Rank 0 already waited in start(), we call check here for consistency and safety
            self._wait_for_service(local_api_url)
            
            # --- Key point: unified logic ---
            # After service starts, local vLLM becomes a standard OpenAI API interface
            # So we force use_api to True, subsequent logic fully reuses OpenAI Client
            print(f"[Rank {local_rank}] Connected to vLLM. Switching to API mode.")
            self.use_api = True
            base_url = local_api_url
            api_key = "EMPTY" # vLLM doesn't need key
            
        # Initialize OpenAI Client (whether remote or local, now uses OpenAI protocol)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        if api_key is None:
            api_key = os.environ.get('DEEPSEEK_API_KEY')
            if not api_key: raise ValueError("API Key missing")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"Verifier initialized against: {base_url}")

    def _wait_for_service(self, url, timeout=300):
        """Non-main process blocks waiting for port to be available"""
        print(f"Waiting for Judge Server at {url}...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                if requests.get(f"{url}/models", timeout=1).status_code == 200:
                    return
            except:
                pass
            time.sleep(2)
        raise RuntimeError("Service wait timeout")
    def verify_answer(self, golden_answer: str, extracted_answer: str, question: str = "") -> str:
        """Verify if extracted answer matches golden answer semantically
        
        Args:
            golden_answer: The correct answer
            extracted_answer: The answer to verify
            question: The question being answered (optional, for context)
            
        Returns:
            One of: "CORRECT", "INCORRECT", "NOT_ATTEMPTED"
        """
        prompt = f"""Your job is to grade a predicted answer based on a question and a gold target. Assign one grade: ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Key Principles:
- CORRECT: Fully contains essential information from the gold target without contradictions. Minor variations (spelling, order, hedging) are okay if the core meaning is right.
- INCORRECT: Contains factual contradictions to the gold target, or is significantly incomplete. Hedged incorrect statements are still INCORRECT.
- NOT_ATTEMPTED: Explicitly states inability to answer (e.g., "I don't know"), asks for more context evasively, or gives a very vague non-answer.

Examples:

1. CORRECT:
Question: Barack Obama's children?
Gold: Malia Obama and Sasha Obama
Predicted: sasha and malia obama (Correct - order/case don't matter)
Predicted: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly called Malia and Sasha. (Correct - extra info okay if not contradictory)

2. INCORRECT:
Question: Barack Obama's children?
Gold: Malia and Sasha
Predicted: Malia. (Incorrect - incomplete)
Predicted: Malia, Sasha, and Susan. (Incorrect - adds wrong info)
Predicted: I think it's Malia and Jackie. (Incorrect - factual contradiction despite hedging)

3. NOT_ATTEMPTED:
Question: Barack Obama's children?
Gold: Malia and Sasha
Predicted: I don't know. (Not Attempted)
Predicted: I need more context. (Not Attempted)

Important Considerations (Summary):
- Numbers: Must be correct to the gold target's significant figures (e.g., Gold "120k", Pred "115k" is CORRECT; "100k" is INCORRECT; "around 100k" is NOT_ATTEMPTED).
- Partial Gold Info: If gold has more info than the question asks, the prediction only needs to answer the question (e.g., Q: "Episode name?", Gold: "S7, E20: White Wedding", Pred: "White Wedding" is CORRECT).
- Inferred Info: Don't penalize omission of info clearly inferred from the question (e.g., Q: "OpenAI HQ city?", Gold: "San Francisco, California", Pred: "San Francisco" is CORRECT).
- Name Typos: Minor typos in names are okay if clearly the same person.

Grade the new example below. Respond with only "A" for CORRECT, "B" for INCORRECT, or "C" for NOT_ATTEMPTED. No extra text.

---
Question: {question if question else "N/A"}
Gold target: {golden_answer}
Predicted answer: {extracted_answer}
---
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10,
            )
            result = response.choices[0].message.content.strip().upper()
            
            # Map A/B/C to grade labels
            if result == "A":
                return "CORRECT"
            elif result == "B":
                return "INCORRECT"
            elif result == "C":
                return "NOT_ATTEMPTED"
            else:
                # Try to extract the grade if model didn't follow format exactly
                if "CORRECT" in result:
                    return "CORRECT"
                elif "NOT_ATTEMPTED" in result or "NOT ATTEMPTED" in result:
                    return "NOT_ATTEMPTED"
                elif "INCORRECT" in result:
                    return "INCORRECT"
                else:
                    print(f"Unexpected LLM response: {result}, defaulting to INCORRECT")
                    return "INCORRECT"
        except Exception as e:
            print(f"Error verifying answer with LLM: {e}")
            # Fallback to simple substring matching
            if golden_answer.lower() in extracted_answer.lower():
                return "CORRECT"
            else:
                return "INCORRECT"


def _normalize_answer(text: str) -> str:
    """Normalize answer text for comparison by removing articles and extra whitespace"""
    text = text.lower().strip()
    # Remove common articles and punctuation
    text = re.sub(r'\b(a|an|the)\b', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove trailing punctuation
    text = re.sub(r'[.,!?;:]+$', '', text)
    return text

def _check_answer_match(golden_answer: str, extracted_answer: str, question: str = "", verifier=None) -> str:
    """
    Check if golden answer matches extracted answer using multiple strategies:
    1. LLM-based verification (if verifier provided) - returns "CORRECT", "INCORRECT", or "NOT_ATTEMPTED"
    2. Exact substring match
    3. Normalized token-based match (all golden tokens present in extracted)
    4. Fuzzy match for minor variations
    
    Returns:
        "CORRECT", "INCORRECT", or "NOT_ATTEMPTED" (for LLM mode)
        "CORRECT" or "INCORRECT" (for rule-based mode)
    """
    # Strategy 1: LLM-based verification (most accurate)
    if verifier is not None:
        return verifier.verify_answer(golden_answer, extracted_answer, question)
    
    # Fallback strategies for rule-based matching
    golden_lower = golden_answer.lower().strip()
    extracted_lower = extracted_answer.lower().strip()
    
    # Strategy 2: Direct substring match
    if golden_lower in extracted_lower:
        return "CORRECT"
    
    # Strategy 3: Normalized token-based matching
    golden_normalized = _normalize_answer(golden_answer)
    extracted_normalized = _normalize_answer(extracted_answer)
    
    # Check if all tokens from golden answer appear in extracted answer
    golden_tokens = set(golden_normalized.split())
    extracted_tokens = set(extracted_normalized.split())
    
    if golden_tokens and golden_tokens.issubset(extracted_tokens):
        return "CORRECT"
    
    # Strategy 4: Check if normalized golden answer is substring of normalized extracted
    if golden_normalized and golden_normalized in extracted_normalized:
        return "CORRECT"
    
    return "INCORRECT"

# Global instance for answer verifier (initialized lazily)
_answer_verifier = None

def ACC_reward(completions, **kwargs):
    """
    Accuracy-based reward function for GRPO with three-tier scoring:
    - +1.0: CORRECT - Golden answer appears in completion (correct answer)
    - -0.5: NOT_ATTEMPTED - Model refuses to answer or abstains
    - -1.0: INCORRECT - Wrong answer (neither correct nor abstention)
    
    Supports two answer formats:
    1. Structured: <answer>content</answer>
    2. Unstructured: answer appears anywhere in completion
    
    Uses robust matching that handles:
    - LLM-based semantic verification (if initialized with init_acc_reward)
      - Uses detailed grading prompt with examples
      - Handles partial matches, typos, and semantic equivalence
    - Exact matches: "coast guard" in "the coast guard"
    - Token-based matches: "coast guard" matches "u.s. coast guard"
    - Normalized matches: handles articles and punctuation variations
    
    To use LLM-based verification, call init_acc_reward() first:
        init_acc_reward('deepseek-chat')
    """
    global _answer_verifier
    import re
    rewards = []
    
    # Extract solutions (golden answers) and questions from kwargs
    solutions = kwargs.get('solution', [])
    questions = kwargs.get('question', [])
    
    for i, completion in enumerate(completions):
        reward = 0.0
        
        if i < len(solutions):
            golden_answer = solutions[i][0]['content']
            question = questions[i] if i < len(questions) else ""
            
            # Handle different completion formats
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict) and 'content' in completion[0]:
                completion_text = completion[0]['content']
            elif isinstance(completion, dict) and 'content' in completion:
                completion_text = completion['content']
            else:
                completion_text = str(completion)
            
            # First, try to extract structured answer using <answer> tags
            answer_match = re.search(r'<answer>(.*?)</answer>', completion_text, re.IGNORECASE | re.DOTALL)
            
            if answer_match:
                # Structured format found - use robust matching
                extracted_answer = answer_match.group(1).strip()
            else:
                # Fallback to full completion text
                extracted_answer = completion_text
            
            # Get grade from matching function
            grade = _check_answer_match(golden_answer, extracted_answer, question, _answer_verifier)
            
            # Map grade to reward
            if grade == "CORRECT":
                reward = 1.0
            elif grade == "NOT_ATTEMPTED":
                reward = -0.5
            else:  # INCORRECT
                reward = -1.0
            
        rewards.append(reward)
    
    return rewards

def init_acc_reward(model_name: str = "deepseek-chat", api_key: str = None, base_url: str = "https://api.deepseek.com",use_api: bool = True, judge_device_map: str = "4,5,6,7"):
    """
    Initialize the ACC reward function with LLM-based answer verification.
    
    Args:
        model_name: Model name for answer verification (default: "deepseek-chat")
        api_key: API key for DeepSeek (reads from DEEPSEEK_API_KEY env var if not provided)
        base_url: Base URL for the API (default: DeepSeek API)
    
    Example:
        init_acc_reward('deepseek-chat')
        # Or with custom API key
        init_acc_reward('deepseek-chat', api_key='your-api-key')
    """
    global _answer_verifier
    _answer_verifier = AnswerVerifier(model_name, api_key, base_url,use_api, judge_device_map)
    print(f"Initialized LLM-based ACC reward with model: {model_name}")

class EntityExtractor:
    """LLM-based entity extractor using OpenAI-compatible API (e.g., DeepSeek)"""
    
    def __init__(self, model_name: str = "deepseek-chat", api_key: str = None, base_url: str = "https://api.deepseek.com"):
        """Initialize the entity extractor with OpenAI-compatible API
        
        Args:
            model_name: Model name (default: "deepseek-chat")
            api_key: API key for the service (reads from DEEPSEEK_API_KEY env var if not provided)
            base_url: Base URL for the API (default: DeepSeek API)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package is required for API-based entity extraction. Install with: pip install openai")
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get('DEEPSEEK_API_KEY')
            if api_key is None:
                raise ValueError("API key must be provided either as argument or via DEEPSEEK_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        print(f"Initialized EntityExtractor with API model: {model_name}")
    
    def extract_entities(self, completions: List[str], question: str = "", golden_answer: str = "") -> List[str]:
        """Extract entities from completions using LLM
        
        Args:
            completions: List of completion texts
            question: The question being answered (optional, for context)
            golden_answer: The golden answer (optional, for context)
            
        Returns:
            List of extracted entities
        """
        extract_prompts = []
        
        for completion in completions:
            # Handle different completion formats
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict) and 'content' in completion[0]:
                completion_text = completion[0]['content']
            elif isinstance(completion, dict) and 'content' in completion:
                completion_text = completion['content']
            else:
                completion_text = str(completion)
            
            # Extract from <answer> tags if present
            answer_match = re.search(r'<answer>(.*?)</answer>', completion_text, re.IGNORECASE | re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
            else:
                answer_text = completion_text
            
            # Build extraction prompt
            if question and golden_answer:
                extract_prompt = f"""
Your task is to extract the **main answer entity** from the **Response** that directly answers the **Question**.

### Extraction Principles

1. **Direct Entity Rule**
   - If the Response itself *is* a single entity (a name, number, date, or short noun phrase), 
     return it directly.

2. **Focused Extraction Rule**
   - Otherwise, find the **one concise noun phrase or named entity** that most directly answers the Question.
   - Use the **Golden answer** only to understand what *type* of entity (person, location, date, quantity, etc.) should be extracted.
   - If multiple entities are present, pick the one that most likely fulfills the Question’s intent.

3. **Cautious None Rule**
   - Only output `None` if and only if the Response explicitly says the answer is *unknown*, *not provided*, *not specified*, *cannot be determined*, or *no information given*; **and**
   
4. **Output Format**
   - Output exactly **Answer entity** or `None`.
   - No punctuation, quotes, or explanation.

---

**Question:** {question}

**Golden answer:** {golden_answer}

**Response:** {answer_text}

**Extracted Entity:**
"""
            
                extract_prompts.append(extract_prompt)
        
        # Batch inference with OpenAI API
        entities = []
        for extract_prompt in extract_prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": extract_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=50,
                )
                entity = response.choices[0].message.content.strip().lower()
                entity = re.sub(r'[.,!?;:]+$', '', entity)
                if not entity:
                    entity = "none"
                entities.append(entity)
            except Exception as e:
                print(f"Error extracting entity: {e}")
                entities.append("none")
        
        return entities


class ConsistencyRewardCalculator:
    """Consistency reward calculator with LLM-based entity extraction"""
    
    def __init__(self, extractor: Optional[EntityExtractor] = None):
        """Initialize with an optional entity extractor
        
        Args:
            extractor: EntityExtractor instance. If None, uses rule-based extraction.
        """
        self.extractor = extractor
    
    def __call__(self, completions, prompts=None, **kwargs):
        """Calculate consistency rewards using majority voting
        
        Args:
            completions: List of completion texts
            prompts: List of prompts (optional, for context)
            **kwargs: Additional arguments including 'question' and 'solution'
        """
        if not completions:
            return []
        
        # Extract question and golden answer from kwargs if available
        questions = kwargs.get('question', [])
        solutions = kwargs.get('solution', [])
        
        # For GRPO, we typically have multiple completions per prompt
        # Group completions by their corresponding prompt
        if prompts and len(prompts) == len(completions):
            # Each completion has its own prompt - group by unique prompts
            from collections import defaultdict
            prompt_groups = defaultdict(list)
            for i, prompt in enumerate(prompts):
                prompt_key = str(prompt)
                prompt_groups[prompt_key].append(i)
            
            all_rewards = [0.0] * len(completions)
            
            for prompt_key, indices in prompt_groups.items():
                group_completions = [completions[i] for i in indices]
                
                # Get question and golden answer for this group
                question = questions[indices[0]] if indices[0] < len(questions) else ""
                golden_answer = ""
                if indices[0] < len(solutions) and solutions[indices[0]]:
                    if isinstance(solutions[indices[0]], list) and len(solutions[indices[0]]) > 0:
                        golden_answer = solutions[indices[0]][0].get('content', '') if isinstance(solutions[indices[0]][0], dict) else str(solutions[indices[0]][0])
                    else:
                        golden_answer = str(solutions[indices[0]])
                
                # Extract entities
                if self.extractor:
                    entities = self.extractor.extract_entities(group_completions, question, golden_answer)
                else:
                    entities = self._rule_based_extraction(group_completions)
                
                # Calculate rewards for this group
                group_rewards = self._calculate_group_rewards(entities)
                
                # Assign rewards back to original positions
                for i, reward in zip(indices, group_rewards):
                    all_rewards[i] = reward
            
            return all_rewards
        else:
            # All completions are for the same prompt
            question = questions[0] if len(questions) > 0 else ""
            golden_answer = ""
            if len(solutions) > 0 and solutions[0]:
                if isinstance(solutions[0], list) and len(solutions[0]) > 0:
                    golden_answer = solutions[0][0].get('content', '') if isinstance(solutions[0][0], dict) else str(solutions[0][0])
                else:
                    golden_answer = str(solutions[0])
            
            if self.extractor:
                entities = self.extractor.extract_entities(completions, question, golden_answer)
            else:
                entities = self._rule_based_extraction(completions)
            
            return self._calculate_group_rewards(entities)
    
    def _rule_based_extraction(self, completions: List) -> List[str]:
        """Rule-based entity extraction fallback"""
        extracted_entities = []
        
        for completion in completions:
            # Handle different completion formats
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict) and 'content' in completion[0]:
                completion_text = completion[0]['content']
            elif isinstance(completion, dict) and 'content' in completion:
                completion_text = completion['content']
            else:
                completion_text = str(completion)
            
            # Try to extract structured answer using <answer> tags first
            answer_match = re.search(r'<answer>(.*?)</answer>', completion_text, re.IGNORECASE | re.DOTALL)
            
            if answer_match:
                answer_text = answer_match.group(1).strip()
            else:
                answer_text = completion_text.strip()
            
            # Extract the core entity/answer from the text
            answer_lower = answer_text.lower()
            
            # Check for abstention/uncertainty patterns
            abstention_patterns = [
                "i do not have", "i don't have", "i cannot", "i can't",
                "not have specific information", "unable to provide",
                "i do not know", "i don't know", "without access to",
                "unknown", "not specified", "cannot answer", "insufficient information"
            ]
            
            if any(pattern in answer_lower for pattern in abstention_patterns):
                entity = "none"
            else:
                # Clean up the answer text to extract core entity
                entity = answer_lower
                entity = re.sub(r'^(the answer is|it is|the|a|an)\s+', '', entity)
                entity = re.sub(r'\s+(is the answer|is correct)$', '', entity)
                entity = re.sub(r'[.,!?;:]+$', '', entity)
                entity = entity.strip()
                
                # If the answer is too long, extract first key phrase
                if len(entity.split()) > 10:
                    first_sentence = re.split(r'[.!?]', entity)[0]
                    if len(first_sentence.split()) <= 10:
                        entity = first_sentence.strip()
                    else:
                        entity = ' '.join(entity.split()[:5])
            
            extracted_entities.append(entity)
        
        return extracted_entities
    
    def _calculate_group_rewards(self, entities: List[str]) -> List[float]:
        """Calculate rewards based on majority voting
        
        Args:
            entities: List of extracted entities
            
        Returns:
            List of rewards
        """
        # Find the majority entity
        entity_counts = Counter(entities)
        majority_entity = entity_counts.most_common(1)[0][0]
        majority_count = entity_counts[majority_entity]
        
        # Calculate reward for each completion
        k = len(entities)
        rewards = []
        
        for entity in entities:
            if entity == majority_entity:
                reward = majority_count / k
            else:
                reward = 0.0
            rewards.append(reward)
        
        return rewards


# Global instance for consistency reward (initialized lazily)
_consistency_reward_calculator = None

def Consistency_reward(completions, prompts=None, **kwargs):
    """
    Consistency-based reward function using majority voting on extracted entities.
    For k completions, the reward for each completion is:
    r(x) = (1/k) * sum_{i=1}^{k} 1[y_i = Maj{y_1, ..., y_k}]
    
    Where Maj{y_1, ..., y_k} is the majority vote among all extracted entities.
    Each completion gets a reward proportional to how many completions agree with the majority.
    
    This function uses LLM-based entity extraction if configured, otherwise falls back to rule-based extraction.
    
    To use LLM-based extraction, call init_consistency_reward() first with a judge model.
    """
    global _consistency_reward_calculator
    
    if _consistency_reward_calculator is None:
        # Initialize with rule-based extraction by default
        _consistency_reward_calculator = ConsistencyRewardCalculator()
    
    return _consistency_reward_calculator(completions, prompts, **kwargs)


def init_consistency_reward(judge_model_name: str = "deepseek-chat", api_key: str = None, base_url: str = "https://api.deepseek.com"):
    """
    Initialize the consistency reward calculator with API-based entity extraction.
    
    Args:
        judge_model_name: Model name for entity extraction (default: "deepseek-chat")
        api_key: API key for the service (reads from DEEPSEEK_API_KEY env var if not provided)
        base_url: Base URL for the API (default: DeepSeek API)
    
    Example:
        init_consistency_reward('deepseek-chat')
        # Or with custom API key
        init_consistency_reward('deepseek-chat', api_key='your-api-key')
    """
    global _consistency_reward_calculator
    extractor = EntityExtractor(judge_model_name, api_key, base_url)
    _consistency_reward_calculator = ConsistencyRewardCalculator(extractor)
    print(f"Initialized API-based consistency reward with model: {judge_model_name}")


# Global reward function registry
REWARD_REGISTRY = {
    'acc': ACC_reward,
    'accuracy': ACC_reward,
    'consistency': Consistency_reward,
}


def get_reward_function(reward_type: str):
    """
    Get reward function from registry.
    
    Args:
        reward_type: Type of reward function ('acc', 'accuracy', 'consistency')
        
    Returns:
        Reward function callable
        
    Raises:
        ValueError: If reward_type is not registered
    """
    reward_type_lower = reward_type.lower()
    if reward_type_lower not in REWARD_REGISTRY:
        available = ', '.join(REWARD_REGISTRY.keys())
        raise ValueError(f"Unknown reward type: '{reward_type}'. Available types: {available}")
    
    return REWARD_REGISTRY[reward_type_lower]


def register_reward_function(name: str, func):
    """
    Register a custom reward function.
    
    Args:
        name: Name to register the function under
        func: Reward function callable
        
    Example:
        def my_custom_reward(completions, **kwargs):
            return [1.0] * len(completions)
        
        register_reward_function('custom', my_custom_reward)
    """
    REWARD_REGISTRY[name.lower()] = func
    print(f"Registered reward function: '{name}'")