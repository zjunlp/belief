from vllm import LLM, SamplingParams


def init_llm(model_path: str, tensor_parallel_size: int, gpu_memory_utilization: float) -> LLM:
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )


def default_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )

