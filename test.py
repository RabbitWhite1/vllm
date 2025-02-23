"""
unset NCCL_ASYNC_ERROR_HANDLING
export NCCL_DEBUG=WARN # INFO/WARN
export CRS_LOGGING_LEVEL=WARN # INFO/WARN
export TORCH_CPP_LOG_LEVEL=WARNING # INFO/WARNING
export TORCH_DISTRIBUTED_DEBUG=INFO # INFO/WARN
"""

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# Input the model name or path. Can be GPTQ or AWQ models.
print("------------------------------------------------------------------------------------------")
llm = LLM(model="Qwen/Qwen2-1.5B-Instruct", tensor_parallel_size=1, enforce_eager=True)
print(llm)
print(llm.llm_engine, type(llm.llm_engine))
print("------------------------------------------------------------------------------------------")

# Prepare your prompts
prompt = "Tell me something about large language models."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
print("*"*192)
outputs = llm.generate([text], sampling_params)
print("*"*192)


# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
torch.distributed.destroy_process_group()