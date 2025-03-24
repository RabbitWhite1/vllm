"""
unset NCCL_ASYNC_ERROR_HANDLING
export NCCL_DEBUG=WARN # INFO/WARN
export CRS_LOGGING_LEVEL=WARN # INFO/WARN
export TORCH_CPP_LOG_LEVEL=WARNING # INFO/WARNING
export TORCH_DISTRIBUTED_DEBUG=INFO # INFO/WARN
"""

import argparse

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.model_executor.models.qwen2 import Qwen2Model

parser = argparse.ArgumentParser("Simple GPT")
parser.add_argument("--tp_size", default=1, type=int)
parser.add_argument("--num_layers", default=1, type=int)
args = parser.parse_args()

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512
)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(
    model="Qwen/Qwen2-1.5B-Instruct",
    tensor_parallel_size=args.tp_size,
    enforce_eager=True,
)
qwen2_for_causal_lm: "Qwen2ForCausalLM" = (
    llm.llm_engine.model_executor.driver_worker.model_runner.model
)
qwen2_model: Qwen2Model = qwen2_for_causal_lm.model
qwen2_model.start_layer = 0
qwen2_model.end_layer = args.num_layers
qwen2_model.layers = qwen2_model.layers[: args.num_layers]

# Prepare your prompts
prompt = "Tell me something about large language models."
messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
torch.distributed.destroy_process_group()
