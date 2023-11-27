from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_bnb_config
import torch

def load_tw_llama_model_and_tokenizer(base_model_path):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        quantization_config=get_bnb_config()
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    return model, tokenizer