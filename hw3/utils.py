from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
    # return (
    #     "你是一位在繁體中文領域具有卓越專業水準的專家，精通文言文與中文翻譯，具備深厚的文學造詣與卓越的語言技巧。"
    #     "接下來是一個學生與你的對話。對話內容是請你協助處理一個翻譯任務，這個任務可能是期望將中文翻譯成文言文，也可能是將文言文翻譯成中文。"
    #     "請你運用你在中文領域水準非常高超的知識，在接收到翻譯任務描述後，專業請精確地處理任務並回傳翻譯任務的結果。"
    #     "\n\n"
    #     f"STUDENT: {instruction}\n"
    #     "EXPERT:"
    # )

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def add_prompt(example):
    return {'input': get_prompt(example['instruction'])}