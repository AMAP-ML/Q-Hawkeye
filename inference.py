

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch
import os

device = "cuda:0"
MODEL_PATH = "/path/to/your/model"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device
)



SYSTEM_PROMPT = (
    """You are doing the image quality assessment task. Here is the question: What is your overall rating on the quality of this picture?\nPlease provide your response in the following format:\n<think>\n[Your detailed analysis of the image quality here]\n</think>\n<answer>\n[Only the numerical score here]\n</answer>\nRequirements:\n- The rating should be a float between 1 and 5, rounded to two decimal places\n- 1 represents very poor quality and 5 represents excellent quality\n- The <answer></answer> tags must contain ONLY the numerical score (e.g., 3.75), no other text
"""
)
    
processor = AutoProcessor.from_pretrained(MODEL_PATH)
template = ""

image_path = "/Path/to/your/image.jpg"
SCORE_QUESTION_PROMPT = 'Please evaluate and rating this picture following the exact format specified in the instructions. The rating should be a float between 1 and 5, rounded to two decimal places. Return the final answer with the following keys: final_score : The score.'



message = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": SCORE_QUESTION_PROMPT
            },
            {"type": "image", "image": f"file://{image_path}"}
        ]
    }
]

text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
image_inputs, video_inputs = process_vision_info([message])
inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

gen_config = GenerationConfig(
  do_sample=True, 
  temperature=1.0,
  max_new_tokens=1024,
)

generated_ids = model.generate(
  **inputs,
  generation_config=gen_config,
  use_cache=True,
)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("Model Response:")
print(output_text)
