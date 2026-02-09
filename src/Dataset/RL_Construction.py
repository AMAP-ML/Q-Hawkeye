import json
import re
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path
from datasets import DatasetDict, Dataset, Features, Sequence, Image as HFImage, Value
from PIL import Image
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class CompleteMLLMDatasetConverter:
    
    def __init__(self, 
                 keep_fields: Optional[List[str]] = None,
                 num_workers: int = None,
                 batch_size: int = 100):
        self.extra_keep_fields = keep_fields or []
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        self.batch_size = batch_size
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    
    def replace_score_in_response(self, response: str, gt_score: float) -> str:
        formatted_score = f"{gt_score:.2f}" if isinstance(gt_score, (int, float)) else str(gt_score)
        
        def replace_answer(match):
            return f"<answer>{formatted_score}</answer>"
        
        modified_response = self.answer_pattern.sub(replace_answer, response)
        
        if modified_response == response:
            score_pattern = re.compile(r'\b\d+\.\d{1,2}\b')
            matches = list(score_pattern.finditer(response))
            if matches:
                last_match = matches[-1]
                modified_response = (
                    response[:last_match.start()] + 
                    formatted_score + 
                    response[last_match.end():]
                )
        
        return modified_response
    
    def is_valid_field(self, field_value: Any) -> bool:
        if field_value is None:
            return False
        if field_value == "null":
            return False
        if isinstance(field_value, str) and field_value.strip() == "":
            return False
        if isinstance(field_value, list) and len(field_value) == 0:
            return False
        return True
    
    def get_user_query(self, item: Dict[str, Any], default_query: str = None) -> str:
        if default_query is None:
            default_query = "Please evaluate and rate this picture following the exact format specified in the instructions. The rating should be a float between 1 and 5, rounded to two decimal places. Return the final answer with the following format: <answer>The score</answer>."
        
        if "conversations" in item and isinstance(item["conversations"], list) and len(item["conversations"]) > 0:
            first_conv = item["conversations"][0]
            if isinstance(first_conv, dict) and "value" in first_conv:
                query = first_conv["value"]
                if query and query.strip():
                    return query
        
        if "query" in item and self.is_valid_field(item["query"]):
            return item["query"]
        
        if "question" in item and self.is_valid_field(item["question"]):
            return item["question"]
        
        return default_query
    
    def ensure_image_tag(self, query: str) -> str:
        query = query.replace("<|image|>", "<image>")
        query = query.replace("[IMAGE]", "<image>")
        
        if "<image>" not in query:
            query = query.rstrip() + "\n<image>"
        
        return query
    
    def get_response(self, item: Dict[str, Any]) -> str:
        response_fields = ["response", "assistant_response", "answer", "output"]
        
        for field in response_fields:
            if field in item and self.is_valid_field(item[field]):
                return item[field]
        
        return ""
    
    def get_image_paths(self, item: Dict[str, Any]) -> List[str]:
        if "image" in item and self.is_valid_field(item["image"]):
            if isinstance(item["image"], list):
                return item["image"]
            else:
                return [item["image"]]
        
        if "images" in item and isinstance(item["images"], list):
            return item["images"]
        
        if "image_path" in item and self.is_valid_field(item["image_path"]):
            return [item["image_path"]]
        
        return []
    
    def load_system_prompt(self, prompt_file: str) -> Optional[str]:
        if prompt_file and os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    print(f"✓ Successfully loaded system prompt from: {prompt_file}")
                    print(f"  Content preview: {content[:100]}..." if len(content) > 100 else f"  Content: {content}")
                    return content
                else:
                    print(f"⚠ Warning: File {prompt_file} is empty")
                    return None
            except Exception as e:
                print(f"✗ Failed to read system prompt file: {e}")
                return None
        elif prompt_file:
            print(f"⚠ Warning: System prompt file does not exist: {prompt_file}")
            return None
        else:
            return None
    
    def convert_to_sft_format(self, 
                             item: Dict[str, Any], 
                             user_query: Optional[str] = None,
                             system_prompt: Optional[str] = None,
                             replace_score: bool = True) -> Dict[str, Any]:
        response_content = self.get_response(item)
        
        if not response_content:
            print(f"Warning: Data item has no valid response, ID: {item.get('id', 'unknown')}")
            return None
        
        if replace_score and "gt_score" in item:
            gt_score = item["gt_score"]
            response_content = self.replace_score_in_response(response_content, gt_score)
        
        if user_query:
            final_user_query = user_query
        else:
            final_user_query = self.get_user_query(item)
        
        final_user_query = self.ensure_image_tag(final_user_query)
        
        messages = []
        
        if system_prompt:
            messages.append({
                "content": system_prompt,
                "role": "system"
            })
        
        messages.extend([
            {
                "content": final_user_query,
                "role": "user"
            },
            {
                "content": response_content,
                "role": "assistant"
            }
        ])
        
        images = self.get_image_paths(item)
        
        new_item = {
            "messages": messages,
            "images": images
        }
        
        if "id" in item:
            new_item["id"] = item["id"]
        
        if "gt_score" in item:
            new_item["gt_score"] = item["gt_score"]
        
        for field in self.extra_keep_fields:
            if field in item and field not in ["response", "conversations", "image", "images"]:
                new_item[field] = item[field]
        
        return new_item
    
    def process_images_batch(self, image_paths: List[str]) -> List[Image.Image]:
        images = []
        for img_path in image_paths:
            try:
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_copy = img.copy()
                    images.append(img_copy)
                else:
                    placeholder = Image.new('RGB', (224, 224), color='gray')
                    images.append(placeholder)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                placeholder = Image.new('RGB', (224, 224), color='gray')
                images.append(placeholder)
        return images
    
    def process_dataset_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        messages = item['messages']
        
        user_content = ""
        assistant_content = ""
        
        for msg in messages:
            role = msg['role']
            if role == 'user':
                user_content = msg['content']
            elif role == 'assistant':
                assistant_content = msg['content']
        
        problem = user_content
        solution = assistant_content
        
        images = []
        if 'images' in item and item['images']:
            images = self.process_images_batch(item['images'])
        
        if not images:
            placeholder = Image.new('RGB', (224, 224), color='gray')
            images.append(placeholder)
        
        result = {
            'image': images,
            'problem': problem,
            'solution': solution,
        }
        
        for key, value in item.items():
            if key not in ['messages', 'images', 'problem', 'solution', 'image']:
                result[key] = value
        
        return result
    
    def convert_to_dataset(self,
                          input_file: str,
                          output_path: str,
                          user_query: Optional[str] = None,
                          system_prompt_file: Optional[str] = None,
                          replace_score: bool = True,
                          sample_only: bool = False,
                          sample_size: int = 10) -> DatasetDict:
        
        print("="*60)
        print("Stage 1: Converting to SFT format")
        print("="*60)
        
        system_prompt = None
        if system_prompt_file:
            system_prompt = self.load_system_prompt(system_prompt_file)
            if not system_prompt:
                print("ℹ Continuing without system prompt")
        else:
            print("ℹ No system prompt file provided")
        
        print(f"\nReading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        print(f"Total data items: {len(data)}")
        
        if sample_only:
            data = data[:sample_size]
            print(f"Sample mode: Processing first {len(data)} items only")
        
        sft_data = []
        skipped_count = 0
        
        print("\nStarting SFT format conversion...")
        for i, item in enumerate(tqdm(data, desc="Converting to SFT")):
            try:
                converted_item = self.convert_to_sft_format(
                    item,
                    user_query=user_query,
                    system_prompt=system_prompt,
                    replace_score=replace_score
                )
                
                if converted_item:
                    sft_data.append(converted_item)
                else:
                    skipped_count += 1
                    
            except Exception as e:
                print(f"Error processing item {i + 1}: {str(e)}")
                skipped_count += 1
                continue
        
        print(f"\nSFT conversion complete: {len(sft_data)} successful, {skipped_count} skipped")
        
        print("\n" + "="*60)
        print("Stage 2: Converting to HuggingFace Dataset")
        print("="*60)
        
        all_fields = set(['image', 'problem', 'solution'])
        for item in sft_data:
            all_fields.update(item.keys())
        all_fields.discard('messages')
        all_fields.discard('images')
        
        print(f"Detected fields: {list(all_fields)}")
        
        converted_data = {field: [] for field in all_fields}
        
        print("\nConverting to Dataset format...")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.process_dataset_item, item) for item in sft_data]
            
            for future in tqdm(futures, desc="Processing to Dataset"):
                result = future.result()
                for field in all_fields:
                    if field in result:
                        converted_data[field].append(result[field])
                    else:
                        converted_data[field].append(None)
        
        print("\nCreating HuggingFace Dataset...")
        features_dict = {
            "image": Sequence(HFImage()),
            "problem": Value("string"),
            "solution": Value("string"),
        }
        
        for field in all_fields:
            if field not in features_dict:
                sample_value = next((v for v in converted_data[field] if v is not None), None)
                if sample_value is not None:
                    if isinstance(sample_value, str):
                        features_dict[field] = Value("string")
                    elif isinstance(sample_value, (int, float)):
                        features_dict[field] = Value("float32")
                    elif isinstance(sample_value, bool):
                        features_dict[field] = Value("bool")
                    else:
                        features_dict[field] = Value("string")
                else:
                    features_dict[field] = Value("string")
        
        features = Features(features_dict)
        
        dataset = Dataset.from_dict(converted_data, features=features)
        dataset_dict = DatasetDict({'train': dataset})
        
        print(f"\nSaving Dataset to: {output_path}")
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
        
        dataset_dict.save_to_disk(
            output_path,
            num_proc=min(4, self.num_workers),
            num_shards={'train': 1}
        )
        
        print(f"\n{'='*60}")
        print("Conversion Complete Statistics")
        print(f"{'='*60}")
        print(f"Final Dataset samples: {len(dataset)}")
        print(f"Save path: {output_path}")
        print(f"Field list: {list(all_fields)}")
        if system_prompt:
            print(f"System prompt used: Yes")
        else:
            print(f"System prompt used: No")
        
        return dataset_dict
    
    def verify_dataset(self, dataset_path: str, sample_size: int = 3):
        from datasets import load_from_disk
        
        print(f"\n{'='*60}")
        print("Dataset Verification")
        print(f"{'='*60}")
        
        dataset = load_from_disk(dataset_path)
        train_dataset = dataset['train']
        
        print(f"Dataset features: {train_dataset.features}")
        print(f"Total samples: {len(train_dataset)}")
        
        print("\nFirst few samples:")
        for i in range(min(sample_size, len(train_dataset))):
            sample = train_dataset[i]
            print(f"\n--- Sample {i+1} ---")
            
            for key in sample.keys():
                if key == 'image':
                    print(f"Image: {'[Image data]' if sample['image'] else '[No image]'}")
                elif key in ['problem', 'solution']:
                    content = sample[key]
                    if content:
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"{key.capitalize()}: {preview}")
                    else:
                        print(f"{key.capitalize()}: None")
                else:
                    print(f"{key}: {sample[key]}")

def main():
    parser = argparse.ArgumentParser(description='Complete MLLM data conversion pipeline: from raw data to HuggingFace Dataset')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output Dataset path')
    parser.add_argument('--user_query', type=str, help='Unified user query (optional)')
    parser.add_argument('--system_prompt_file', type=str, help='System prompt txt file path')
    parser.add_argument('--no_replace_score', action='store_true', help='Do not replace scores (keep original scores)')
    parser.add_argument('--sample', action='store_true', help='Process sample data only')
    parser.add_argument('--sample_size', type=int, default=10, help='Sample size')
    parser.add_argument('--keep_fields', type=str, nargs='*', help='Additional fields to keep')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel processing threads')
    parser.add_argument('--verify', action='store_true', help='Verify the generated Dataset')
    
    args = parser.parse_args()
    
    converter = CompleteMLLMDatasetConverter(
        keep_fields=args.keep_fields,
        num_workers=args.num_workers
    )
    
    dataset = converter.convert_to_dataset(
        input_file=args.input,
        output_path=args.output,
        user_query=args.user_query,
        system_prompt_file=args.system_prompt_file,
        replace_score=not args.no_replace_score,
        sample_only=args.sample,
        sample_size=args.sample_size
    )
    
    if args.verify:
        converter.verify_dataset(args.output)
    
    print("\n✅ Conversion complete!")

if __name__ == "__main__":
    main()
