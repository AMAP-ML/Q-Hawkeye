import os
import json
import base64
import time
import argparse
from openai import OpenAI
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API配置
API_KEY = os.environ.get("VLM_API_KEY", "<YOUR_API_KEY>")
BASE_URL = os.environ.get("VLM_BASE_URL", "https://ai-llm-gateway.amap.com/v1/")

# 初始化OpenAI客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 文件锁
file_lock = threading.Lock()

def encode_image(image_path: str) -> Optional[str]:
    """编码图像为base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def create_comparison_prompt(degraded_path: str, original_path: str) -> tuple:
    """创建对比两张图像的prompt"""
    
    system_message = """You are an expert at comparing image quality. Your task is to determine if two images appear virtually identical to human perception, despite one being a processed/degraded version of the other."""
    
    # 编码两张图像
    degraded_base64 = encode_image(degraded_path)
    original_base64 = encode_image(original_path)
    
    if not degraded_base64 or not original_base64:
        raise ValueError(f"Failed to encode images: {degraded_path} or {original_path}")
    
    content = [
        {
            "type": "text",
            "text": """Compare these two images carefully:

Image 1 (Degraded version):"""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{degraded_base64}"
            }
        },
        {
            "type": "text",
            "text": "\n\nImage 2 (Original version):"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{original_base64}"
            }
        },
        {
            "type": "text",
            "text": """

Question: Are these two images virtually indistinguishable to human perception? 
Consider aspects like:
- Overall visual quality
- Sharpness and detail
- Color accuracy
- Noise levels
- Compression artifacts
- Any other visible differences

If the differences are so subtle that most people wouldn't notice them in normal viewing conditions, answer 'yes'.
If there are noticeable quality differences that would be apparent to most viewers, answer 'no'.

IMPORTANT: Reply with ONLY 'yes' or 'no' (lowercase, no other text)."""
        }
    ]
    
    return system_message, content

def compare_single_sample(sample: Dict, max_retries: int = 3) -> Optional[bool]:
    """
    对单个样本进行对比评估
    返回True表示几乎看不出区别，False表示有明显区别，None表示处理失败
    """
    
    # 获取图像路径
    degraded_path = sample['images'][0]
    original_path = sample['original_images'][0]
    
    logger.info(f"Comparing images for sample: {sample['id']}")
    
    for attempt in range(max_retries):
        try:
            # 创建prompt
            system_message, content = create_comparison_prompt(degraded_path, original_path)
            
            # 调用API
            completion = client.chat.completions.create(
                model="gpt4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": content}
                ],
                max_tokens=10,
                temperature=0.3  # 低温度以获得更一致的结果
            )
            
            response = completion.choices[0].message.content.strip().lower()
            
            # 解析响应
            if response == 'yes':
                return True
            elif response == 'no':
                return False
            else:
                logger.warning(f"Unexpected response: {response}. Retrying...")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get valid response for {sample['id']}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error processing {sample['id']}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            return None
    
    return None

def process_json_with_comparison(
    input_json_path: str,
    output_json_path: str,
    max_workers: int = 5,
    checkpoint_interval: int = 20
):
    """
    处理JSON文件，筛选出原图和退化图几乎无差别的样本
    """
    
    # 读取输入JSON
    logger.info(f"Loading data from {input_json_path}")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # 筛选需要处理的样本（is_degraded=true）
    degraded_samples = [sample for sample in data if sample.get('is_degraded', False)]
    logger.info(f"Found {len(degraded_samples)} degraded samples to process")
    
    # 检查是否有checkpoint
    checkpoint_path = Path(output_json_path).with_suffix('.checkpoint.json')
    processed_ids = set()
    indistinguishable_samples = []
    
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                processed_ids = set(checkpoint_data['processed_ids'])
                indistinguishable_samples = checkpoint_data['indistinguishable_samples']
                logger.info(f"Resumed from checkpoint: {len(processed_ids)} already processed")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
    
    # 过滤已处理的样本
    remaining_samples = [s for s in degraded_samples if s['id'] not in processed_ids]
    
    if not remaining_samples:
        logger.info("All samples already processed!")
        save_results(indistinguishable_samples, output_json_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        return indistinguishable_samples
    
    logger.info(f"Processing {len(remaining_samples)} remaining samples...")
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_sample = {
            executor.submit(compare_single_sample, sample): sample 
            for sample in remaining_samples
        }
        
        # 处理结果
        with tqdm(total=len(remaining_samples), desc="Comparing images") as pbar:
            completed_count = 0
            
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                
                try:
                    result = future.result(timeout=30)
                    
                    # 更新已处理集合
                    processed_ids.add(sample['id'])
                    
                    # 如果判定为几乎无差别，添加到结果中
                    if result is True:
                        indistinguishable_samples.append(sample)
                        logger.info(f"Sample {sample['id']}: Indistinguishable (YES)")
                        pbar.set_postfix({'status': 'YES', 'total_yes': len(indistinguishable_samples)})
                    elif result is False:
                        logger.info(f"Sample {sample['id']}: Distinguishable (NO)")
                        pbar.set_postfix({'status': 'NO', 'total_yes': len(indistinguishable_samples)})
                    else:
                        logger.warning(f"Sample {sample['id']}: Processing failed")
                        pbar.set_postfix({'status': 'FAILED', 'total_yes': len(indistinguishable_samples)})
                    
                    completed_count += 1
                    pbar.update(1)
                    
                    # 定期保存checkpoint
                    if completed_count % checkpoint_interval == 0:
                        save_checkpoint(checkpoint_path, processed_ids, indistinguishable_samples)
                        
                except Exception as e:
                    logger.error(f"Error processing sample {sample['id']}: {e}")
                    processed_ids.add(sample['id'])
                    pbar.update(1)
    
    # 保存最终结果
    save_results(indistinguishable_samples, output_json_path)
    
    # 删除checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file")
    
    # 输出统计信息
    print_statistics(degraded_samples, indistinguishable_samples)
    
    return indistinguishable_samples

def save_checkpoint(checkpoint_path: Path, processed_ids: set, indistinguishable_samples: List[Dict]):
    """保存checkpoint"""
    checkpoint_data = {
        'processed_ids': list(processed_ids),
        'indistinguishable_samples': indistinguishable_samples,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with file_lock:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    logger.debug(f"Checkpoint saved: {len(processed_ids)} processed, {len(indistinguishable_samples)} indistinguishable")

def save_results(samples: List[Dict], output_path: str):
    """保存最终结果"""
    with file_lock:
        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")

def print_statistics(all_degraded_samples: List[Dict], indistinguishable_samples: List[Dict]):
    """打印统计信息"""
    print("\n" + "="*60)
    print("Processing Complete - Statistics")
    print("="*60)
    print(f"Total degraded samples: {len(all_degraded_samples)}")
    print(f"Indistinguishable samples (YES): {len(indistinguishable_samples)}")
    print(f"Distinguishable samples (NO): {len(all_degraded_samples) - len(indistinguishable_samples)}")
    print(f"Percentage indistinguishable: {len(indistinguishable_samples)/len(all_degraded_samples)*100:.1f}%")
    
    # 按退化类型统计
    degradation_stats = {}
    for sample in indistinguishable_samples:
        deg_type = sample.get('degradation_type', 'unknown')
        degradation_stats[deg_type] = degradation_stats.get(deg_type, 0) + 1
    
    if degradation_stats:
        print("\nIndistinguishable samples by degradation type:")
        for deg_type, count in sorted(degradation_stats.items()):
            print(f"  {deg_type}: {count}")
    
    print("="*60)

def check_status(output_json_path: str):
    """检查处理状态"""
    output_path = Path(output_json_path)
    checkpoint_path = output_path.with_suffix('.checkpoint.json')
    
    print(f"\nChecking status for: {output_json_path}")
    print("="*60)
    
    if output_path.exists():
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Output file exists with {len(results)} samples")
    else:
        print("Output file does not exist yet")
    
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        print(f"Checkpoint found:")
        print(f"  - Processed: {len(checkpoint['processed_ids'])} samples")
        print(f"  - Indistinguishable: {len(checkpoint['indistinguishable_samples'])} samples")
        print(f"  - Last update: {checkpoint.get('timestamp', 'Unknown')}")
    else:
        print("No checkpoint found")
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter degraded samples whose degraded vs original images are virtually indistinguishable.",
    )
    parser.add_argument(
        "--mode",
        choices=["process", "status"],
        default="process",
        help="Run mode: process to generate filtered JSON, status to inspect progress.",
    )
    parser.add_argument(
        "--input_json",
        help="Input JSON path (output of Degradation.py). Required when --mode=process.",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Output JSON path for indistinguishable samples. Also used for --mode=status.",
    )
    parser.add_argument("--max_workers", type=int, default=3, help="Parallel workers for API calls.")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="Save checkpoint every N samples.")

    args = parser.parse_args()

    if args.mode == "process":
        if not args.input_json:
            raise SystemExit("--input_json is required when --mode=process")
        process_json_with_comparison(
            input_json_path=args.input_json,
            output_json_path=args.output_json,
            max_workers=args.max_workers,
            checkpoint_interval=args.checkpoint_interval,
        )
    else:
        check_status(args.output_json)
