import json
import os
import random
import argparse
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path
from tqdm import tqdm

class ImageDegradation:
    """图像退化处理类"""
    
    def __init__(self, noise_std=25, blur_radius=3.0, jpeg_quality=40, darken_factor=0.6):
        """
        初始化退化参数
        
        Args:
            noise_std: 高斯噪声标准差 (推荐范围: 10-40)
            blur_radius: 高斯模糊半径 (推荐范围: 1.0-5.0)
            jpeg_quality: JPEG压缩质量 (推荐范围: 20-60, 越小退化越严重)
            darken_factor: 亮度因子 (推荐范围: 0.4-0.8, 越小越暗)
        """
        self.noise_std = noise_std
        self.blur_radius = blur_radius
        self.jpeg_quality = jpeg_quality
        self.darken_factor = darken_factor
    
    def add_noise(self, image):
        """添加高斯噪声"""
        img_array = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, self.noise_std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def add_blur(self, image):
        """添加高斯模糊"""
        return image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
    
    
    def darken_image(self, image):
        """降低图像亮度"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(self.darken_factor)
    
    def add_jpeg_compression(self, image):
        """多次压缩以增强伪影效果"""
        from io import BytesIO
        
        # 多次压缩以累积伪影
        compressed_image = image
        for _ in range(3):  # 压缩2次
            buffer = BytesIO()
            compressed_image.save(buffer, 'JPEG', quality=self.jpeg_quality)
            buffer.seek(0)
            compressed_image = Image.open(buffer)
        
        return compressed_image


    def degrade_image(self, image_path, degradation_type, output_path):
        image = Image.open(image_path).convert('RGB')
        
        if degradation_type == 'noise':
            degraded_image = self.add_noise(image)
        elif degradation_type == 'blur':
            degraded_image = self.add_blur(image)
        elif degradation_type == 'jpeg':
            degraded_image = self.add_jpeg_compression(image)
        elif degradation_type == 'darken':
            degraded_image = self.darken_image(image)
        else:
            raise ValueError(f"Unknown degradation type: {degradation_type}")
        
        # 对于JPEG退化，这里保存时使用原始压缩质量
        if degradation_type == 'jpeg':
            degraded_image.save(output_path, 'JPEG', quality=self.jpeg_quality)
        else:
            degraded_image.save(output_path, 'JPEG', quality=95)
        
        return output_path



def process_dataset(input_json_path, output_json_path, degraded_images_dir,
                   noise_std=25, blur_radius=3.0, jpeg_quality=40, darken_factor=0.6,
                   degradation_types=['noise', 'blur', 'jpeg', 'darken']):
    """
    处理数据集,生成退化图像和新的JSON文件
    
    Args:
        input_json_path: 输入JSON文件路径
        output_json_path: 输出JSON文件路径
        degraded_images_dir: 退化图像保存目录
        noise_std: 高斯噪声标准差 (推荐: 10-40)
        blur_radius: 高斯模糊半径 (推荐: 1.0-5.0)
        jpeg_quality: JPEG压缩质量 (推荐: 20-60)
        darken_factor: 亮度因子 (推荐: 0.4-0.8)
        degradation_types: 可选的退化类型列表
    """
    # 创建输出目录
    os.makedirs(degraded_images_dir, exist_ok=True)
    
    # 读取原始JSON
    print(f"Reading input JSON from: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化退化处理器
    degrader = ImageDegradation(
        noise_std=noise_std,
        blur_radius=blur_radius,
        jpeg_quality=jpeg_quality,
        darken_factor=darken_factor
    )
    
    # 新的数据列表
    new_data = []
    
    print(f"Processing {len(data)} images...")
    print(f"Degradation parameters:")
    print(f"  - Noise std: {noise_std}")
    print(f"  - Blur radius: {blur_radius}")
    print(f"  - JPEG quality: {jpeg_quality}")
    print(f"  - Darken factor: {darken_factor}")
    
    for idx, item in enumerate(tqdm(data)):
        # 生成group_id
        group_id = f"group_{idx:06d}"
        
        # 处理原始样本,添加新字段
        original_item = item.copy()
        original_item['degradation_type'] = None
        original_item['degradation_level'] = None
        original_item['group_id'] = group_id
        original_item['is_degraded'] = False
        new_data.append(original_item)
        
        # 获取原始图像路径
        original_image_path = item['images'][0]
        
        # 检查图像是否存在
        if not os.path.exists(original_image_path):
            print(f"Warning: Image not found: {original_image_path}")
            continue
        
        # 随机选择一种退化类型
        degradation_type = random.choice(degradation_types)
        
        # 生成退化图像文件名
        image_filename = os.path.basename(original_image_path)
        image_name, image_ext = os.path.splitext(image_filename)
        degraded_filename = f"{image_name}_{degradation_type}.jpg"
        degraded_image_path = os.path.join(degraded_images_dir, degraded_filename)
        
        try:
            # 应用退化处理
            degrader.degrade_image(
                original_image_path, 
                degradation_type,
                degraded_image_path
            )
            
            # 创建退化样本
            degraded_item = item.copy()
            
            # 更新ID
            degraded_item['id'] = f"{item['id']}_{degradation_type}"
            
            # 更新图像路径
            degraded_item['images'] = [degraded_image_path]
            
            # 添加新字段
            degraded_item['degradation_type'] = degradation_type
            degraded_item['degradation_level'] = None  # 不再使用level字段
            degraded_item['group_id'] = group_id
            degraded_item['is_degraded'] = True
            degraded_item['original_images'] = [original_image_path]
            
            new_data.append(degraded_item)
            
        except Exception as e:
            print(f"Error processing image {original_image_path}: {str(e)}")
            continue
    
    # 保存新的JSON文件
    print(f"Saving output JSON to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete!")
    print(f"Original samples: {len(data)}")
    print(f"Total samples (original + degraded): {len(new_data)}")
    print(f"Degraded images saved to: {degraded_images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate degraded images and a paired JSON dataset (original + degraded).",
    )
    parser.add_argument("--input_json", required=True, help="Input JSON path with original samples.")
    parser.add_argument("--output_json", required=True, help="Output JSON path for (original + degraded) samples.")
    parser.add_argument(
        "--degraded_images_dir",
        required=True,
        help="Directory to save degraded images.",
    )
    parser.add_argument("--noise_std", type=float, default=45, help="Gaussian noise std (higher = stronger).")
    parser.add_argument("--blur_radius", type=float, default=2.0, help="Gaussian blur radius (higher = blurrier).")
    parser.add_argument("--jpeg_quality", type=int, default=5, help="JPEG quality (lower = more artifacts).")
    parser.add_argument("--darken_factor", type=float, default=0.6, help="Brightness factor (lower = darker).")
    parser.add_argument(
        "--degradation_types",
        nargs="+",
        default=["noise", "blur", "jpeg", "darken"],
        help="Candidate degradation types to sample from.",
    )

    args = parser.parse_args()

    process_dataset(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        degraded_images_dir=args.degraded_images_dir,
        noise_std=args.noise_std,
        blur_radius=args.blur_radius,
        jpeg_quality=args.jpeg_quality,
        darken_factor=args.darken_factor,
        degradation_types=args.degradation_types,
    )
