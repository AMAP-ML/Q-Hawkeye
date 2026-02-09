import json
import os
import random
import argparse
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
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
        for _ in range(3):  # 压缩3次
            buffer = BytesIO()
            compressed_image.save(buffer, 'JPEG', quality=self.jpeg_quality)
            buffer.seek(0)
            compressed_image = Image.open(buffer)
        
        return compressed_image
    
    def degrade_image_inplace(self, image_path, degradation_type):
        """
        直接在原图路径上进行退化处理（覆盖原文件）
        
        Args:
            image_path: 图像路径
            degradation_type: 退化类型
        
        Returns:
            bool: 处理是否成功
        """
        try:
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
            
            # 直接覆盖原文件
            if degradation_type == 'jpeg':
                degraded_image.save(image_path, 'JPEG', quality=self.jpeg_quality)
            else:
                degraded_image.save(image_path, 'JPEG', quality=95)
            
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False


def apply_second_degradation(input_json_path, output_json_path=None,
                            noise_std=25, blur_radius=3.0, 
                            jpeg_quality=40, darken_factor=0.6,
                            backup_images=False):
    """
    对JSON中的图像应用第二次退化处理
    
    Args:
        input_json_path: 输入JSON文件路径
        output_json_path: 输出JSON文件路径（可选，默认覆盖原文件）
        noise_std: 高斯噪声标准差
        blur_radius: 高斯模糊半径
        jpeg_quality: JPEG压缩质量
        darken_factor: 亮度因子
        backup_images: 是否备份原始图像
    """
    
    # 读取JSON文件
    print(f"Reading JSON from: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化退化处理器
    degrader = ImageDegradation(
        noise_std=noise_std,
        blur_radius=blur_radius,
        jpeg_quality=jpeg_quality,
        darken_factor=darken_factor
    )
    
    # 定义所有退化类型
    all_degradation_types = ['noise', 'blur', 'jpeg', 'darken']
    
    print(f"Processing {len(data)} samples...")
    print(f"Degradation parameters:")
    print(f"  - Noise std: {noise_std}")
    print(f"  - Blur radius: {blur_radius}")
    print(f"  - JPEG quality: {jpeg_quality}")
    print(f"  - Darken factor: {darken_factor}")
    print(f"  - Backup images: {backup_images}")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for item in tqdm(data):
        # 检查是否有退化类型字段
        if 'degradation_type' not in item:
            print(f"Warning: No degradation_type field for item {item.get('id', 'unknown')}")
            skip_count += 1
            continue
        
        first_degradation = item['degradation_type']
        
        # 如果没有原始退化类型，跳过
        if first_degradation is None:
            skip_count += 1
            continue
        
        # 获取可用的退化类型（排除已使用的）
        available_types = [t for t in all_degradation_types if t != first_degradation]
        
        if not available_types:
            print(f"Warning: No available degradation types for item {item.get('id', 'unknown')}")
            skip_count += 1
            continue
        
        # 随机选择第二种退化类型
        second_degradation = random.choice(available_types)
        
        # 获取图像路径
        if 'images' not in item or not item['images']:
            print(f"Warning: No images field for item {item.get('id', 'unknown')}")
            skip_count += 1
            continue
        
        image_path = item['images'][0]
        
        # 检查图像是否存在
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            error_count += 1
            continue
        
        # 备份图像（如果需要）
        if backup_images:
            backup_path = image_path.replace('.jpg', '_backup.jpg')
            try:
                image = Image.open(image_path)
                image.save(backup_path)
            except Exception as e:
                print(f"Failed to backup {image_path}: {str(e)}")
        
        # 应用第二次退化
        if degrader.degrade_image_inplace(image_path, second_degradation):
            # 更新JSON中的退化信息
            # 记录两次退化类型
            item['first_degradation'] = first_degradation
            item['second_degradation'] = second_degradation
            # 更新degradation_type为组合形式
            item['degradation_type'] = f"{first_degradation}+{second_degradation}"
            success_count += 1
        else:
            error_count += 1
    
    # 保存更新后的JSON
    if output_json_path is None:
        output_json_path = input_json_path
    
    print(f"\nSaving updated JSON to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Total samples: {len(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a second degradation on already-degraded images.")
    parser.add_argument("--input_json", required=True, help="Input JSON path to update.")
    parser.add_argument(
        "--output_json",
        default=None,
        help="Output JSON path. If omitted, overwrite --input_json.",
    )
    parser.add_argument("--noise_std", type=float, default=65, help="Gaussian noise std for 2nd degradation.")
    parser.add_argument("--blur_radius", type=float, default=4.0, help="Gaussian blur radius for 2nd degradation.")
    parser.add_argument("--jpeg_quality", type=int, default=5, help="JPEG quality for 2nd degradation.")
    parser.add_argument("--darken_factor", type=float, default=0.5, help="Brightness factor for 2nd degradation.")
    parser.add_argument(
        "--backup_images",
        action="store_true",
        help="Backup each image before overwriting it.",
    )

    args = parser.parse_args()

    apply_second_degradation(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        noise_std=args.noise_std,
        blur_radius=args.blur_radius,
        jpeg_quality=args.jpeg_quality,
        darken_factor=args.darken_factor,
        backup_images=args.backup_images,
    )
