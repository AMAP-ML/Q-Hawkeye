<div align="center">
<h3>

Q-Hawkeye: Reliable Visual Policy Optimization for Image Quality Assessment

Wulin Xie, Rui Dai, Ruidong Ding, Kaikui Liu, Xiangxiang Chu, Xinwen Hou, Jie Wen
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2601.10477-b31b1b.svg)](https://arxiv.org/abs/2601.22920)

## 🚩 Updates
- 01.28 Released our paper.
- 01.28 Released the code of Q-Hawkeye.

## 🔥 Introduction
In this paper, we propose Q-Hawkeye, a GRPO-based framework for reliable visual policy optimization in image quality assessment. Built on Qwen2.5-VL-7B, Q-Hawkeye reshapes the RL learning signal from two complementary perspectives: an Uncertainty-Aware Dynamic Optimization strategy that adaptively reweights per image updates based on score variance across rollouts, and a Perception-Aware Optimization module that enforces consistent distributional differences between original and degraded images via an implicit perception loss with double entropy regularization. Extensive experiments on eight IQA benchmarks further demonstrate the effectiveness of the proposed modules,  and show that Q-Hawkeye consistently outperforms existing state-of-the-art methods in both single- and multi-dataset settings, with clear gains in average PLCC/SRCC metrics and improves the model's robustness on challenging out-of-distribution distortions.
<p align="center">
  <img src="assets/Framework.pdf">
</p>


## 🔧 Dependencies and Installation
```bash
pip install -r requirements.txt
```

## ⚡ Quick Inference

```
python inference.py
```


## 📖 Dataset Preparation for Training

#### Training Dataset Preparation
Download meta files from [Data-DeQA-Score](https://huggingface.co/datasets/zhiyuanyou/Data-DeQA-Score/tree/main) and the source images from the [KONIQ](https://database.mmsp-kn.de/koniq-10k-database.html) dataset.

Your JSON data should follow this format:
```json
[
  {
    "id": "sample_001",
    "images": ["/path/to/image.jpg"],
    "gt_score": 3.75
  }
]
```

#### Degradation Dataset Preparation
Generate degraded images for Perception Loss training through a four-stage pipeline:

**Stage 1: Initial Degradation**
```bash
cd src/Dataset/Degradation_Dataset/
python Degradation.py \
    --input_json /path/to/original_data.json \
    --output_json /path/to/degraded_data.json \
    --degraded_images_dir /path/to/degraded_images/
```
Applies one degradation type (`noise`, `blur`, `jpeg`, `darken`) to each image. Each original image generates one degraded variant paired by `group_id`.

**Stage 2: VLM-based Filtering**
```bash
python VLM_filter.py \
    --input_json /path/to/degraded_data.json \
    --output_json /path/to/vlm_filtered_data.json \
    --model_path /path/to/qwen2-vl-7b
```
Uses Qwen2-VL to automatically filter samples by comparing quality scores between original and degraded images.

**Stage 3: Human Verification**
```bash
python Human_filter.py \
    --input_json /path/to/vlm_filtered_data.json \
    --output_json /path/to/human_verified_data.json
```
Provides a GUI for manual review and verification of degraded samples.

**Stage 4: Second Degradation**
```bash
python Second_Degradation.py \
    --input_json /path/to/human_verified_data.json \
    --output_json /path/to/final_degraded_data.json \
    --noise_std 65 --blur_radius 4.0 --jpeg_quality 5 --darken_factor 0.5
```
Applies a second, different degradation type to increase difficulty. The `degradation_type` field is updated to combined form (e.g., `noise+blur`).

#### RL Dataset Preparation
Convert JSON to HuggingFace Dataset format:
```bash
cd src/Dataset/
python RL_Construction.py \
    --input /path/to/data.json \
    --output /path/to/hf_dataset \
    --keep_fields gt_score group_id degradation_type \
    --verify
```

This converts data into the format required by training scripts with fields: `image`, `problem`, `solution`, plus any additional fields specified in `--keep_fields`.

## 🚀 Training
```bash
cd src/scripts/
bash qw7b_local.sh
```

## Citation
```bibtex
@misc{xie2026qhawkeyereliablevisualpolicy,
      title={Q-Hawkeye: Reliable Visual Policy Optimization for Image Quality Assessment}, 
      author={Wulin Xie and Rui Dai and Ruidong Ding and Kaikui Liu and Xiangxiang Chu and Xinwen Hou and Jie Wen},
      year={2026},
      eprint={2601.22920},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.22920}, 
}
```


## ✏️ To Do List
- [x] Release inference code
- [x] Release training code
- [x] Release the paper

## Acknowledgement
We appreciate the releasing codes and data of [Visual-RFT](https://github.com/Liuziyu77/Visual-RFT), [Q-insight](https://github.com/bytedance/Q-Insight) and [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score).
