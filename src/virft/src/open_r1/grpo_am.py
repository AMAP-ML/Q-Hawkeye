# Copyright 2025 The HuggingFace Team.
# Apache 2.0 License

import os
import re
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from datasets import DatasetDict
import ml_tracker.integration.transformers

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json
import math
from typing import List, Optional as Opt, Dict


# ================== Script & Config ==================


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy_correctness', 'format', 'repetition'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy_correctness", "format", "repetition"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy_correctness', 'format', 'repetition'"},
    )
    max_pixels: Opt[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Opt[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    preprocessing_num_workers: Opt[int] = field(
        default=1,
        metadata={"help": "Number of workers for the preprocessing"},
    )


@dataclass
class GRPOConfig(TRLGRPOConfig):
    vllm_device: str = field(default="auto", metadata={"help": "Device for vLLM backend."})
    where: str = field(default="local", metadata={"help": "Determine where to run, 'local' or 'nebula'."})
    curriculum_learning: bool = field(
        default=False,
        metadata={"help": "Enable curriculum learning sampler (easy->hard order per epoch)."},
    )
    Curricuum_Learning: Optional[bool] = field(
        default=None,
        metadata={"help": "Alias for curriculum_learning; when True enables curriculum sampler."},
    )

    enable_perception_loss: bool = field(
        default=False,
        metadata={"help": "Enable Perception Loss objective for perceptual quality awareness."},
    )
    perception_loss_gamma: float = field(
        default=0.0,
        metadata={"help": "Weight gamma for KL divergence in Perception Loss."},
    )
    perception_loss_eta1: float = field(
        default=0.0,
        metadata={"help": "Weight eta1 for entropy on original image."},
    )
    perception_loss_eta2: float = field(
        default=0.0,
        metadata={"help": "Weight eta2 for entropy on degraded image."},
    )
    perception_loss_kl_zero_epoch: int = field(
        default=0,
        metadata={"help": "Epoch index after which KL weight is set to 0 (cosine annealing)."},
    )

    use_ua_grpo: bool = field(
        default=True,
        metadata={"help": "Enable UA-GRPO (Uncertainty-Aware GRPO) with advantage scaling based on <answer> score variance."},
    )
    ua_alpha: float = field(
        default=0.2,
        metadata={"help": "Scaling strength alpha in exp(-alpha * uncertainty). Larger alpha -> stronger down-scaling."},
    )
    ua_score_min: float = field(
        default=1.0,
        metadata={"help": "Minimum possible quality score in <answer> for IQA."},
    )
    ua_score_max: float = field(
        default=5.0,
        metadata={"help": "Maximum possible quality score in <answer> for IQA."},
    )


# ================== 各种 Reward & Utils ==================


def clean_text(text):
    return re.sub(r"[^\w\s]", "", text)


def repetition_reward(completions, **kwargs):
    thre = 5
    scale = 0.25
    punishments = []
    for completion in completions:
        content = completion[0]["content"]

        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            answer_content = match.group(1).strip()
        else:
            answer_content = content.strip()

        cleaned_answer = clean_text(answer_content)

        word_tokens = cleaned_answer.split(" ")
        word_freq = Counter(word_tokens)
        word_repeat_count = sum(v for v in word_freq.values() if v > thre)
        word_punishment = -word_repeat_count * scale

        char_tokens = list(cleaned_answer)
        char_freq = Counter(char_tokens)
        char_repeat_count = sum(v for v in char_freq.values() if v > thre)
        char_punishment = -char_repeat_count * scale

        total_punishment = word_punishment + char_punishment

        punishments.append(max(-2.0, total_punishment))

    return punishments


_ANSWER_TAG_PATTERN = r"<answer>\s*([\d.]+)\s*</answer>"


def _extract_score(s: str) -> Opt[float]:
    """从带有 <answer>3.45</answer> 的字符串中提取分数；失败返回 None。"""
    if not isinstance(s, str):
        return None
    m = re.search(_ANSWER_TAG_PATTERN, s, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (ValueError, AttributeError):
        return None


def score_reward(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    tau: float = 0.30,
    min_score: float = 1.0,
    max_score: float = 5.0,
    verbose: bool = False,
    **kwargs,
) -> List[float]:
    """
    简化版指数奖励函数：
        r = exp(-|pred - gt| / tau)
    """

    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    contents = [c[0]["content"] for c in completions]
    rewards = []

    for i, (content, gt_str) in enumerate(zip(contents, solution)):
        try:
            pred = _extract_score(content)
            gt = _extract_score(gt_str)

            if pred is None or gt is None:
                if verbose:
                    print(f"[Sample {i}] Parse failed")
                rewards.append(0.0)
                continue

            pred = min(max(pred, min_score), max_score)
            gt = min(max(gt, min_score), max_score)

            err = abs(pred - gt)
            r = math.exp(-err / tau)
            r = max(0.0, min(1.0, r))

            rewards.append(r)

            if verbose:
                print(
                    f"[Sample {i}] GT={gt:.2f}, Pred={pred:.2f}, "
                    f"Err={err:.4f}, Reward={r:.4f}"
                )

        except Exception as e:
            if verbose:
                print(f"[Sample {i}] Exception: {e}")
            rewards.append(0.0)

    return rewards


def format_reward(completions, **kwargs):
    """
    格式奖励：匹配 <think>...</think><answer>x.xx</answer>
    """
    pattern = (
        r"^\s*<think>\s*"
        r".+?"
        r"</think>\s*"
        r"<answer>\s*"
        r"[\d\.]+?"
        r"\s*</answer>\s*$"
    )

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [0.8 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy_correctness": score_reward,
    "format": format_reward,
    "repetition": repetition_reward,
}

SYSTEM_PROMPT = """"""


# ================== main ==================


def main(script_args, training_args, model_args):
    print("\n===== Script Args =====")
    print(script_args)

    print("\n===== Training Args =====")
    print(training_args)

    print("\n===== Model Args =====")
    print(model_args)

    # Get reward functions
    script_args.reward_funcs = ["accuracy_correctness", "format", "repetition"]
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # 加载 HF dataset
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)

    # 统一 train / test split 名
    if not hasattr(script_args, "dataset_train_split"):
        script_args.dataset_train_split = "train"
        script_args.dataset_test_split = "test" if "test" in dataset else script_args.dataset_train_split

    # =============== 构造 prompt（支持 image_orig / image） ===============

    # 既兼容旧字段 image，又支持新字段 image_orig / image_deg
    features = dataset[script_args.dataset_train_split].features
    has_image_orig = "image_orig" in features
    has_image = "image" in features

    if has_image_orig or has_image:
        print("found image field in dataset:", "image_orig" if has_image_orig else "image")

        def make_conversation_image(example):
            # 使用原图数量来放 <image> 占位
            imgs = example["image_orig"] if has_image_orig else example["image"]
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"} for _ in imgs
                        ] + [
                            {"type": "text", "text": example["problem"]},
                        ],
                    },
                ],
            }

        # 注意：map 之后，原来的字段（image_orig / image_deg / solution 等）都会保留
        dataset = dataset.map(make_conversation_image)
    else:
        print("no image field found in dataset, using text-only prompts")

        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }

        dataset = dataset.map(make_conversation)
        # 如果原来有 messages 字段，可以删掉
        if "messages" in dataset[script_args.dataset_train_split].column_names:
            dataset = dataset.remove_columns("messages")

    # =============== 选择 Trainer 类 ===============
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using trainer class:", trainer_cls)

    # temp checker
    print("training_args.max_grad_norm =", training_args.max_grad_norm)
    assert training_args.max_grad_norm > 0, "training_args.max_grad_norm must > 0!"

    # =============== 初始化 Trainer（Perception Loss + UA-GRPO 配置在 training_args/GRPOConfig 里） ===============
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
