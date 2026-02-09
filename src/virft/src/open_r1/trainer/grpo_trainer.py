import os
import re
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import math

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from openlm_hub import repo_download
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb



RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]



def data_collator(features):
    return features


def _low_var_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
) -> torch.Tensor:
    """
    Low-variance KL form used in Perception_Loss:
      - clamp (ref_log_probs - log_probs) before exp to avoid overflow;
      - clamp the final kld for numerical stability.
    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
    kld = (kl.exp() - kl - 1).contiguous()
    return torch.clamp(kld, min=-10.0, max=10.0)


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):

        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")



        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )

            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                if args.where == 'nebula':
                    model = Qwen2VLForConditionalGeneration.from_pretrained(repo_download(model), **model_init_kwargs)
                else:
                    model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                if args.where == 'nebula':
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo_download(model), **model_init_kwargs)
                else:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                if args.where == 'nebula':
                    model = AriaForConditionalGeneration.from_pretrained(repo_download(model), **model_init_kwargs)
                else:
                    model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                if args.where == 'nebula':
                    model = AutoModelForCausalLM.from_pretrained(repo_download(model), **model_init_kwargs)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)


        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                if args.where == 'nebula':
                    self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(repo_download(model_id), **model_init_kwargs)
                else:
                    self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                if args.where == 'nebula':
                    self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo_download(model_id), **model_init_kwargs)
                else:
                    self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                if args.where == 'nebula':
                    self.ref_model = AriaForConditionalGeneration.from_pretrained(repo_download(model_id), **model_init_kwargs)
                else:
                    self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                if args.where == 'nebula':
                    self.ref_model = AutoModelForCausalLM.from_pretrained(repo_download(model_id), **model_init_kwargs)
                else:
                    self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:

            self.ref_model = create_reference_model(model)
        else:


            self.ref_model = None


        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                if args.where == 'nebula':
                    processing_class = AutoProcessor.from_pretrained(repo_download(model_id))
                else:
                    processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                if args.where == 'nebula':
                    processing_class = AutoTokenizer.from_pretrained(repo_download(model.config._name_or_path), padding_side="left")
                else:
                    processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id


        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                if args.where == 'nebula':
                    reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                        repo_download(reward_func), num_labels=1, **model_init_kwargs
                    )
                else:
                    reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                        reward_func, num_labels=1, **model_init_kwargs
                    )
        self.reward_funcs = reward_funcs


        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    if args.where == 'nebula':
                        reward_processing_class = AutoTokenizer.from_pretrained(repo_download(reward_func.config._name_or_path))
                    else:
                        reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token


                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes


        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True


        self._metrics = defaultdict(list)


        # Perception Loss configuration
        self.use_kl_prcp = getattr(args, "enable_perception_loss", False)
        self.use_aug_entropy_loss = getattr(args, "use_aug_entropy_loss", False)
        self.use_ori_entropy_loss = getattr(args, "use_ori_entropy_loss", False)

        self.prcp_gamma = float(getattr(args, "perception_loss_gamma", 0.0))
        self.prcp_eta1 = float(getattr(args, "perception_loss_eta1", 0.0))
        self.prcp_eta2 = float(getattr(args, "perception_loss_eta2", 0.0))
        self.prcp_gamma_base = self.prcp_gamma
        self.prcp_kl_zero_epoch = getattr(args, "perception_loss_kl_zero_epoch", 0)


        self.enable_prcp_loss = self.use_kl_prcp or self.use_aug_entropy_loss or self.use_ori_entropy_loss


        # UA-GRPO (Uncertainty-Aware GRPO) configuration
        self.use_ua_grpo = getattr(args, "use_ua_grpo", False)
        self.ua_alpha = float(getattr(args, "ua_alpha", 0.2))
        self.score_min = float(getattr(args, "ua_score_min", 1.0))
        self.score_max = float(getattr(args, "ua_score_max", 5.0))


        self.use_self_certainty = getattr(args, "use_self_certainty", False)
        self.self_certainty_coef = float(getattr(args, "self_certainty_coef", 1.0))
        self.self_certainty_only = getattr(args, "self_certainty_only", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )




        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):



        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "image_orig", "image_deg"]


    @staticmethod
    def _extract_score_from_text(text: str):
        """
        Parse a float score wrapped in <answer>...</answer> from the model output.

        Example:
            "... <answer>3.75</answer> ..." -> 3.75

        Returns:
            float or None if not found / parse failed.
        """
        if not isinstance(text, str):
            return None
        m = re.search(r"<answer>\s*([-+]?\d+(\.\d+)?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None


    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]

        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _get_per_token_self_certainty(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        """
        Compute per-token self-certainty from logits, following Intuitor-style definition:
            SCe = logsumexp(logits) - mean(logits)
        """
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits
        logits = logits[:, :-1, :]
        sce = torch.logsumexp(logits, dim=-1) - logits.mean(dim=-1)
        return sce

    def _get_sce_advantage(self, sce_tokens, completion_mask, num_prompts: int, std_norm_per_prompt=None):
        """
        Aggregate per-token self-certainty to sentence-level and normalize per GRPO group.
        Optionally use `std_norm_per_prompt` as a target shape for the normalized self-certainty.

        Args:
            sce_tokens: (B*G, L_all) per-token SCe (already trimmed to completion region length).
            completion_mask: (B*G, L_comp) mask for completion tokens.
            num_prompts: number of distinct prompts (B).
            std_norm_per_prompt: optional sequence / tensor of shape (B,)
                Normalized target values (e.g. in [0, 1]) for how "confident" the model is expected to be.

        Returns:
            sce_advantage: (B*G,) normalized self-certainty advantages.
        """
        lengths = completion_mask.sum(dim=1)
        sce_sum = (sce_tokens * completion_mask).sum(dim=1)
        sce_mean = torch.zeros_like(sce_sum)
        valid = lengths > 0
        if valid.any():
            denom = lengths[valid].to(dtype=sce_sum.dtype)
            sce_mean[valid] = sce_sum[valid] / denom

        G = self.num_generations
        sce_group = sce_mean.view(num_prompts, G)




        if std_norm_per_prompt is not None:
            if not isinstance(std_norm_per_prompt, torch.Tensor):
                std_norm_tensor = torch.tensor(
                    std_norm_per_prompt, device=sce_tokens.device, dtype=sce_tokens.dtype
                )
            else:
                std_norm_tensor = std_norm_per_prompt.to(device=sce_tokens.device, dtype=sce_tokens.dtype)


            group_mean = sce_group.mean(dim=1, keepdim=True)
            group_std = sce_group.std(dim=1, keepdim=True)
            sce_group_norm = (sce_group - group_mean) / (group_std + 1e-4)


            target_z = 2.0 * std_norm_tensor.view(-1, 1) - 1.0


            error = (sce_group_norm - target_z) ** 2
            adv = -error


            adv_mean = adv.mean(dim=1, keepdim=True)
            adv_std = adv.std(dim=1, keepdim=True)
            sce_group_adv = (adv - adv_mean) / (adv_std + 1e-4)
            return sce_group_adv.view(-1)


        group_mean = sce_group.mean(dim=1, keepdim=True)
        group_std = sce_group.std(dim=1, keepdim=True)
        sce_group_norm = (sce_group - group_mean) / (group_std + 1e-4)
        return sce_group_norm.view(-1)




    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        num_prompts = len(prompts)
        std_norm_per_prompt = None
        if "std_norm" in inputs[0]:
            std_norm_per_prompt = [example["std_norm"] for example in inputs]
        task_type = [...]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]



        if "image_orig" in inputs[0]:
            images_orig = [x["image_orig"] for x in inputs]
        else:
            images_orig = [x["image"] for x in inputs]


        has_deg = "image_deg" in inputs[0] and inputs[0]["image_deg"] is not None
        images_deg = [x["image_deg"] for x in inputs] if has_deg else None


        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images_orig,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]




        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]


        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)


        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)

        per_token_logps = per_token_logps[:, prompt_length - 1 :]


        sce_advantage = None
        if getattr(self, "use_self_certainty", False):



            with torch.no_grad():
                sce_tokens = self._get_per_token_self_certainty(
                    model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                )

            sce_tokens = sce_tokens[:, prompt_length - 1 :]
            sce_advantage = self._get_sce_advantage(sce_tokens, completion_mask, num_prompts, std_norm_per_prompt)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]


        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1


        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]


        device = self.accelerator.device
        if getattr(self, "use_ua_grpo", False):
            num_prompts = len(inputs)
            G = self.num_generations
            assert len(completions) == num_prompts * G, "num_completions != B * num_generations"

            pred_scores = []
            for comp in completions:
                text = comp[0]["content"] if is_conversational(inputs[0]) else comp
                val = self._extract_score_from_text(text)
                pred_scores.append(float("nan") if val is None else val)

            scores_tensor = torch.tensor(pred_scores, device=device, dtype=torch.float32)
            scores_tensor = scores_tensor.view(num_prompts, G)
            scores_tensor = scores_tensor.clamp(self.score_min, self.score_max)

            valid_mask = ~torch.isnan(scores_tensor)
            var_per_prompt = torch.zeros(num_prompts, device=device)
            for idx in range(num_prompts):
                mask = valid_mask[idx]
                if mask.sum() > 1:
                    vals = scores_tensor[idx][mask]
                    var_per_prompt[idx] = vals.var(unbiased=False)
                else:
                    var_per_prompt[idx] = 0.0

            score_range = self.score_max - self.score_min
            max_var = 0.25 * (score_range ** 2) + 1e-8
            uncertainty_norm = (var_per_prompt / max_var).clamp(0.0, 1.0)

            ua_alpha = self.ua_alpha
            ua_scale_per_prompt = torch.exp(-ua_alpha * uncertainty_norm)
            ua_scale = ua_scale_per_prompt.repeat_interleave(G, dim=0)
        else:
            ua_scale = None


        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:

                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:

                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)


        rewards = rewards_per_func.sum(dim=1)


        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)


        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)


        if getattr(self, "use_self_certainty", False) and sce_advantage is not None:
            if getattr(self, "self_certainty_only", False):
                advantages = self.self_certainty_coef * sce_advantage
            else:
                advantages = advantages + self.self_certainty_coef * sce_advantage


        if ua_scale is not None:
            advantages = advantages * ua_scale


        per_token_loss_main = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss_main = -(per_token_loss_main - self.beta * per_token_kl)

        seq_loss_main = (per_token_loss_main * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        seq_loss_total = seq_loss_main


        if getattr(self, "enable_prcp_loss", False) and has_deg:

            flat_images_deg = []
            for idx, sample_imgs in enumerate(images_deg):
                if isinstance(sample_imgs, (list, tuple)):
                    if len(sample_imgs) == 0:

                        orig_imgs = images_orig[idx]
                        if isinstance(orig_imgs, (list, tuple)):
                            flat_images_deg.append(orig_imgs[0] if len(orig_imgs) > 0 else orig_imgs)
                        else:
                            flat_images_deg.append(orig_imgs)
                    else:
                        flat_images_deg.append(sample_imgs[0])
                else:
                    flat_images_deg.append(sample_imgs)


            img_inputs_deg = self.processing_class.image_processor(
                flat_images_deg,
                return_tensors="pt",
            )
            img_inputs_deg = super()._prepare_inputs(img_inputs_deg)

            pixel_values_deg = img_inputs_deg["pixel_values"].repeat(self.num_generations, 1)
            image_grid_thw_deg = img_inputs_deg["image_grid_thw"].repeat_interleave(
                self.num_generations,
                dim=0,
            )





            with torch.no_grad():
                per_token_logps_deg = self._get_per_token_logps(
                    model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values_deg,
                    image_grid_thw_deg,
                )
            per_token_logps_deg = per_token_logps_deg[:, prompt_length - 1 :].detach()




            per_token_kl_prcp = _low_var_kl(
                log_probs=per_token_logps,
                ref_log_probs=per_token_logps_deg,
            )


            total_kl_prcp = (per_token_kl_prcp * completion_mask).sum()
            total_tokens = completion_mask.sum().clamp_min(1)
            kl_prcp_token_avg = total_kl_prcp / total_tokens


            kl_prcp_seq = (per_token_kl_prcp * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)


            logp_seq_orig = (per_token_logps * completion_mask).sum(dim=1)
            logp_seq_deg = (per_token_logps_deg * completion_mask).sum(dim=1)

            H_orig_seq = logp_seq_orig
            H_deg_seq = logp_seq_deg


            dynamic_gamma = self.prcp_gamma_base
            zero_epoch = getattr(self, "prcp_kl_zero_epoch", None)

            if (
                getattr(self, "use_kl_prcp", False)
                and dynamic_gamma != 0.0
                and zero_epoch is not None
                and zero_epoch > 0
            ):
                current_epoch = self.state.epoch if self.state.epoch is not None else 0.0
                max_epoch = float(zero_epoch)

                if current_epoch >= max_epoch:
                    dynamic_gamma = 0.0
                else:
                    t = max(current_epoch, 0.0) / max_epoch
                    t = min(t, 1.0)
                    factor = 0.5 * (1.0 + math.cos(math.pi * t))
                    dynamic_gamma = self.prcp_gamma_base * factor

                self._metrics["prcp_gamma"].append(dynamic_gamma)





            loss_prcp_seq = torch.zeros_like(seq_loss_main)
            if getattr(self, "use_kl_prcp", False) and dynamic_gamma != 0.0:

                loss_prcp_seq = loss_prcp_seq - dynamic_gamma * kl_prcp_token_avg
            if getattr(self, "use_ori_entropy_loss", False) and self.prcp_eta1 != 0.0:
                loss_prcp_seq = loss_prcp_seq - self.prcp_eta1 * H_orig_seq
            if getattr(self, "use_aug_entropy_loss", False) and self.prcp_eta2 != 0.0:
                loss_prcp_seq = loss_prcp_seq - self.prcp_eta2 * H_deg_seq


            seq_loss_total = seq_loss_main + loss_prcp_seq


            self._metrics["prcp_kl"].append(kl_prcp_token_avg.detach().item())
            self._metrics["prcp_H_orig"].append(
                self.accelerator.gather_for_metrics(H_orig_seq).mean().item()
            )
            self._metrics["prcp_H_deg"].append(
                self.accelerator.gather_for_metrics(H_deg_seq).mean().item()
            )


        loss = seq_loss_total.mean()


        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())


        if sce_advantage is not None:
            sce_all = self.accelerator.gather_for_metrics(sce_advantage)
            self._metrics["sce/mean"].append(sce_all.mean().item())
            self._metrics["sce/std"].append(sce_all.std().item())

        if ua_scale is not None:
            mean_scale = self.accelerator.gather_for_metrics(ua_scale).mean().item()
            self._metrics["ua/mean_scale"].append(mean_scale)
            if "uncertainty_norm" in locals():
                mean_uncertainty = self.accelerator.gather_for_metrics(uncertainty_norm).mean().item()
                self._metrics["ua/mean_uncertainty"].append(mean_uncertainty)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
