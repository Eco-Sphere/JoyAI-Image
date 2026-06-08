# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from packaging import version
from einops import rearrange

from transformers import AutoProcessor

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from modules.models.mmdit.dit import Transformer3DModel
from modules.utils import env_to_bool

try:
    from mindiesd import CacheAgent, CacheConfig
except Exception:
    CacheAgent = None
    CacheConfig = None

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def _raise_if_non_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"{name} contains NaN or Inf.")


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class PipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class Pipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: Any,
        tokenizer: Any,
        transformer: Transformer3DModel,
        scheduler: KarrasDiffusionSchedulers,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.enable_multi_task = getattr(
            self.args, "enable_multi_task_training", False)
        if hasattr(self.vae, "ffactor_spatial"):
            self.vae_scale_factor = self.vae.ffactor_spatial
            self.vae_scale_factor_temporal = self.vae.ffactor_temporal
        else:
            self.vae_scale_factor = 2 ** (
                len(self.vae.config.block_out_channels) - 1)
            self.vae_scale_factor_temporal = 4  # hard code for HunyuanVideoVAE

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor)

        text_encoder_ckpt = dict(args.text_encoder_arch_config.get("params", {}))[
            'text_encoder_ckpt']
        self.qwen_processor = AutoProcessor.from_pretrained(text_encoder_ckpt)

        self.text_token_max_length = self.args.text_token_max_length
        self.prompt_template_encode = {
            'image': "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            'multiple_images': "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n{}<|im_start|>assistant\n",
            'video': "<|im_start|>system\n \\nDescribe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        }
        # [36:-4]
        # [36:-4]
        # [93:-4]
        self.prompt_template_encode_start_idx = {
            'image': 34,
            'multiple_images': 34,
            'video': 91,
        }
        self._dit_cache_steps_count: Optional[int] = None
        self._dit_cache_announced = False
        if not hasattr(self.transformer, "cache_cond"):
            self.transformer.cache_cond = None
        if not hasattr(self.transformer, "cache_uncond"):
            self.transformer.cache_uncond = None

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        template_type: str = 'image',
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode[template_type]
        drop_idx = self.prompt_template_encode_start_idx[template_type]
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=self.text_token_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(
            hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(
            e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = min([
            self.text_token_max_length,
            max([u.size(0) for u in split_hidden_states]),
            max([u.size(0) for u in attn_mask_list])])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
             for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
             for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask


    def encode_prompt_multiple_images(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        images: Optional[torch.Tensor] = None,
        template_type: Optional[str] = 'multiple_images',
        max_sequence_length: Optional[int] = None,
        drop_vit_feature: Optional[float] = False,
    ):
        assert template_type == 'multiple_images', "template_type must be 'multiple_images'"
        device = device or self._execution_device
        template = self.prompt_template_encode[template_type]
        drop_idx = self.prompt_template_encode_start_idx[template_type]
        prompt = [p.replace(
            '<image>\n', '<|vision_start|><|image_pad|><|vision_end|>') for p in prompt]
        prompt = [template.format(p) for p in prompt]

        inputs = self.qwen_processor(
            text=prompt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            **inputs,
            output_hidden_states=True,
        )
        last_hidden_states = encoder_hidden_states.hidden_states[-1]
        if drop_vit_feature:
            input_ids = inputs['input_ids']
            vlm_image_end_idx = torch.where(input_ids[0] == 151653)[0][-1]
            drop_idx = vlm_image_end_idx + 1
        prompt_embeds = last_hidden_states[:, drop_idx:]
        prompt_embeds_mask = inputs['attention_mask'][:, drop_idx:]
        if max_sequence_length is not None and prompt_embeds.shape[1] > max_sequence_length:
            prompt_embeds = prompt_embeds[:, -max_sequence_length:, :]
            prompt_embeds_mask = prompt_embeds_mask[:, -max_sequence_length:]
        return prompt_embeds, prompt_embeds_mask

    def encode_prompt_images(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        images: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
        """
        device = device or self._execution_device
        bs = images.shape[0]
        if images.shape[2] == 1:
            template = self.prompt_template_encode['image']
            drop_idx = self.prompt_template_encode_start_idx['image']
            prompt = template.replace(
                '{}', '<|vision_start|><|image_pad|><|vision_end|>')
            prompt = [prompt] * bs
            inputs = self.qwen_processor(
                text=prompt,
                images=images.squeeze(2),
                padding=True,
                return_tensors="pt",
            ).to(device)
            output_tensor = self.text_encoder(
                **inputs,
                output_hidden_states=True,
            )
            last_hidden_states = output_tensor.hidden_states[-1]
            vis_hidden_states = last_hidden_states[:, drop_idx+3:-6]
            patchify_size = self.qwen_processor.image_processor.merge_size
            image_grid_thw = inputs['image_grid_thw']
            vis_hidden_states = vis_hidden_states.view(
                bs, image_grid_thw[0, 1]//patchify_size, image_grid_thw[0, 2]//patchify_size, -1)
        return vis_hidden_states

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        images: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
        template_type: str = 'image',
        drop_vit_feature: bool = False,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        if images is not None:
            ##################################################
            # from PIL import Image
            # images = [img.resize((512, 512), Image.LANCZOS) for img in images]
            ##################################################
            return self.encode_prompt_multiple_images(
                prompt=prompt,
                images=images,
                device=device,
                max_sequence_length=max_sequence_length,
                drop_vit_feature=drop_vit_feature,
            )
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(
            prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt, template_type, device)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(
            1, num_videos_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(
            batch_size * num_videos_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask


    def check_inputs(
        self,
        prompt,
        height,
        width,
        images=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
        #     logger.warning(
        #         f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
        #     )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    def prepare_conditions(self, latents: torch.Tensor, image=None, last_image=None):
        """
        Prepare conditional inputs for video generation.

        Args:
            latents: Generated latent tensor with shape (B, N, C, T, latent_H, latent_W)
            image: First frame condition, shape (B, N, 3, 1, H, W)
            last_image: Last frame condition, shape (B, N, 3, 1, H, W)

        Returns:
            Combined condition tensor with shape (B, N, C+1, T, H, W)
        """
        device, dtype = latents.device, latents.dtype
        batch_size, num_items, latent_channels, latent_frames, latent_h, latent_w = latents.shape

        # If no conditions provided, return zero condition
        if image is None and last_image is None:
            return torch.zeros(
                batch_size, num_items, latent_channels + 1, latent_frames, latent_h, latent_w,
                device=device, dtype=dtype
            )

        num_frame = (latent_frames - 1) * self.vae_scale_factor_temporal + 1
        height = latent_h * self.vae_scale_factor
        width = latent_w * self.vae_scale_factor

        # Initialize mask
        mask = torch.zeros(batch_size, num_items, 1, latent_frames,
                           latent_h, latent_w, device=device, dtype=dtype)

        # Build video condition
        if image is not None and last_image is not None:
            # Both first and last frame conditions
            image = image.to(device=device, dtype=dtype)
            last_image = last_image.to(device=device, dtype=dtype)

            middle_frames = torch.zeros(
                batch_size, num_items, image.shape[2], num_frame -
                2, height, width,
                device=device, dtype=dtype
            )
            video_condition = torch.cat(
                [image, middle_frames, last_image], dim=3)
            mask[:, :, :, 0] = 1   # Mark first frame as conditional
            mask[:, :, :, -1] = 1  # Mark last frame as conditional

        elif image is not None:
            # Only first frame condition
            image = image.to(device=device, dtype=dtype)
            remaining_frames = torch.zeros(
                batch_size, num_items, image.shape[2], num_frame -
                1, height, width,
                device=device, dtype=dtype
            )
            video_condition = torch.cat([image, remaining_frames], dim=3)
            mask[:, :, :, 0] = 1   # Mark first frame as conditional
        else:
            raise NotImplementedError

        # VAE encode the video condition
        video_condition = rearrange(
            video_condition, "b n c t h w -> (b n) c t h w")
        latent_condition = self.vae.encode(
            video_condition).latent_dist.sample()

        # Normalize
        latent_condition = self.normalize_latents(latent_condition)

        # Reshape back to (B, N, C, T, H, W)
        latent_condition = rearrange(
            latent_condition, "(b n) c t h w -> b n c t h w", b=batch_size)

        # Concat
        return torch.cat([latent_condition, mask], dim=2)

    def prepare_latents(
        self,
        batch_size,
        num_items,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
        reference_images=None,
        image=None,
        last_image=None,
    ):
        shape = (
            batch_size,
            num_items,
            num_channels_latents,
            (video_length - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if reference_images is not None:
                ref_img = [torch.from_numpy(
                    np.array(x.convert("RGB"))) for x in reference_images]
                ref_img = torch.stack(ref_img).to(device=device, dtype=dtype)
                ref_img = ref_img / 127.5 - 1.0
                ref_img = rearrange(ref_img, "x h w c -> x c 1 h w")
                ref_vae = self.vae.encode(ref_img)

                ref_vae = rearrange(
                    ref_vae, "(b n) c 1 h w -> b n c 1 h w", n=(num_items - 1))
                noise = randn_tensor(
                    (shape[0], 1, *shape[2:]),
                    generator=generator, device=device, dtype=dtype
                )
                latents = torch.cat([ref_vae, noise], dim=1)
            else:
                latents = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
        else:
            latents = latents.to(device)

        if not self.enable_multi_task:
            return latents, None

        # image: (b, n, c, 1, h, w), last_image: (b, n, c, 1, h, w)
        condition = self.prepare_conditions(latents, image, last_image)

        return latents, condition

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        # return self._guidance_scale > 1 and self.transformer.config.time_cond_proj_dim is None
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def pad_sequence(self, x: torch.Tensor, target_length: int):
        current_length = x.shape[1]
        if current_length >= target_length:
            return x[:, -target_length:]
        padding_length = target_length - current_length
        if x.ndim >= 3:
            padding = torch.zeros(
                (x.shape[0], padding_length, *x.shape[2:]),
                dtype=x.dtype, device=x.device
            )
        else:
            padding = torch.zeros(
                (x.shape[0], padding_length),
                dtype=x.dtype, device=x.device
            )
        return torch.cat([x, padding], dim=1)

    def _cfg_parallel_enabled(self) -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return False
        world_size = dist.get_world_size()
        ulysses_size = int(os.environ.get("ULYSSES_SIZE", "1"))
        cfg_size_raw = os.environ.get("CFG_SIZE", "auto").strip().lower()

        if cfg_size_raw in ("", "auto"):
            # Backward-compatible default:
            # - world_size==2 and no sequence parallel -> enable cfg parallel
            # - otherwise -> disable cfg parallel
            cfg_size = 2 if (world_size == 2 and ulysses_size == 1) else 1
        else:
            try:
                cfg_size = int(cfg_size_raw)
            except ValueError as exc:
                raise RuntimeError(
                    f"CFG_SIZE must be one of: auto, 1, 2; got {cfg_size_raw!r}."
                ) from exc
            if cfg_size not in (1, 2):
                raise RuntimeError(
                    f"CFG_SIZE must be one of: auto, 1, 2; got {cfg_size}."
                )

        if cfg_size == 1:
            return False

        # cfg_size == 2 path
        if ulysses_size > 1:
            raise RuntimeError(
                f"CFG_SIZE=2 is incompatible with ULYSSES_SIZE={ulysses_size}. "
                "Please set ULYSSES_SIZE=1 or CFG_SIZE=1."
            )
        if world_size != 2:
            raise RuntimeError(
                f"CFG_SIZE=2 currently requires WORLD_SIZE=2, but got WORLD_SIZE={world_size}."
            )
        return True

    @staticmethod
    def _cfg_parallel_rank() -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 0
        return dist.get_rank()

    @staticmethod
    def _all_gather_cfg_noise_pred(
        noise_pred: torch.Tensor,
        gathered: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("CFG parallel all_gather requires initialized torch.distributed.")
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"CFG parallel expects world_size == 2, but got {world_size}.")
        dist.all_gather(gathered, noise_pred)
        # rank 0 computes unconditional branch, rank 1 computes conditional branch
        return gathered[0], gathered[1]

    @staticmethod
    def _get_dit_cache_flags() -> Tuple[bool, bool]:
        return env_to_bool("COND_CACHE"), env_to_bool("UNCOND_CACHE")

    @staticmethod
    def _get_dit_cache_method() -> str:
        # Keep a fixed, known set of methods from checked-in references:
        # - Qwen-Image: dit_block_cache
        # - FLUX.2-dev: attention_cache
        attention_cache_enabled = os.environ.get("ATTENTION_CACHE", "0").strip()
        try:
            if bool(int(attention_cache_enabled)):
                return "attention_cache"
        except ValueError as exc:
            raise ValueError(
                f"Environment variable ATTENTION_CACHE must be 0 or 1, but got {attention_cache_enabled!r}."
            ) from exc
        return "dit_block_cache"

    def _ensure_dit_cache_agents(
        self,
        num_inference_steps: int,
        cond_cache_enabled: bool,
        uncond_cache_enabled: bool,
        cache_method: str,
    ) -> None:
        if not (cond_cache_enabled or uncond_cache_enabled):
            return
        if CacheConfig is None or CacheAgent is None:
            raise RuntimeError(
                "COND_CACHE/UNCOND_CACHE is enabled but mindiesd cache modules are unavailable."
            )

        block_count = len(getattr(self.transformer, "double_blocks", []))
        if block_count <= 0:
            raise RuntimeError("Transformer does not expose double_blocks; cannot enable DiT block cache.")

        has_cond_agent = getattr(self.transformer, "cache_cond", None) is not None
        has_uncond_agent = getattr(self.transformer, "cache_uncond", None) is not None
        cache_matches_steps = self._dit_cache_steps_count == int(num_inference_steps)
        if (
            cache_matches_steps
            and (not cond_cache_enabled or has_cond_agent)
            and (not uncond_cache_enabled or has_uncond_agent)
        ):
            return

        max_step_index = max(0, int(num_inference_steps) - 1)
        step_interval = 3
        if cache_method == "attention_cache":
            # FLUX.2-dev reference: step_start=5, step_interval=3, step_end=45.
            step_start = min(5, max_step_index)
            step_end = min(45, max_step_index)
            if step_end < step_start:
                step_end = step_start
            try:
                cache_config = CacheConfig(
                    method=cache_method,
                    blocks_count=block_count,
                    steps_count=int(num_inference_steps),
                    step_start=step_start,
                    step_interval=step_interval,
                    step_end=step_end,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create CacheConfig with method={cache_method!r}. "
                    "Current supported methods: dit_block_cache, attention_cache."
                ) from exc
        elif cache_method == "dit_block_cache":
            # Qwen-Image reference: step_start=10, step_interval=3, step_end=35, block=[10,50].
            step_start = min(10, max_step_index)
            step_end = min(35, max_step_index)
            if step_end < step_start:
                step_end = step_start

            max_block_index = max(0, block_count - 1)
            block_start = min(10, max_block_index)
            block_end = min(50, max_block_index)
            if block_end < block_start:
                block_end = block_start

            try:
                cache_config = CacheConfig(
                    method=cache_method,
                    blocks_count=block_count,
                    steps_count=int(num_inference_steps),
                    step_start=step_start,
                    step_interval=step_interval,
                    step_end=step_end,
                    block_start=block_start,
                    block_end=block_end,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create CacheConfig with method={cache_method!r}. "
                    "Current supported methods: dit_block_cache, attention_cache."
                ) from exc
        else:
            raise RuntimeError(
                f"Unsupported cache method: {cache_method!r}. "
                "Current supported methods: dit_block_cache, attention_cache."
            )

        self.transformer.cache_cond = CacheAgent(cache_config) if cond_cache_enabled else None
        self.transformer.cache_uncond = CacheAgent(cache_config) if uncond_cache_enabled else None
        self._dit_cache_steps_count = int(num_inference_steps)

        is_rank0 = (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )
        if is_rank0 and not self._dit_cache_announced:
            logger.info(
                "Enable transformer cache: method=%s, cond=%s, uncond=%s, blocks=%s, steps=%s",
                cache_method,
                cond_cache_enabled,
                uncond_cache_enabled,
                block_count,
                num_inference_steps,
            )
            self._dit_cache_announced = True

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        num_frames: int,
        images: Optional[torch.Tensor] = None,
        image_condition: torch.Tensor | None = None,
        last_image_condition: torch.Tensor | None = None,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        enable_tiling: bool = False,
        max_sequence_length: int = 4096,
        drop_vit_feature: bool = False,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.

            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            images=images,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False
        cfg_parallel_enabled = (
            self.do_classifier_free_guidance and self._cfg_parallel_enabled()
        )
        cfg_parallel_rank = self._cfg_parallel_rank() if cfg_parallel_enabled else 0
        cfg_gathered_noise_pred: Optional[List[torch.Tensor]] = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.transformer.device

        # 3. Encode input prompt
        template_type = 'image' if num_frames == 1 else "video"
        num_items = 1 if images is None or len(
            images) == 0 else 1 + len(images)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            images=images,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            template_type=template_type,
            drop_vit_feature=drop_vit_feature,
        )
        # For classifier free guidance, we need cond + uncond branches.
        # In single-rank mode we concatenate branches in batch dimension.
        # In cfg-parallel mode (world_size == 2), each rank runs one branch.
        if self.do_classifier_free_guidance:
            if negative_prompt is None and negative_prompt_embeds is None:
                # default_negative_prompt = 'low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers.'  # noqa
                default_negative_prompt = ""
                if num_items <= 1:
                    negative_prompt = [
                        f"<|im_start|>user\n{default_negative_prompt}<|im_end|>\n"] * batch_size
                else:
                    image_tokens = "<image>\n" * (num_items - 1)
                    negative_prompt = [
                        f"<|im_start|>user\n{image_tokens}{default_negative_prompt}<|im_end|>\n"] * batch_size

            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                images=images,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                template_type=template_type,
            )

            max_seq_len = max(
                prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
            cond_prompt_embeds = self.pad_sequence(prompt_embeds, max_seq_len)
            uncond_prompt_embeds = self.pad_sequence(
                negative_prompt_embeds, max_seq_len
            )

            cond_prompt_embeds_mask = prompt_embeds_mask
            uncond_prompt_embeds_mask = negative_prompt_embeds_mask
            if prompt_embeds_mask is not None:
                cond_prompt_embeds_mask = self.pad_sequence(
                    prompt_embeds_mask, max_seq_len
                )
                uncond_prompt_embeds_mask = self.pad_sequence(
                    negative_prompt_embeds_mask, max_seq_len
                )

            if cfg_parallel_enabled:
                # Keep cond/uncond branches on different ranks and all_gather after forward.
                prompt_embeds = cond_prompt_embeds
                negative_prompt_embeds = uncond_prompt_embeds
                prompt_embeds_mask = cond_prompt_embeds_mask
                negative_prompt_embeds_mask = uncond_prompt_embeds_mask
            else:
                # Single-rank fallback: batch-concat cond/uncond and split after one forward.
                prompt_embeds = torch.cat([uncond_prompt_embeds, cond_prompt_embeds])
                if cond_prompt_embeds_mask is not None:
                    prompt_embeds_mask = torch.cat(
                        [uncond_prompt_embeds_mask, cond_prompt_embeds_mask]
                    )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
        cond_cache_enabled, uncond_cache_enabled = self._get_dit_cache_flags()
        cache_method = self._get_dit_cache_method()
        self._ensure_dit_cache_agents(
            num_inference_steps=num_inference_steps,
            cond_cache_enabled=cond_cache_enabled,
            uncond_cache_enabled=uncond_cache_enabled,
            cache_method=cache_method,
        )

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents, condition = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_items,
            num_channels_latents,
            height,
            width,
            num_frames,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            reference_images=images,
            image=image_condition,
            last_image=last_image_condition,
        )

        target_dtype = PRECISION_TO_TYPE[self.args.dit_precision]
        autocast_enabled = (
            target_dtype != torch.float32
        )
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if num_items > 1:
            ref_latents = latents[:, :(num_items - 1)].clone()

        # if is_progress_bar:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # copy reference latents
                if num_items > 1:
                    latents[:, :(num_items - 1)] = ref_latents.clone()

                # concat condition if enable multi-task
                if condition is not None:
                    latents_ = torch.cat([latents, condition], dim=2)
                else:
                    latents_ = latents

                noise_pred_uncond = None
                noise_pred_text = None

                if cfg_parallel_enabled:
                    # rank 0: uncond branch; rank 1: cond branch
                    latent_model_input = latents_
                    branch_is_cond = cfg_parallel_rank != 0
                    local_prompt_embeds = (
                        prompt_embeds if branch_is_cond else negative_prompt_embeds
                    )
                    local_prompt_embeds_mask = (
                        prompt_embeds_mask if branch_is_cond else negative_prompt_embeds_mask
                    )
                    local_use_cache = (
                        cond_cache_enabled if branch_is_cond else uncond_cache_enabled
                    )
                    t_expand = t.repeat(latent_model_input.shape[0])

                    with torch.autocast(
                        device_type=latent_model_input.device.type,
                        dtype=target_dtype,
                        enabled=autocast_enabled,
                    ):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_expand,
                            encoder_hidden_states=local_prompt_embeds,
                            encoder_hidden_states_mask=local_prompt_embeds_mask,
                            return_dict=False,
                            use_cache=local_use_cache,
                            if_cond=branch_is_cond,
                        )[0]
                        _raise_if_non_finite(noise_pred, "noise_pred")
                elif self.do_classifier_free_guidance and (
                    cond_cache_enabled or uncond_cache_enabled
                ):
                    # Single-rank CFG with cache: run cond/uncond forward separately.
                    latent_model_input = latents_
                    t_expand = t.repeat(latent_model_input.shape[0])

                    with torch.autocast(
                        device_type=latent_model_input.device.type,
                        dtype=target_dtype,
                        enabled=autocast_enabled,
                    ):
                        noise_pred_text = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_expand,
                            encoder_hidden_states=cond_prompt_embeds,
                            encoder_hidden_states_mask=cond_prompt_embeds_mask,
                            return_dict=False,
                            use_cache=cond_cache_enabled,
                            if_cond=True,
                        )[0]
                        noise_pred_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_expand,
                            encoder_hidden_states=uncond_prompt_embeds,
                            encoder_hidden_states_mask=uncond_prompt_embeds_mask,
                            return_dict=False,
                            use_cache=uncond_cache_enabled,
                            if_cond=False,
                        )[0]
                        _raise_if_non_finite(noise_pred_text, "noise_pred_text")
                        _raise_if_non_finite(noise_pred_uncond, "noise_pred_uncond")
                else:
                    # Default path: single forward (concat cond/uncond when CFG is enabled).
                    latent_model_input = (
                        torch.cat([latents_] * 2)
                        if self.do_classifier_free_guidance
                        else latents_
                    )
                    t_expand = t.repeat(latent_model_input.shape[0])

                    with torch.autocast(
                        device_type=latent_model_input.device.type,
                        dtype=target_dtype,
                        enabled=autocast_enabled,
                    ):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_expand,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            return_dict=False,
                            use_cache=(cond_cache_enabled and not self.do_classifier_free_guidance),
                            if_cond=True,
                        )[0]
                        _raise_if_non_finite(noise_pred, "noise_pred")

                # perform guidance
                if self.do_classifier_free_guidance:
                    if cfg_parallel_enabled:
                        if (
                            cfg_gathered_noise_pred is None
                            or cfg_gathered_noise_pred[0].shape != noise_pred.shape
                            or cfg_gathered_noise_pred[0].dtype != noise_pred.dtype
                            or cfg_gathered_noise_pred[0].device != noise_pred.device
                        ):
                            cfg_gathered_noise_pred = [
                                torch.empty_like(noise_pred),
                                torch.empty_like(noise_pred),
                            ]
                        noise_pred_uncond, noise_pred_text = (
                            self._all_gather_cfg_noise_pred(
                                noise_pred, cfg_gathered_noise_pred
                            )
                        )
                    else:
                        if noise_pred_uncond is None or noise_pred_text is None:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    cond_norm = torch.norm(
                        noise_pred_text, dim=2, keepdim=True)
                    noise_norm = torch.norm(noise_pred, dim=2, keepdim=True)
                    noise_pred = noise_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop(
                        "prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        if not output_type == "latent":
            if not (len(latents.shape) == 5 or len(latents.shape) == 6):
                raise ValueError(
                    f"Only support latents with shape (b, n, c, h, w) or (b, n, c, f, h, w), but got {latents.shape}."
                )

            _raise_if_non_finite(latents, "latents")

            # Decode latents (VAE handles denormalization internally via scale)
            latents = rearrange(
                latents, "b n c f h w -> (b n) c f h w")

            with torch.autocast(
                device_type=latents.device.type, dtype=vae_dtype, enabled=vae_autocast_enabled
            ):
                image = self.vae.decode(
                    latents, return_dict=False
                )[0]
                image = rearrange(
                    image, "(b n) c f h w -> b n c f h w", b=batch_size)

            # Replace the first frame with the image condition
            if image_condition is not None:
                image[:, :, :, :1] = image_condition

            # Replace the last frame with the last image condition
            if last_image_condition is not None:
                image[:, :, :, -1:] = last_image_condition

        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float().permute(0, 1, 3, 2, 4, 5)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return PipelineOutput(videos=image)
