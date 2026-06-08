from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

from PIL import Image
import torch

from infer_runtime.infer_config import InferConfig, load_infer_config_class_from_pyfile
from infer_runtime.prompt_rewrite import rewrite_prompt
from infer_runtime.settings import InferSettings
from modules.models import load_dit, load_pipeline
from modules.utils import _dynamic_resize_from_bucket, env_to_bool, seed_everything

try:
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE_LANCZOS = Image.LANCZOS


def preprocess_edit_image(
    image: Image.Image,
    basesize: int,
    keep_image_size: bool = False,
) -> Image.Image:
    if keep_image_size:
        # keep_image_size is handled as output post-resize, not input crop/align.
        pass
    return _dynamic_resize_from_bucket(image, basesize=basesize)


def apply_transformer_quantization(transformer: torch.nn.Module, quant_desc_path: str) -> None:
    try:
        from mindiesd import quantize
    except ImportError as exc:
        raise RuntimeError(
            "mindiesd is required for quantized inference. "
            "Please install mindiesd and make sure runtime env is initialized."
        ) from exc

    use_nz = False if "w8a8_mxfp8" in quant_desc_path else True
    quantize(
        model=transformer,
        quant_desc_path=quant_desc_path,
        use_nz=use_nz,
    )
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()


def _env_to_optional_positive_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return None
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, but got {raw!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, but got {parsed}.")
    return parsed


def configure_vae_runtime(
    vae: torch.nn.Module,
) -> list[str]:
    vae_tiling = env_to_bool("VAE_TILING", default=False)
    vae_slicing = env_to_bool("VAE_SLICING", default=False)
    vae_decode_batch_size = _env_to_optional_positive_int("VAE_DECODE_BATCH_SIZE")

    messages: list[str] = []

    if vae_tiling:
        if hasattr(vae, "enable_tiling") and callable(getattr(vae, "enable_tiling")):
            vae.enable_tiling()
            messages.append("VAE runtime: tiling enabled via enable_tiling().")
        elif hasattr(vae, "use_tiling"):
            setattr(vae, "use_tiling", True)
            messages.append("VAE runtime: tiling enabled via use_tiling=True.")
        else:
            messages.append("VAE runtime: tiling requested but current VAE does not support it.")

    if vae_slicing:
        if hasattr(vae, "enable_slicing") and callable(getattr(vae, "enable_slicing")):
            vae.enable_slicing()
            messages.append("VAE runtime: slicing enabled via enable_slicing().")
        elif hasattr(vae, "use_slicing"):
            setattr(vae, "use_slicing", True)
            messages.append("VAE runtime: slicing enabled via use_slicing=True.")
        else:
            messages.append("VAE runtime: slicing requested but current VAE does not support it.")

    if vae_decode_batch_size is not None:
        if vae_decode_batch_size <= 0:
            raise ValueError(
                f"vae_decode_batch_size must be > 0, got {vae_decode_batch_size}"
            )
        setattr(vae, "decode_batch_size", int(vae_decode_batch_size))
        messages.append(
            f"VAE runtime: decode_batch_size set to {int(vae_decode_batch_size)}."
        )

    return messages


@dataclass
class InferenceParams:
    prompt: str
    image: Optional[Image.Image]
    height: int
    width: int
    steps: int
    guidance_scale: float
    seed: int
    neg_prompt: str
    basesize: int
    keep_image_size: bool = False


class EditModel:
    def __init__(
        self,
        settings: InferSettings,
        device: torch.device,
        hsdp_shard_dim_override: int | None = None,
        quant_desc_path: str | None = None,
    ):
        self.settings = settings
        self.device = device
        self._rewrite_cache: dict[str, str] = {}

        config_class = load_infer_config_class_from_pyfile(settings.config_path)
        self.cfg: InferConfig = config_class()
        self.cfg.dit_ckpt = settings.ckpt_path
        self.cfg.training_mode = False
        if hsdp_shard_dim_override is not None:
            self.cfg.hsdp_shard_dim = hsdp_shard_dim_override
        if int(os.environ.get('WORLD_SIZE', '1')) > 1 and self.cfg.hsdp_shard_dim > 1:
            self.cfg.use_fsdp_inference = True
            # Keep parameters sharded after each forward to reduce peak memory in 2-card inference.
            self.cfg.reshard_after_forward = True

        self.dit = load_dit(self.cfg, device=self.device)
        if quant_desc_path:
            if self.cfg.use_fsdp_inference:
                raise RuntimeError(
                    "Quantized inference with --quant-desc-path is not supported when FSDP inference is enabled "
                    "(hsdp_shard_dim > 1). Please use --hsdp-shard-dim 1."
                )
            if int(os.environ.get("RANK", "0")) == 0:
                print("Applying quantization to transformer...")
            apply_transformer_quantization(self.dit, quant_desc_path)
        self.dit.requires_grad_(False)
        self.dit.eval()
        self.pipeline = load_pipeline(self.cfg, self.dit, self.device)
        self.vae_runtime_messages = configure_vae_runtime(self.pipeline.vae)
        if int(os.environ.get("RANK", "0")) == 0:
            for msg in self.vae_runtime_messages:
                print(msg)

    def maybe_rewrite_prompt(self, prompt: str, image: Optional[Image.Image], enabled: bool) -> str:
        if not enabled:
            return str(prompt or '')
        cache_key = f"prompt={prompt.strip()}"
        if image is not None:
            cache_key += f"|image={image.size[0]}x{image.size[1]}"
        if cache_key not in self._rewrite_cache:
            self._rewrite_cache[cache_key] = rewrite_prompt(
                prompt,
                image,
                model=self.settings.rewrite_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        return self._rewrite_cache[cache_key]

    @torch.no_grad()
    def infer(self, params: InferenceParams) -> Image.Image:
        if params.image is None:
            prompts = [params.prompt]
            negative_prompt = [params.neg_prompt]
            images = None
            height = params.height
            width = params.width
            original_size = None
        else:
            original_size = params.image.size
            processed = preprocess_edit_image(
                params.image,
                basesize=params.basesize,
                keep_image_size=params.keep_image_size,
            )
            width, height = processed.size
            image_tokens = '<image>\n'
            prompts = [f"<|im_start|>user\n{image_tokens}{params.prompt}<|im_end|>\n"]
            negative_prompt = [f"<|im_start|>user\n{image_tokens}{params.neg_prompt}<|im_end|>\n"]
            images = [processed]

        generator_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        generator = torch.Generator(device=generator_device).manual_seed(int(params.seed))
        output = self.pipeline(
            prompt=prompts,
            negative_prompt=negative_prompt,
            images=images,
            height=height,
            width=width,
            num_frames=1,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
            num_videos_per_prompt=1,
            output_type='pt',
            return_dict=False,
        )
        image_tensor = (output[0, -1, 0] * 255).to(torch.uint8).cpu()
        image = Image.fromarray(image_tensor.permute(1, 2, 0).numpy())

        if params.keep_image_size and original_size is not None and image.size != original_size:
            image = image.resize(original_size, resample=_RESAMPLE_LANCZOS)

        return image


def check_dependency_versions() -> None:
    try:
        import transformers
    except ImportError as e:
        raise RuntimeError(
            'transformers is not installed. '
            'Required version: >=4.57.0 and <4.58.0'
        ) from e

    try:
        import diffusers
    except ImportError as e:
        raise RuntimeError(
            'diffusers is not installed. '
            'Required version: ==0.36.0'
        ) from e

    try:
        from packaging.version import Version
    except ImportError as e:
        raise RuntimeError(
            'packaging is required for version checks. '
            'Please install it with: pip install packaging'
        ) from e

    transformers_version = Version(transformers.__version__)
    diffusers_version = Version(diffusers.__version__)

    min_transformers = Version('4.57.0')
    max_transformers = Version('4.58.0')
    required_diffusers = Version('0.36.0')

    if not (min_transformers <= transformers_version < max_transformers):
        raise RuntimeError(
            f'Unsupported transformers version: {transformers.__version__}. '
            f'Required: >=4.57.0 and <4.58.0'
        )

    if diffusers_version != required_diffusers:
        raise RuntimeError(
            f'Unsupported diffusers version: {diffusers.__version__}. '
            f'Required: ==0.36.0'
        )

def build_model(
    settings: InferSettings,
    device: torch.device | None = None,
    hsdp_shard_dim_override: int | None = None,
    quant_desc_path: str | None = None,
) -> EditModel:
    check_dependency_versions()
    seed_everything(settings.default_seed)
    if device is None:
        if hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.device("npu:0")
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return EditModel(
        settings=settings,
        device=device,
        hsdp_shard_dim_override=hsdp_shard_dim_override,
        quant_desc_path=quant_desc_path,
    )
