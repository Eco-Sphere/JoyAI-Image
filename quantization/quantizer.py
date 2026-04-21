import os
from types import SimpleNamespace

import torch
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator


class JoyAIImageQuantizer:
    def __init__(self, model, quant_config):
        self.model = model
        self.quant_config = quant_config
        self._prepare_model_for_quant()

    def _prepare_model_for_quant(self):
        if not hasattr(self.model, "config"):
            self.model.config = SimpleNamespace()

        if not hasattr(self.model.config, "torch_dtype"):
            self.model.config.torch_dtype = torch.bfloat16
        if not hasattr(self.model, "dtype"):
            self.model.dtype = torch.bfloat16
        if not hasattr(self.model.config, "model_type"):
            self.model.config.model_type = "joyai_image"

    def quantize(self) -> Calibrator:
        calibrator = Calibrator(
            model=self.model,
            cfg=self.quant_config,
            disable_level="L0",
        )
        calibrator.run()
        return calibrator

    @staticmethod
    def save_quantized_weights(calibrator: Calibrator, save_dir: str):
        try:
            os.makedirs(save_dir, exist_ok=True)
            calibrator.save(save_dir, save_type=["safe_tensor"])
            print(f"Quantized weights saved to: {save_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to save quantized weights: {e}") from e

