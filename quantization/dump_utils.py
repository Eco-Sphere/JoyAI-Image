import fnmatch
from typing import List, Tuple, Union

import torch.nn as nn


def _normalize_patterns(patterns: Union[List[str], Tuple[str], str]) -> list[str]:
    if isinstance(patterns, str):
        return [patterns]
    return list(patterns)


def get_non_quantized_linear_names(
    model: nn.Module,
    layer_include: Union[List[str], Tuple[str], str],
    layer_exclude: Union[List[str], Tuple[str], str],
) -> List[str]:
    layer_include = _normalize_patterns(layer_include)
    layer_exclude = _normalize_patterns(layer_exclude)

    non_quantized_linear_names: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue

        included = not layer_include or any(
            fnmatch.fnmatch(name, pattern) for pattern in layer_include
        )
        excluded = bool(layer_exclude) and any(
            fnmatch.fnmatch(name, pattern) for pattern in layer_exclude
        )
        if not included or excluded:
            non_quantized_linear_names.append(name)

    return non_quantized_linear_names
