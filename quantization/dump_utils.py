import fnmatch
from typing import List, Tuple, Union

import torch.nn as nn


def get_disable_layer_names(
    model: nn.Module,
    layer_include: Union[List[str], Tuple[str], str],
    layer_exclude: Union[List[str], Tuple[str], str],
) -> List[str]:
    if isinstance(layer_include, str):
        layer_include = [layer_include]
    if isinstance(layer_exclude, str):
        layer_exclude = [layer_exclude]

    all_linear_names: list[str] = []
    quant_layer_names: set[str] = set()

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            all_linear_names.append(name)

        if layer_include and not any(fnmatch.fnmatch(name, pattern) for pattern in layer_include):
            continue
        if layer_exclude and any(fnmatch.fnmatch(name, pattern) for pattern in layer_exclude):
            continue

        quant_layer_names.add(name)

    return [name for name in all_linear_names if name not in quant_layer_names]

