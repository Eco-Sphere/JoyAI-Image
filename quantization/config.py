from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig


def get_joyai_image_quant_config(
    quant_mode: str,
    is_dynamic: bool,
    w_sym: bool,
    act_method: int,
    disable_names: list | None = None,
    dev_type: str = "npu",
    dev_id: int | None = None,
    **kwargs,
) -> QuantConfig:
    if act_method not in [1, 2, 3]:
        raise ValueError(
            f"Unsupported act_method {act_method}. Supported values are 1(min-max), 2(histogram), 3(auto-mixed)."
        )

    if quant_mode == "w8a8":
        w_bit = 8
        a_bit = 8
    elif quant_mode == "w8a16":
        w_bit = 8
        a_bit = 16
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}. Choose from ['w8a8', 'w8a16'].")

    return QuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        disable_names=disable_names,
        dev_type=dev_type,
        dev_id=dev_id,
        act_method=act_method,
        pr=1.0,
        w_sym=w_sym,
        mm_tensor=False,
        is_dynamic=is_dynamic,
        **kwargs,
    )

