import torch
import torch.nn as nn


def parse_gpu_ids(gpu_ids):
    if gpu_ids is None:
        return None

    text = str(gpu_ids).strip()
    if not text:
        return None

    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"invalid gpu_ids value: {gpu_ids!r}")

    resolved = [int(part) for part in parts]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"gpu_ids must not contain duplicates: {gpu_ids!r}")
    return resolved


def _gpu_signature(device_props):
    return (
        device_props.name,
        int(device_props.total_memory),
        int(device_props.major),
        int(device_props.minor),
    )


def resolve_gpu_ids(gpu_ids=None):
    requested = parse_gpu_ids(gpu_ids)

    if not torch.cuda.is_available():
        if requested is not None:
            raise RuntimeError("CUDA is not available; --gpu_ids cannot be used")
        return []

    available = torch.cuda.device_count()
    resolved = [0] if requested is None else requested

    for gpu_id in resolved:
        if gpu_id < 0 or gpu_id >= available:
            raise ValueError(f"gpu_id={gpu_id} is out of range for {available} visible CUDA devices")

    if len(resolved) > 1:
        props = {gpu_id: torch.cuda.get_device_properties(gpu_id) for gpu_id in resolved}
        ref_signature = _gpu_signature(props[resolved[0]])
        mismatched = [gpu_id for gpu_id in resolved if _gpu_signature(props[gpu_id]) != ref_signature]
        if mismatched:
            details = ", ".join(
                f"{gpu_id}:{props[gpu_id].name} ({props[gpu_id].total_memory // (1024 ** 2)} MiB)"
                for gpu_id in resolved
            )
            raise RuntimeError(
                "selected --gpu_ids must be homogeneous; got mixed devices: "
                f"{details}"
            )

    return resolved


def resolve_device(gpu_ids=None):
    resolved = resolve_gpu_ids(gpu_ids)
    if not resolved:
        return torch.device("cpu"), resolved
    return torch.device(f"cuda:{resolved[0]}"), resolved


def gpu_ids_to_cli_args(gpu_ids):
    resolved = parse_gpu_ids(gpu_ids)
    if resolved is None:
        return []
    return ["--gpu_ids", ",".join(str(gpu_id) for gpu_id in resolved)]


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_wrap_dataparallel(model, device_ids):
    if not device_ids or len(device_ids) <= 1:
        return model
    return nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])


def model_state_dict(model):
    return unwrap_model(model).state_dict()


def model_attr(model, name, default=None):
    return getattr(unwrap_model(model), name, default)
