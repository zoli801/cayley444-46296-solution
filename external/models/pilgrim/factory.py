from .model import Pilgrim, PilgrimCube4PieceTransformer


def build_model(
    *,
    num_classes,
    state_size,
    output_dim=1,
    dropout_rate=0.0,
    hd1=1024,
    hd2=256,
    nrd=4,
):
    return Pilgrim(
        num_classes=num_classes,
        state_size=state_size,
        hd1=hd1,
        hd2=hd2,
        nrd=nrd,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    )


def build_model_from_info(
    info,
    *,
    num_classes,
    state_size,
    output_dim=1,
    all_moves=None,
    move_names=None,
):
    del all_moves, move_names
    model_spec = info.get("model")
    if not isinstance(model_spec, dict):
        model_spec = info.get("config", {}).get("model")
    if isinstance(model_spec, dict) and model_spec.get("provider") == "piece_transformer":
        layout = str(model_spec.get("layout", "")).lower()
        if layout != "cube4":
            raise ValueError(f"unsupported piece transformer layout: {layout!r}")
        kwargs = model_spec.get("kwargs", {})
        return PilgrimCube4PieceTransformer(
            num_classes=num_classes,
            state_size=state_size,
            output_dim=output_dim,
            d_model=int(kwargs.get("transformer_d_model", 256)),
            nhead=int(kwargs.get("transformer_heads", 8)),
            num_layers=int(kwargs.get("transformer_layers", 4)),
            ff_dim=int(kwargs.get("transformer_ff_dim", 1024)),
            dropout_rate=float(kwargs.get("dropout_rate", 0.0)),
            activation=str(kwargs.get("transformer_activation", "silu")),
            pooling=str(kwargs.get("transformer_pooling", "cls")),
        )
    return build_model(
        num_classes=num_classes,
        state_size=state_size,
        output_dim=output_dim,
        dropout_rate=float(info.get("dropout", 0.0)),
        hd1=int(info.get("hd1", 1024)),
        hd2=int(info.get("hd2", 256)),
        nrd=int(info.get("nrd", 4)),
    )
