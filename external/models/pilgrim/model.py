import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .parallel import model_attr


class LegacyCompatibleEmbeddingBagLinear(nn.Module):
    """Exact replacement for Linear(one_hot(flat(indices))) with legacy checkpoint I/O."""

    def __init__(self, state_size, num_classes, out_features, bias=True):
        super().__init__()
        self.state_size = int(state_size)
        self.num_classes = int(num_classes)
        self.out_features = int(out_features)
        self.in_features = self.state_size * self.num_classes

        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

        position_offsets = torch.arange(self.state_size, dtype=torch.int64) * self.num_classes
        self.register_buffer("position_offsets", position_offsets, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, indices):
        if indices.ndim != 2 or indices.size(1) != self.state_size:
            raise ValueError(
                f"expected indices with shape [batch, {self.state_size}], got {tuple(indices.shape)}"
            )
        token_ids = indices + self.position_offsets.unsqueeze(0)
        out = F.embedding_bag(token_ids, self.weight, mode="sum")
        if self.bias is not None:
            out = out + self.bias
        return out

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        legacy_weight = self.weight.transpose(0, 1)
        destination[prefix + "weight"] = legacy_weight if keep_vars else legacy_weight.detach()
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            loaded_weight = state_dict[weight_key]
            expected_legacy_shape = (self.out_features, self.in_features)
            expected_native_shape = (self.in_features, self.out_features)
            if loaded_weight.shape == expected_legacy_shape:
                state_dict[weight_key] = loaded_weight.transpose(0, 1)
            elif loaded_weight.shape != expected_native_shape:
                error_msgs.append(
                    f"size mismatch for {weight_key}: expected {expected_legacy_shape} or "
                    f"{expected_native_shape}, got {tuple(loaded_weight.shape)}"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class Pilgrim(nn.Module):
    def __init__(self, state_size, hd1=5000, hd2=1000, nrd=2, output_dim=1, dropout_rate=0.0, num_classes=6):
        super(Pilgrim, self).__init__()
        self.dtype = torch.float32
        self.state_size = state_size
        self.num_classes = num_classes
        self.hd1 = hd1
        self.hd2 = hd2
        self.nrd = nrd
        self.output_dim = int(output_dim)
        self.z_add = 0
        if self.output_dim <= 0:
            raise ValueError("output_dim must be > 0")

        self.input_layer = LegacyCompatibleEmbeddingBagLinear(
            state_size=state_size,
            num_classes=num_classes,
            out_features=hd1,
        )
        self.bn1 = nn.BatchNorm1d(hd1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        if hd2 > 0:
            self.hidden_layer = nn.Linear(hd1, hd2)
            self.bn2 = nn.BatchNorm1d(hd2)
            hidden_dim_for_output = hd2
        else:
            self.hidden_layer = None
            self.bn2 = None
            hidden_dim_for_output = hd1

        if nrd > 0 and hd2 > 0:
            self.residual_blocks = nn.ModuleList(
                [ResidualBlock(hd2, dropout_rate) for _ in range(nrd)]
            )
        else:
            self.residual_blocks = None

        self.output_layer = nn.Linear(hidden_dim_for_output, self.output_dim)

    def forward(self, z):
        x = self.input_layer(z.long() + self.z_add).to(self.dtype)

        # Input block
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Optional hidden block
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Optional residual stack
        if self.residual_blocks is not None:
            for block in self.residual_blocks:
                x = block(x)

        # Output
        x = self.output_layer(x)
        if self.output_dim == 1:
            return x.squeeze(-1)
        return x


def make_cube4_piece_layout():
    corners = torch.tensor([
        [0, 51, 64], [3, 35, 48], [12, 16, 67], [15, 19, 32],
        [28, 79, 80], [31, 44, 83], [47, 60, 95], [63, 76, 92],
    ])
    wings = torch.tensor([
        [1, 50, 0], [2, 49, 0], [4, 65, 0], [7, 34, 0],
        [8, 66, 0], [11, 33, 0], [13, 17, 0], [14, 18, 0],
        [20, 71, 0], [23, 36, 0], [24, 75, 0], [27, 40, 0],
        [29, 81, 0], [30, 82, 0], [39, 52, 0], [43, 56, 0],
        [45, 87, 0], [46, 91, 0], [55, 68, 0], [59, 72, 0],
        [61, 94, 0], [62, 93, 0], [77, 88, 0], [78, 84, 0],
    ])
    centers = torch.tensor([
        [face * 16 + offset, 0, 0]
        for face in range(6)
        for offset in (5, 6, 9, 10)
    ])
    positions = torch.cat((corners, wings, centers))
    mask = torch.cat((
        torch.ones((8, 3), dtype=torch.bool),
        torch.tensor([True, True, False]).view(1, 3).expand(24, 3),
        torch.tensor([True, False, False]).view(1, 3).expand(24, 3),
    ))
    types = torch.cat((
        torch.zeros(8, dtype=torch.int64),
        torch.ones(24, dtype=torch.int64),
        torch.full((24,), 2, dtype=torch.int64),
    ))
    return positions, mask, types


def _transformer_activation(name):
    return {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }[str(name).lower()]()


class Cube4EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, dropout_rate, activation):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            _transformer_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, inputs):
        hidden = self.norm1(inputs)
        attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        outputs = inputs + self.dropout(attended)
        return outputs + self.dropout(self.ff(self.norm2(outputs)))


class PilgrimCube4PieceTransformer(nn.Module):
    def __init__(
        self,
        *,
        state_size,
        num_classes,
        output_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        ff_dim=1024,
        dropout_rate=0.0,
        activation="silu",
        pooling="cls",
    ):
        super().__init__()
        if int(state_size) != 96 or int(num_classes) != 6:
            raise ValueError("cube4 piece transformer expects 96 positions and 6 color classes")
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")

        self.dtype = torch.float32
        self.state_size = int(state_size)
        self.num_classes = int(num_classes)
        self.output_dim = int(output_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.ff_dim = int(ff_dim)
        self.pooling = str(pooling)
        self.activation = str(activation)
        self.piece_layout = "cube4"
        self.piece_embed_mode = "piece_local"
        self.z_add = 0

        positions, mask, types = make_cube4_piece_layout()
        self.num_pieces, self.max_piece_size = positions.size()
        self.num_piece_types = int(types.max()) + 1
        self.register_buffer("piece_positions", positions, persistent=False)
        self.register_buffer("piece_mask", mask, persistent=False)
        self.register_buffer("piece_types", types, persistent=False)
        self.local_value_embedding = nn.Embedding(
            self.max_piece_size * self.num_classes,
            self.d_model,
        )
        self.piece_projection = nn.Linear(
            self.max_piece_size * self.d_model,
            self.d_model,
        )
        self.piece_position_embedding = nn.Embedding(self.num_pieces, self.d_model)
        self.piece_type_embedding = nn.Embedding(self.num_piece_types, self.d_model)
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, self.d_model))
            if self.pooling == "cls"
            else None
        )
        self.input_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            Cube4EncoderBlock(
                self.d_model,
                self.nhead,
                self.ff_dim,
                dropout_rate,
                self.activation,
            )
            for _ in range(self.num_layers)
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_layer = nn.Linear(self.d_model, self.output_dim)
        self.register_buffer(
            "piece_indices",
            torch.arange(self.num_pieces),
            persistent=False,
        )
        self.register_buffer(
            "local_offsets",
            torch.arange(self.max_piece_size) * self.num_classes,
            persistent=False,
        )
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, states):
        tokens = states.long() + self.z_add
        values = tokens.index_select(
            1,
            self.piece_positions.reshape(-1),
        ).view(states.size(0), self.num_pieces, self.max_piece_size)
        local = self.local_value_embedding(
            values + self.local_offsets.view(1, 1, -1)
        )
        local = local * self.piece_mask.view(
            1,
            self.num_pieces,
            self.max_piece_size,
            1,
        ).to(local.dtype)
        hidden = self.piece_projection(
            local.reshape(states.size(0), self.num_pieces, -1)
        )
        hidden = hidden + self.piece_position_embedding(
            self.piece_indices
        ).unsqueeze(0)
        hidden = hidden + self.piece_type_embedding(self.piece_types).unsqueeze(0)
        if self.cls_token is not None:
            hidden = torch.cat(
                (self.cls_token.expand(states.size(0), -1, -1), hidden),
                dim=1,
            )
        hidden = self.dropout(self.input_norm(hidden))
        for block in self.blocks:
            hidden = block(hidden)
        pooled = hidden[:, 0] if self.pooling == "cls" else hidden.mean(dim=1)
        outputs = self.output_layer(self.output_norm(pooled))
        return outputs.squeeze(-1) if self.output_dim == 1 else outputs


def count_parameters(model):
    """Count the trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_process(model, data, device, batch_size):
    """
    Process data through a model in batches.

    :param data: Tensor of input data
    :param model: A PyTorch model with a forward method that accepts data
    :param device: Device to perform computations (e.g., 'cuda', 'cpu')
    :param batch_size: Number of samples per batch
    :return: Concatenated tensor of model outputs
    """
    output_dtype = model_attr(model, "dtype", torch.float32)
    input_dtype = model_attr(model, "input_dtype", None)
    output_dim = int(model_attr(model, "output_dim", 1))
    if output_dim == 1:
        outputs = torch.empty(data.size(0), dtype=output_dtype, device=device)
    else:
        outputs = torch.empty((data.size(0), output_dim), dtype=output_dtype, device=device)

    # Process each batch
    with torch.inference_mode():
        for i in range(0, data.size(0), batch_size):
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            batch = data[i:i + batch_size]
            if input_dtype is not None and batch.dtype != input_dtype:
                batch = batch.to(dtype=input_dtype)
            batch_output = model(batch)
            if output_dim == 1:
                outputs[i:i + batch_size] = batch_output.view(-1)
            else:
                outputs[i:i + batch_size] = batch_output.view(batch.size(0), output_dim)

    return outputs
