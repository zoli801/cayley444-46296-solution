# Modified for the CayleyPy 4x4x4 reproducibility bundle (2026).
# Derived from AnanasClassic/cayleypy-training-core under Apache-2.0.
# This copy contains Cube4 inference/model integration changes.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .problem import Problem


class PositionClassLinear(nn.Module):
    def __init__(self, state_size: int, num_classes: int, output_size: int) -> None:
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(state_size * num_classes, output_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.register_buffer("position_offsets", torch.arange(state_size) * num_classes, persistent=False)
        bound = 1.0 / math.sqrt(state_size * num_classes)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        token_ids = states.long() + self.position_offsets.unsqueeze(0)
        return F.embedding_bag(token_ids, self.weight, mode="sum") + self.bias


class PairResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.linear2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.bn1(self.linear1(inputs)))
        hidden = self.dropout(hidden)
        hidden = self.bn2(self.linear2(hidden))
        return F.silu(inputs + hidden)


class PairQMLP(nn.Module):
    def __init__(self, state_size: int, num_classes: int, actions: int, hd1: int, hd2: int,
                 residual_blocks: int, dropout: float) -> None:
        super().__init__()
        self.config = dict(state_size=state_size, num_classes=num_classes, actions=actions, hd1=hd1,
                           hd2=hd2, residual_blocks=residual_blocks, dropout=dropout)
        self.input_layer = PositionClassLinear(state_size, num_classes, hd1)
        self.input_bn = nn.BatchNorm1d(hd1)
        self.hidden = nn.Linear(hd1, hd2)
        self.hidden_bn = nn.BatchNorm1d(hd2)
        self.blocks = nn.ModuleList(PairResidualBlock(hd2, dropout) for _ in range(residual_blocks))
        self.output_layer = nn.Linear(hd2, actions)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input_bn(self.input_layer(states)))
        hidden = F.silu(self.hidden_bn(self.hidden(hidden)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_layer(hidden)


def _activation(name: str):
    return {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[str(name).lower()]()


class MegaminxEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout_rate: float, activation: str):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), _activation(activation),
                                nn.Dropout(dropout_rate), nn.Linear(ff_dim, d_model))
        self.fast_inference = False
        self.head_dim = d_model // nhead

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(inputs)
        attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        outputs = inputs + self.dropout(attended)
        return outputs + self.dropout(self.ff(self.norm2(outputs)))


class IHESEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout_rate: float, activation: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = str(activation).lower()

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        return {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}[self.activation](value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(inputs)
        attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        outputs = inputs + self.dropout(attended)
        hidden = self.fc2(self.dropout(self._activate(self.fc1(self.norm2(outputs)))))
        return outputs + self.dropout(hidden)


def _p900_layout():
    corners = torch.arange(60).view(20, 3)
    edges = torch.arange(60, 120).view(30, 2)
    positions = torch.cat((corners, torch.cat((edges, torch.zeros((30, 1), dtype=torch.int64)), dim=1)))
    mask = torch.cat((torch.ones((20, 3), dtype=torch.bool),
                      torch.tensor([True, True, False]).view(1, 3).expand(30, 3)))
    types = torch.cat((torch.zeros(20, dtype=torch.int64), torch.ones(30, dtype=torch.int64)))
    return positions, mask, types


def _ihes_layout():
    corners = torch.tensor([[0,38,48,0],[2,26,36,0],[9,12,50,0],[11,14,24,0],
                            [21,59,60,0],[23,33,62,0],[35,45,71,0],[47,57,69,0]])
    edges = torch.tensor([[1,37,0,0],[3,49,0,0],[8,25,0,0],[10,13,0,0],
                          [15,56,0,0],[20,27,0,0],[22,61,0,0],[32,39,0,0],
                          [34,68,0,0],[44,51,0,0],[46,70,0,0],[58,63,0,0]])
    centers = torch.tensor([[4,5,6,7],[16,17,18,19],[28,29,30,31],
                            [40,41,42,43],[52,53,54,55],[64,65,66,67]])
    positions = torch.cat((corners, edges, centers))
    mask = torch.cat((torch.tensor([True,True,True,False]).view(1,4).expand(8,4),
                      torch.tensor([True,True,False,False]).view(1,4).expand(12,4),
                      torch.ones((6,4), dtype=torch.bool)))
    types = torch.cat((torch.zeros(8, dtype=torch.int64), torch.ones(12, dtype=torch.int64),
                       torch.full((6,), 2, dtype=torch.int64)))
    return positions, mask, types


def _cube4_layout():
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


class PieceTransformer(nn.Module):
    def __init__(self, *, state_size: int, num_classes: int, output_dim: int, layout: str,
                 d_model: int, nhead: int, num_layers: int, ff_dim: int,
                 dropout_rate: float, activation: str, pooling: str):
        super().__init__()
        self.dtype = torch.float32
        self.state_size, self.num_classes, self.output_dim = state_size, num_classes, output_dim
        self.d_model, self.nhead, self.num_layers, self.ff_dim = d_model, nhead, num_layers, ff_dim
        self.pooling, self.activation, self.z_add = pooling, activation, 0
        self.piece_layout = layout
        self.piece_embed_mode = "full_s120" if layout == "p900" else "piece_local"
        if layout == "p900":
            if state_size != 120:
                raise ValueError("p900 layout expects state_size=120")
            positions, mask, types = _p900_layout()
            block = MegaminxEncoderBlock
        elif layout == "ihes":
            if state_size != 72 or num_classes != 72:
                raise ValueError("ihes layout expects 72 positions/classes")
            positions, mask, types = _ihes_layout()
            block = IHESEncoderBlock
        elif layout == "cube4":
            if state_size != 96 or num_classes != 6:
                raise ValueError("cube4 layout expects 96 positions and 6 color classes")
            positions, mask, types = _cube4_layout()
            block = MegaminxEncoderBlock
        else:
            raise ValueError(f"unknown piece layout {layout!r}")
        self.num_pieces, self.max_piece_size = positions.size()
        self.num_piece_types = int(types.max()) + 1
        self.register_buffer("piece_positions", positions, persistent=False)
        self.register_buffer("piece_mask", mask, persistent=False)
        self.register_buffer("piece_types", types, persistent=False)
        self.local_value_embedding = nn.Embedding(self.max_piece_size * num_classes, d_model)
        self.piece_projection = nn.Linear(self.max_piece_size * d_model, d_model)
        self.piece_position_embedding = nn.Embedding(self.num_pieces, d_model)
        self.piece_type_embedding = nn.Embedding(self.num_piece_types, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(block(d_model, nhead, ff_dim, dropout_rate, activation)
                                    for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.register_buffer("piece_indices", torch.arange(self.num_pieces), persistent=False)
        self.register_buffer("local_offsets", torch.arange(self.max_piece_size) * num_classes, persistent=False)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        tokens = states.long() + self.z_add
        values = tokens.index_select(1, self.piece_positions.reshape(-1)).view(
            states.size(0), self.num_pieces, self.max_piece_size)
        local = self.local_value_embedding(values + self.local_offsets.view(1, 1, -1))
        local = local * self.piece_mask.view(1, self.num_pieces, self.max_piece_size, 1).to(local.dtype)
        hidden = self.piece_projection(local.reshape(states.size(0), self.num_pieces, -1))
        hidden = hidden + self.piece_position_embedding(self.piece_indices).unsqueeze(0)
        hidden = hidden + self.piece_type_embedding(self.piece_types).unsqueeze(0)
        if self.cls_token is not None:
            hidden = torch.cat((self.cls_token.expand(states.size(0), -1, -1), hidden), dim=1)
        hidden = self.dropout(self.input_norm(hidden))
        for block in self.blocks:
            hidden = block(hidden)
        pooled = hidden[:, 0] if self.pooling == "cls" else hidden.mean(dim=1)
        outputs = self.output_layer(self.output_norm(pooled))
        return outputs.squeeze(-1) if self.output_dim == 1 else outputs


def build_model(spec: dict[str, Any], problem: Problem, config_path: Path, device: torch.device) -> nn.Module:
    provider = spec["provider"]
    kwargs = dict(spec.get("kwargs", {}))
    if provider == "pair_qmlp":
        model = PairQMLP(
            state_size=problem.state_size, num_classes=problem.num_classes, actions=problem.num_actions,
            hd1=int(kwargs.get("hd1", 64)), hd2=int(kwargs.get("hd2", 256)),
            residual_blocks=int(kwargs.get("residual_blocks", 2)), dropout=float(kwargs.get("dropout", 0.0)),
        )
    else:
        model = PieceTransformer(
            state_size=problem.state_size, num_classes=problem.num_classes,
            output_dim=problem.num_actions, layout=str(spec["layout"]),
            d_model=int(kwargs.get("transformer_d_model", 256)),
            nhead=int(kwargs.get("transformer_heads", 8)),
            num_layers=int(kwargs.get("transformer_layers", 4)),
            ff_dim=int(kwargs.get("transformer_ff_dim", 1024)),
            dropout_rate=float(kwargs.get("dropout_rate", 0.0)),
            activation=str(kwargs.get("transformer_activation", "silu")),
            pooling=str(kwargs.get("transformer_pooling", "cls")),
        )
    return model.to(device)


def model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def load_compatible_weights(model: nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    current = model_state_dict(model)
    compatible = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
    result = model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "missing": list(result.missing_keys),
            "skipped": sorted(set(state) - set(compatible))}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
