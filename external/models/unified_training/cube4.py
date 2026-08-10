# Added for the CayleyPy 4x4x4 reproducibility bundle (2026).
# Package lineage: AnanasClassic/cayleypy-training-core, Apache-2.0.
# Implements the bundle-specific Cube4 representation and integration.

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import _cube4_layout


# Face order in the p002 facelet representation is U, F, R, B, L, D.
# Orders below are the conventional oriented 3x3 cubie orders expressed with
# those face ids.  Keeping an order (rather than just a color set) is what
# makes single-dedge flip parity observable.
EDGE_ORDERS = (
    (0, 2), (0, 1), (0, 4), (0, 3),
    (5, 2), (5, 1), (5, 4), (5, 3),
    (1, 2), (1, 4), (3, 4), (3, 2),
)
CORNER_ORDERS = (
    (0, 2, 1), (0, 1, 4), (0, 4, 3), (0, 3, 2),
    (5, 1, 2), (5, 4, 1), (5, 3, 4), (5, 2, 3),
)
CENTER_OFFSETS = (5, 6, 9, 10)


@dataclass(frozen=True)
class Cube4ReductionStatus:
    centers_solved: torch.Tensor
    corners_valid: torch.Tensor
    edges_paired: torch.Tensor
    edges_valid: torch.Tensor
    corner_twist_valid: torch.Tensor
    oll_parity: torch.Tensor
    pll_parity: torch.Tensor
    strict: torch.Tensor


def _permutation_parity(permutation: torch.Tensor) -> torch.Tensor:
    left = permutation.unsqueeze(2)
    right = permutation.unsqueeze(1)
    upper = torch.triu(
        torch.ones(
            permutation.size(1),
            permutation.size(1),
            dtype=torch.bool,
            device=permutation.device,
        ),
        diagonal=1,
    )
    return ((left > right) & upper).sum(dim=(1, 2)).remainder(2).bool()


class Cube4StrictReduction:
    """Exact p002 predicate for the outer-turn orbit of the solved 4x4 cube."""

    def __init__(self, target: torch.Tensor):
        target = torch.as_tensor(target, dtype=torch.int64).flatten()
        if target.numel() != 96:
            raise ValueError("cube4 strict reduction expects 96 facelets")
        self.device = target.device
        self.target = target

        positions, _, _ = _cube4_layout()
        corners = positions[:8].to(self.device)
        wings = positions[8:32, :2].to(self.device)

        corner_rows = {}
        for row in corners.tolist():
            corner_rows[tuple(sorted(index // 16 for index in row))] = row
        ordered_corners = []
        for faces in CORNER_ORDERS:
            row = corner_rows[tuple(sorted(faces))]
            by_face = {index // 16: index for index in row}
            ordered_corners.append([by_face[face] for face in faces])
        self.corner_positions = torch.tensor(ordered_corners, dtype=torch.int64, device=self.device)

        wing_rows: dict[tuple[int, int], list[list[int]]] = {}
        for row in wings.tolist():
            wing_rows.setdefault(tuple(sorted(index // 16 for index in row)), []).append(row)
        ordered_edges = []
        for faces in EDGE_ORDERS:
            rows = wing_rows[tuple(sorted(faces))]
            if len(rows) != 2:
                raise ValueError(f"edge slot {faces} does not contain exactly two wings")
            slot = []
            for row in rows:
                by_face = {index // 16: index for index in row}
                slot.append([by_face[face] for face in faces])
            ordered_edges.append(slot)
        self.edge_positions = torch.tensor(ordered_edges, dtype=torch.int64, device=self.device)

        self.center_positions = torch.tensor(
            [face * 16 + offset for face in range(6) for offset in CENTER_OFFSETS],
            dtype=torch.int64,
            device=self.device,
        )
        self.corner_piece_colors = target[self.corner_positions]
        self.edge_piece_colors = target[self.edge_positions[:, 0]]
        self.corner_signatures = self.corner_piece_colors.sort(dim=1).values
        self.edge_signatures = self.edge_piece_colors.sort(dim=1).values

    def _as_batch(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, device=self.device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.size(1) != 96:
            raise ValueError(f"expected states shaped [batch, 96], got {tuple(states.shape)}")
        return states.long()

    def classify(self, states: torch.Tensor) -> Cube4ReductionStatus:
        states = self._as_batch(states)
        batch = states.size(0)

        centers_solved = states[:, self.center_positions].eq(
            self.target[self.center_positions].unsqueeze(0)
        ).all(dim=1)

        corner_colors = states[:, self.corner_positions]
        corner_sorted = corner_colors.sort(dim=2).values
        corner_matches = corner_sorted.unsqueeze(2).eq(
            self.corner_signatures.view(1, 1, 8, 3)
        ).all(dim=3)
        corner_match_count = corner_matches.sum(dim=2)
        corner_permutation = corner_matches.to(torch.int64).argmax(dim=2)
        corner_unique = torch.nn.functional.one_hot(
            corner_permutation, num_classes=8
        ).sum(dim=1).eq(1).all(dim=1)
        corners_valid = corner_match_count.eq(1).all(dim=1) & corner_unique

        corner_primary = self.corner_piece_colors[:, 0].index_select(
            0, corner_permutation.reshape(-1)
        ).view(batch, 8)
        corner_twists = corner_colors.eq(corner_primary.unsqueeze(2)).to(torch.int64).argmax(dim=2)
        corner_twist_valid = corners_valid & corner_twists.sum(dim=1).remainder(3).eq(0)

        edge_wing_colors = states[:, self.edge_positions]
        edges_paired = edge_wing_colors[:, :, 0].eq(edge_wing_colors[:, :, 1]).all(dim=2).all(dim=1)
        edge_colors = edge_wing_colors[:, :, 0]
        edge_sorted = edge_colors.sort(dim=2).values
        edge_matches = edge_sorted.unsqueeze(2).eq(
            self.edge_signatures.view(1, 1, 12, 2)
        ).all(dim=3)
        edge_match_count = edge_matches.sum(dim=2)
        edge_permutation = edge_matches.to(torch.int64).argmax(dim=2)
        edge_unique = torch.nn.functional.one_hot(
            edge_permutation, num_classes=12
        ).sum(dim=1).eq(1).all(dim=1)
        edges_valid = edges_paired & edge_match_count.eq(1).all(dim=1) & edge_unique

        edge_primary = self.edge_piece_colors[:, 0].index_select(
            0, edge_permutation.reshape(-1)
        ).view(batch, 12)
        edge_flips = edge_colors[:, :, 0].ne(edge_primary)
        oll_parity = edges_valid & edge_flips.sum(dim=1).remainder(2).eq(1)
        corner_parity = _permutation_parity(corner_permutation)
        edge_parity = _permutation_parity(edge_permutation)
        pll_parity = corners_valid & edges_valid & corner_parity.ne(edge_parity)

        strict = (
            centers_solved
            & corners_valid
            & edges_valid
            & corner_twist_valid
            & ~oll_parity
            & ~pll_parity
        )
        return Cube4ReductionStatus(
            centers_solved=centers_solved,
            corners_valid=corners_valid,
            edges_paired=edges_paired,
            edges_valid=edges_valid,
            corner_twist_valid=corner_twist_valid,
            oll_parity=oll_parity,
            pll_parity=pll_parity,
            strict=strict,
        )

    def is_strict(self, states: torch.Tensor) -> torch.Tensor:
        states = self._as_batch(states)
        result = torch.zeros(states.size(0), dtype=torch.bool, device=states.device)
        # Center equality is a very cheap necessary condition and rejects
        # almost every positive-depth training/search state. Avoid extracting
        # all corner/edge cubies for those rows.
        candidates = states[:, self.center_positions].eq(
            self.target[self.center_positions].unsqueeze(0)
        ).all(dim=1)
        if bool(candidates.any()):
            rows = torch.nonzero(candidates, as_tuple=False).flatten()
            result[rows] = self.classify(states.index_select(0, rows)).strict
        return result

    def is_centers_edges(self, states: torch.Tensor) -> torch.Tensor:
        """Return whether centers are solved and both wings of every edge are paired."""
        states = self._as_batch(states)
        centers = states[:, self.center_positions].eq(
            self.target[self.center_positions].unsqueeze(0)
        ).all(dim=1)
        result = torch.zeros(states.size(0), dtype=torch.bool, device=states.device)
        if bool(centers.any()):
            rows = torch.nonzero(centers, as_tuple=False).flatten()
            candidates = states.index_select(0, rows)
            wing_colors = candidates[:, self.edge_positions]
            paired = wing_colors[:, :, 0].eq(wing_colors[:, :, 1]).all(dim=2).all(dim=1)
            result[rows] = paired
        return result

    def canonical_reduction_sectors(self) -> torch.Tensor:
        """Return one reduced representative for strict, OLL, PLL, and both sectors."""
        return torch.cat((self.target.unsqueeze(0), self.canonical_hard_negatives()), dim=0)

    def canonical_hard_negatives(self) -> torch.Tensor:
        """Return physical facelet representatives for OLL, PLL, and both sectors."""
        base = self.target.clone()

        oll = base.clone()
        first = self.edge_positions[0]
        oll[first[:, 0]] = base[first[:, 1]]
        oll[first[:, 1]] = base[first[:, 0]]

        pll = base.clone()
        first, second = self.edge_positions[0], self.edge_positions[1]
        first_piece = self.edge_piece_colors[0]
        second_piece = self.edge_piece_colors[1]
        pll[first[:, 0]] = second_piece[0]
        pll[first[:, 1]] = second_piece[1]
        pll[second[:, 0]] = first_piece[0]
        pll[second[:, 1]] = first_piece[1]

        both = pll.clone()
        both[first[:, 0]] = pll[first[:, 1]]
        both[first[:, 1]] = pll[first[:, 0]]

        result = torch.stack((oll, pll, both))
        status = self.classify(result)
        expected_oll = torch.tensor([True, False, True], device=self.device)
        expected_pll = torch.tensor([False, True, True], device=self.device)
        if not bool(
            status.centers_solved.all()
            and status.edges_paired.all()
            and status.edges_valid.all()
            and status.corner_twist_valid.all()
            and status.oll_parity.eq(expected_oll).all()
            and status.pll_parity.eq(expected_pll).all()
            and (~status.strict).all()
        ):
            raise RuntimeError("failed to construct canonical cube4 parity sectors")
        return result
