#!/usr/bin/env python3
"""Build a faithful compact twsearch definition from sticker permutations.

The construction is group-theoretic rather than tied to a face-layout
convention.  It discovers the 24 centers, 24 paired wings, and eight
three-sticker corners as invariant blocks of the supplied action.  Every
generated cubie move is reconstructed back to all 96 sticker destinations
and compared with puzzle_info.json before the definition is written.
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


Permutation = tuple[int, ...]


def inverse(permutation: Sequence[int]) -> Permutation:
    result = [0] * len(permutation)
    for destination, source in enumerate(permutation):
        result[source] = destination
    return tuple(result)


def sticker_orbits(permutations: Sequence[Permutation], width: int) -> list[list[int]]:
    adjacency = [set() for _ in range(width)]
    for permutation in permutations:
        for destination, source in enumerate(permutation):
            adjacency[destination].add(source)
            adjacency[source].add(destination)
    unseen = set(range(width))
    orbits: list[list[int]] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        pending = [seed]
        orbit: list[int] = []
        while pending:
            point = pending.pop()
            orbit.append(point)
            for neighbor in adjacency[point]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        orbits.append(sorted(orbit))
    return orbits


def generated_partition(
    seed_a: int,
    seed_b: int,
    orbit: Sequence[int],
    permutations: Sequence[Permutation],
) -> list[list[int]]:
    """Smallest invariant equivalence relation containing seed_a ~ seed_b."""

    width = len(permutations[0])
    parent = list(range(width))
    size = [1] * width

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> bool:
        left = find(left)
        right = find(right)
        if left == right:
            return False
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
        return True

    union(seed_a, seed_b)
    changed = True
    while changed:
        changed = False
        classes: dict[int, list[int]] = collections.defaultdict(list)
        for point in orbit:
            classes[find(point)].append(point)
        for members in classes.values():
            first = members[0]
            for permutation in permutations:
                image = permutation[first]
                for member in members[1:]:
                    changed |= union(image, permutation[member])

    classes = collections.defaultdict(list)
    for point in orbit:
        classes[find(point)].append(point)
    return sorted((sorted(members) for members in classes.values()), key=lambda block: block[0])


def find_blocks(
    orbit: Sequence[int], permutations: Sequence[Permutation], block_size: int
) -> list[list[int]] | None:
    seed = orbit[0]
    for partner in orbit[1:]:
        partition = generated_partition(seed, partner, orbit, permutations)
        if all(len(block) == block_size for block in partition):
            return partition
    return None


def equivariant_bijection(
    first: Sequence[int],
    second: Sequence[int],
    permutations: Sequence[Permutation],
) -> dict[int, int] | None:
    first_set = set(first)
    second_set = set(second)
    seed = first[0]
    for image_seed in second:
        mapping = {seed: image_seed}
        reverse = {image_seed: seed}
        pending = [seed]
        valid = True
        while pending and valid:
            point = pending.pop()
            for permutation in permutations:
                point_image = permutation[point]
                mapped_image = permutation[mapping[point]]
                if point_image in mapping:
                    valid = mapping[point_image] == mapped_image
                elif mapped_image in reverse:
                    valid = False
                else:
                    mapping[point_image] = mapped_image
                    reverse[mapped_image] = point_image
                    pending.append(point_image)
                if not valid:
                    break
        if valid and set(mapping) == first_set and set(mapping.values()) == second_set:
            return mapping
    return None


def permutation_parity(values: Sequence[int]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return inversions & 1


def orient_corner_blocks(
    blocks: Sequence[Sequence[int]], permutations: Sequence[Permutation]
) -> list[tuple[int, int, int]]:
    point_location = {
        point: (block_index, coordinate)
        for block_index, block in enumerate(blocks)
        for coordinate, point in enumerate(block)
    }
    constraints: list[list[tuple[int, int]]] = [[] for _ in blocks]
    for permutation in permutations:
        for destination, block in enumerate(blocks):
            images = [point_location[permutation[point]] for point in block]
            source_blocks = {item[0] for item in images}
            if len(source_blocks) != 1:
                raise ValueError("corner block is not preserved by a generator")
            source = images[0][0]
            parity = permutation_parity([item[1] for item in images])
            constraints[destination].append((source, parity))
            constraints[source].append((destination, parity))

    signs: list[int | None] = [None] * len(blocks)
    signs[0] = 0
    pending = [0]
    while pending:
        block = pending.pop()
        assert signs[block] is not None
        for neighbor, parity in constraints[block]:
            required = signs[block] ^ parity
            if signs[neighbor] is None:
                signs[neighbor] = required
                pending.append(neighbor)
            elif signs[neighbor] != required:
                raise ValueError("corner action contains an orientation reflection")
    if any(sign is None for sign in signs):
        raise ValueError("corner blocks are not transitive")

    oriented: list[tuple[int, int, int]] = []
    for block, sign in zip(blocks, signs):
        ordered = list(block)
        if sign:
            ordered[0], ordered[1] = ordered[1], ordered[0]
        oriented.append(tuple(ordered))
    return oriented


@dataclass(frozen=True)
class CubieMove:
    center_permutation: tuple[int, ...]
    edge_permutation: tuple[int, ...]
    corner_permutation: tuple[int, ...]
    corner_orientation: tuple[int, ...]


@dataclass(frozen=True)
class Decomposition:
    centers: tuple[int, ...]
    edge_primary: tuple[int, ...]
    edge_partner: dict[int, int]
    corners: tuple[tuple[int, int, int], ...]

    def encode_move(self, sticker_permutation: Permutation) -> CubieMove:
        center_index = {point: index for index, point in enumerate(self.centers)}
        edge_index = {point: index for index, point in enumerate(self.edge_primary)}
        corner_location = {
            point: (block, coordinate)
            for block, points in enumerate(self.corners)
            for coordinate, point in enumerate(points)
        }

        centers = tuple(center_index[sticker_permutation[point]] for point in self.centers)
        edges = tuple(edge_index[sticker_permutation[point]] for point in self.edge_primary)
        corner_permutation: list[int] = []
        corner_orientation: list[int] = []
        for points in self.corners:
            images = [corner_location[sticker_permutation[point]] for point in points]
            if len({item[0] for item in images}) != 1:
                raise ValueError("generator splits a corner")
            source = images[0][0]
            coordinates = [item[1] for item in images]
            orientation = coordinates[0]
            if coordinates != [(coordinate + orientation) % 3 for coordinate in range(3)]:
                raise ValueError("generator reflects a corner")
            corner_permutation.append(source)
            corner_orientation.append(orientation)
        return CubieMove(
            centers,
            edges,
            tuple(corner_permutation),
            tuple(corner_orientation),
        )

    def decode_move(self, move: CubieMove) -> Permutation:
        width = len(self.centers) + 2 * len(self.edge_primary) + 3 * len(self.corners)
        result = [-1] * width

        for destination, source in enumerate(move.center_permutation):
            result[self.centers[destination]] = self.centers[source]

        inverse_partner = {partner: primary for primary, partner in self.edge_partner.items()}
        for destination, source in enumerate(move.edge_permutation):
            destination_primary = self.edge_primary[destination]
            source_primary = self.edge_primary[source]
            result[destination_primary] = source_primary
            result[self.edge_partner[destination_primary]] = self.edge_partner[source_primary]
        if set(inverse_partner) != set(self.edge_partner.values()):
            raise AssertionError("invalid edge partner map")

        for destination, source in enumerate(move.corner_permutation):
            orientation = move.corner_orientation[destination]
            for coordinate in range(3):
                result[self.corners[destination][coordinate]] = self.corners[source][
                    (coordinate + orientation) % 3
                ]
        if sorted(result) != list(range(width)):
            raise ValueError("decoded cubie move is not a sticker permutation")
        return tuple(result)


def discover_decomposition(permutations: Sequence[Permutation]) -> Decomposition:
    width = len(permutations[0])
    action = list(permutations) + [inverse(permutation) for permutation in permutations]
    orbits = sticker_orbits(permutations, width)
    if sorted(map(len, orbits)) != [24, 24, 24, 24]:
        raise ValueError(f"unexpected sticker orbits: {list(map(len, orbits))}")

    corner_orbit: list[int] | None = None
    corner_blocks: list[list[int]] | None = None
    for orbit in orbits:
        blocks = find_blocks(orbit, action, 3)
        if blocks is not None:
            if corner_orbit is not None:
                raise ValueError("multiple corner-like sticker orbits")
            corner_orbit = orbit
            corner_blocks = blocks
    if corner_orbit is None or corner_blocks is None:
        raise ValueError("could not discover corner blocks")

    remaining = [orbit for orbit in orbits if orbit is not corner_orbit]
    edge_pair: tuple[list[int], list[int], dict[int, int]] | None = None
    for first_index, first in enumerate(remaining):
        for second in remaining[first_index + 1 :]:
            mapping = equivariant_bijection(first, second, action)
            if mapping is not None:
                if edge_pair is not None:
                    raise ValueError("multiple wing-orbit pairings")
                edge_pair = (first, second, mapping)
    if edge_pair is None:
        raise ValueError("could not pair wing sticker orbits")
    edge_first, edge_second, partner = edge_pair
    centers = next(orbit for orbit in remaining if orbit is not edge_first and orbit is not edge_second)
    corners = orient_corner_blocks(corner_blocks, action)
    return Decomposition(tuple(centers), tuple(edge_first), partner, tuple(corners))


def number_line(values: Iterable[int]) -> str:
    return " ".join(map(str, values))


def definition_text(raw: dict[str, object]) -> str:
    generator_data = raw["generators"]
    assert isinstance(generator_data, dict)
    forward = {
        name: tuple(map(int, values))
        for name, values in generator_data.items()
        if not name.startswith("-")
    }
    decomposition = discover_decomposition(list(forward.values()))
    moves = {name: decomposition.encode_move(permutation) for name, permutation in forward.items()}
    for name, permutation in forward.items():
        decoded = decomposition.decode_move(moves[name])
        if decoded != permutation:
            mismatch = sum(left != right for left, right in zip(decoded, permutation))
            raise ValueError(f"cubie reconstruction failed for {name}: {mismatch} stickers")

    lines = [
        "Name CayleyPy444Cubies",
        "",
        "Set CENTER 24 1",
        "Set EDGE 24 1",
        "Set CORNER 8 3",
        "",
        "Solved",
        "CENTER",
        number_line(range(1, 25)),
        number_line([0] * 24),
        "EDGE",
        number_line(range(1, 25)),
        number_line([0] * 24),
        "CORNER",
        number_line(range(1, 9)),
        number_line([0] * 8),
        "End",
        "",
    ]
    for name, move in moves.items():
        lines.extend(
            [
                f"MoveTransformation {name}",
                "CENTER",
                number_line(move.center_permutation),
                number_line([0] * 24),
                "EDGE",
                number_line(move.edge_permutation),
                number_line([0] * 24),
                "CORNER",
                number_line(move.corner_permutation),
                number_line(move.corner_orientation),
                "End",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--output", type=Path, default=Path("work/cayleypy444_cubies.tws"))
    args = parser.parse_args()
    raw = json.loads(args.puzzle_info.read_text())
    text = definition_text(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
