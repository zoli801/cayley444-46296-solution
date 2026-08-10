import heapq
import math
import time

import torch
from tqdm import tqdm

from .searcher import Searcher
from .qsearcher import QSearcher
from .utils import state2hash


class BestFirstSearcher(Searcher):
    """Bounded batched weighted-A* search guided by a value model."""

    def _make_stats(self):
        return {
            "expanded": 0,
            "generated": 0,
            "scored": 0,
            "max_frontier": 0,
            "nodes": 0,
            "elapsed": 0.0,
        }

    def _reconstruct_moves(self, node_idx, final_move, parents, node_moves):
        moves = [int(final_move)]
        while node_idx >= 0:
            move = int(node_moves[node_idx])
            if move >= 0:
                moves.append(move)
            node_idx = int(parents[node_idx])
        moves.reverse()
        moves_seq = torch.tensor(moves, dtype=torch.int64)
        return self.normalize_moves_seq(moves_seq)

    def _trim_heap(self, heap, max_frontier):
        if max_frontier is None or max_frontier <= 0 or len(heap) <= max_frontier:
            return heap
        heap = heapq.nsmallest(int(max_frontier), heap)
        heapq.heapify(heap)
        return heap

    def get_solution_best_first(
        self,
        state,
        *,
        weight=1.0,
        max_expansions=100000,
        max_depth=200,
        pop_batch_size=256,
        max_frontier=None,
        incumbent_length=None,
    ):
        """Search for a solution with priority ``g + weight * h``.

        This is intentionally bounded and non-optimal: the model heuristic is not
        admissible, and the frontier can be capped. The goal is to test whether a
        global priority queue improves over layer-wise beam search.
        """
        started = time.time()
        stats = self._make_stats()

        if torch.equal(state, self.V0):
            stats["elapsed"] = time.time() - started
            return torch.empty((0,), dtype=torch.int64), stats

        weight = float(weight)
        max_expansions = int(max_expansions)
        max_depth = int(max_depth)
        pop_batch_size = int(pop_batch_size)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be a finite non-negative number")
        if max_expansions <= 0:
            raise ValueError("max_expansions must be > 0")
        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        if pop_batch_size <= 0:
            raise ValueError("pop_batch_size must be > 0")

        root_state = state.detach().to("cpu").clone()
        root_state_device = state.unsqueeze(0).to(self.device)
        root_hash = int(state2hash(root_state_device, self.hash_vec, self.batch_size).item())
        root_h = float(self.pred_d(root_state_device)[0, 0].item())

        node_states = [root_state]
        parents = [-1]
        node_moves = [-1]
        depths = [0]
        hashes = [root_hash]
        best_depth_by_hash = {root_hash: 0}

        heap = [(weight * root_h, root_h, 0, 0)]
        next_serial = 1
        stats["scored"] = 1
        stats["max_frontier"] = 1
        stats["nodes"] = 1

        pbar = tqdm(total=max_expansions, disable=not self.verbose, desc="best-first")

        try:
            while heap and stats["expanded"] < max_expansions:
                selected = []
                while heap and len(selected) < pop_batch_size and stats["expanded"] + len(selected) < max_expansions:
                    _, _, _, node_idx = heapq.heappop(heap)
                    if depths[node_idx] >= max_depth:
                        continue
                    if best_depth_by_hash.get(hashes[node_idx], depths[node_idx]) < depths[node_idx]:
                        continue
                    selected.append(node_idx)

                if not selected:
                    break

                parent_states = torch.stack([node_states[idx] for idx in selected]).to(self.device)
                parent_node_ids = torch.tensor(selected, dtype=torch.int64, device=self.device)
                parent_depths = torch.tensor([depths[idx] for idx in selected], dtype=torch.int64, device=self.device)
                parent_last_moves = torch.tensor([node_moves[idx] for idx in selected], dtype=torch.int64, device=self.device)

                parent_rows = torch.arange(len(selected), device=self.device).repeat_interleave(self.n_gens)
                child_moves = torch.arange(self.n_gens, device=self.device).repeat(len(selected))

                if self.inverse_moves is not None:
                    last = parent_last_moves.index_select(0, parent_rows)
                    valid = torch.ones_like(child_moves, dtype=torch.bool)
                    has_last = last >= 0
                    valid[has_last] = child_moves[has_last] != self.inverse_moves[last[has_last]]
                    parent_rows = parent_rows[valid]
                    child_moves = child_moves[valid]

                if parent_rows.numel() == 0:
                    stats["expanded"] += len(selected)
                    pbar.update(len(selected))
                    continue

                child_depths = parent_depths.index_select(0, parent_rows) + 1
                if incumbent_length is not None:
                    depth_mask = child_depths < int(incumbent_length)
                    parent_rows = parent_rows[depth_mask]
                    child_moves = child_moves[depth_mask]
                    child_depths = child_depths[depth_mask]
                    if parent_rows.numel() == 0:
                        stats["expanded"] += len(selected)
                        pbar.update(len(selected))
                        continue

                child_parent_ids = parent_node_ids.index_select(0, parent_rows)
                child_states = self.apply_move(parent_states.index_select(0, parent_rows), child_moves)
                stats["generated"] += int(child_states.size(0))

                solved = (child_states == self.V0).all(dim=1)
                if solved.any():
                    solved_pos = int(torch.nonzero(solved, as_tuple=False)[0].item())
                    moves_seq = self._reconstruct_moves(
                        int(child_parent_ids[solved_pos].item()),
                        int(child_moves[solved_pos].item()),
                        parents,
                        node_moves,
                    )
                    stats["expanded"] += len(selected)
                    stats["max_frontier"] = max(stats["max_frontier"], len(heap))
                    stats["nodes"] = len(node_states)
                    stats["elapsed"] = time.time() - started
                    pbar.update(len(selected))
                    return moves_seq, stats

                child_hashes = state2hash(child_states, self.hash_vec, self.batch_size).to("cpu").tolist()
                child_depths_cpu = child_depths.to("cpu").tolist()

                keep_positions = []
                seen_in_batch = set()
                for pos, child_hash in enumerate(child_hashes):
                    child_hash = int(child_hash)
                    child_depth = int(child_depths_cpu[pos])
                    if child_hash in seen_in_batch:
                        continue
                    previous_depth = best_depth_by_hash.get(child_hash)
                    if previous_depth is not None and previous_depth <= child_depth:
                        continue
                    seen_in_batch.add(child_hash)
                    best_depth_by_hash[child_hash] = child_depth
                    keep_positions.append(pos)

                if keep_positions:
                    keep_idx = torch.tensor(keep_positions, dtype=torch.int64, device=self.device)
                    kept_states = child_states.index_select(0, keep_idx)
                    kept_h = self.pred_d(kept_states)[0].to(torch.float32).to("cpu")
                    kept_depths = child_depths.index_select(0, keep_idx).to("cpu")
                    kept_parents = child_parent_ids.index_select(0, keep_idx).to("cpu")
                    kept_moves = child_moves.index_select(0, keep_idx).to("cpu")
                    kept_hashes = [int(child_hashes[pos]) for pos in keep_positions]
                    kept_states_cpu = kept_states.to("cpu")
                    stats["scored"] += int(kept_states.size(0))

                    for local_idx in range(kept_states.size(0)):
                        depth = int(kept_depths[local_idx].item())
                        heuristic = float(kept_h[local_idx].item())
                        priority = depth + weight * heuristic
                        node_idx = len(node_states)
                        node_states.append(kept_states_cpu[local_idx].clone())
                        parents.append(int(kept_parents[local_idx].item()))
                        node_moves.append(int(kept_moves[local_idx].item()))
                        depths.append(depth)
                        hashes.append(kept_hashes[local_idx])
                        heapq.heappush(heap, (priority, heuristic, next_serial, node_idx))
                        next_serial += 1

                    heap = self._trim_heap(heap, max_frontier)

                stats["expanded"] += len(selected)
                stats["max_frontier"] = max(stats["max_frontier"], len(heap))
                stats["nodes"] = len(node_states)
                pbar.update(len(selected))
                if self.verbose:
                    pbar.set_postfix(
                        frontier=len(heap),
                        nodes=len(node_states),
                        scored=stats["scored"],
                    )
        finally:
            pbar.close()

        stats["max_frontier"] = max(stats["max_frontier"], len(heap))
        stats["nodes"] = len(node_states)
        stats["elapsed"] = time.time() - started
        return None, stats


class QBestFirstSearcher(QSearcher):
    """Bounded batched best-first search guided by a per-action Q model."""

    def _make_stats(self):
        return {
            "expanded": 0,
            "generated": 0,
            "scored_states": 0,
            "scored_actions": 0,
            "max_frontier": 0,
            "nodes": 0,
            "elapsed": 0.0,
        }

    def _reconstruct_moves(self, node_idx, final_move, parents, node_moves):
        moves = [int(final_move)]
        while node_idx >= 0:
            move = int(node_moves[node_idx])
            if move >= 0:
                moves.append(move)
            node_idx = int(parents[node_idx])
        moves.reverse()
        moves_seq = torch.tensor(moves, dtype=torch.int64)
        return self.normalize_moves_seq(moves_seq)

    def _trim_heap(self, heap, max_frontier):
        if max_frontier is None or max_frontier <= 0 or len(heap) <= max_frontier:
            return heap
        heap = heapq.nsmallest(int(max_frontier), heap)
        heapq.heapify(heap)
        return heap

    def get_solution_q_best_first(
        self,
        state,
        *,
        weight=1.0,
        max_expansions=100000,
        max_depth=200,
        pop_batch_size=1024,
        top_actions_per_parent=None,
        max_frontier=None,
        incumbent_length=None,
    ):
        """Search with priority ``g + weight * q(parent, move)``.

        The Q model is assumed to output child-state costs for every move from a
        parent state. Lower values are better, matching ``QSearcher`` beam logic.
        """
        started = time.time()
        stats = self._make_stats()

        if torch.equal(state, self.V0):
            stats["elapsed"] = time.time() - started
            return torch.empty((0,), dtype=torch.int64), stats

        weight = float(weight)
        max_expansions = int(max_expansions)
        max_depth = int(max_depth)
        pop_batch_size = int(pop_batch_size)
        if top_actions_per_parent is not None:
            top_actions_per_parent = int(top_actions_per_parent)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be a finite non-negative number")
        if max_expansions <= 0:
            raise ValueError("max_expansions must be > 0")
        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        if pop_batch_size <= 0:
            raise ValueError("pop_batch_size must be > 0")
        if top_actions_per_parent is not None and top_actions_per_parent <= 0:
            raise ValueError("top_actions_per_parent must be > 0")

        root_state = state.detach().to("cpu").clone()
        root_hash = int(state2hash(state.unsqueeze(0).to(self.device), self.hash_vec, self.batch_size).item())

        node_states = [root_state]
        parents = [-1]
        node_moves = [-1]
        depths = [0]
        hashes = [root_hash]
        best_depth_by_hash = {root_hash: 0}

        heap = [(0.0, 0.0, 0, 0)]
        next_serial = 1
        stats["max_frontier"] = 1
        stats["nodes"] = 1

        pbar = tqdm(total=max_expansions, disable=not self.verbose, desc="q-best-first")

        try:
            while heap and stats["expanded"] < max_expansions:
                selected = []
                while heap and len(selected) < pop_batch_size and stats["expanded"] + len(selected) < max_expansions:
                    _, _, _, node_idx = heapq.heappop(heap)
                    if depths[node_idx] >= max_depth:
                        continue
                    if best_depth_by_hash.get(hashes[node_idx], depths[node_idx]) < depths[node_idx]:
                        continue
                    selected.append(node_idx)

                if not selected:
                    break

                parent_states = torch.stack([node_states[idx] for idx in selected]).to(self.device)
                parent_node_ids = torch.tensor(selected, dtype=torch.int64, device=self.device)
                parent_depths = torch.tensor([depths[idx] for idx in selected], dtype=torch.int64, device=self.device)
                parent_last_moves = torch.tensor([node_moves[idx] for idx in selected], dtype=torch.int64, device=self.device)

                q_values = self.pred_q(parent_states).to(torch.float32)
                stats["scored_states"] += int(parent_states.size(0))
                valid_mask = self._valid_action_mask(parent_states, parent_last_moves)
                stats["scored_actions"] += int(valid_mask.sum().item())
                q_values = q_values.masked_fill(~valid_mask, float("inf"))

                if top_actions_per_parent is None or top_actions_per_parent >= self.n_gens:
                    flat = torch.nonzero(torch.isfinite(q_values).reshape(-1), as_tuple=False).view(-1)
                    q_flat = q_values.reshape(-1)
                    candidate_values = q_flat.index_select(0, flat)
                    candidate_flat = flat
                else:
                    k = min(int(top_actions_per_parent), self.n_gens)
                    candidate_values_2d, candidate_moves_2d = torch.topk(q_values, k=k, largest=False, sorted=True)
                    finite = torch.isfinite(candidate_values_2d)
                    if not finite.any():
                        stats["expanded"] += len(selected)
                        pbar.update(len(selected))
                        continue
                    rows = torch.arange(parent_states.size(0), device=self.device).unsqueeze(1).expand(-1, k)
                    candidate_flat = (rows[finite] * self.n_gens) + candidate_moves_2d[finite]
                    candidate_values = candidate_values_2d[finite]

                if candidate_flat.numel() == 0:
                    stats["expanded"] += len(selected)
                    pbar.update(len(selected))
                    continue

                parent_rows = torch.div(candidate_flat, self.n_gens, rounding_mode="floor")
                child_moves = torch.remainder(candidate_flat, self.n_gens)
                child_depths = parent_depths.index_select(0, parent_rows) + 1

                if incumbent_length is not None:
                    depth_mask = child_depths < int(incumbent_length)
                    parent_rows = parent_rows[depth_mask]
                    child_moves = child_moves[depth_mask]
                    child_depths = child_depths[depth_mask]
                    candidate_values = candidate_values[depth_mask]
                    if parent_rows.numel() == 0:
                        stats["expanded"] += len(selected)
                        pbar.update(len(selected))
                        continue

                child_parent_ids = parent_node_ids.index_select(0, parent_rows)
                child_states = self.apply_move(parent_states.index_select(0, parent_rows), child_moves)
                stats["generated"] += int(child_states.size(0))

                solved = (child_states == self.V0).all(dim=1)
                if solved.any():
                    solved_pos = int(torch.nonzero(solved, as_tuple=False)[0].item())
                    moves_seq = self._reconstruct_moves(
                        int(child_parent_ids[solved_pos].item()),
                        int(child_moves[solved_pos].item()),
                        parents,
                        node_moves,
                    )
                    stats["expanded"] += len(selected)
                    stats["max_frontier"] = max(stats["max_frontier"], len(heap))
                    stats["nodes"] = len(node_states)
                    stats["elapsed"] = time.time() - started
                    pbar.update(len(selected))
                    return moves_seq, stats

                child_hashes = state2hash(child_states, self.hash_vec, self.batch_size).to("cpu").tolist()
                child_depths_cpu = child_depths.to("cpu").tolist()

                keep_positions = []
                seen_in_batch = set()
                for pos, child_hash in enumerate(child_hashes):
                    child_hash = int(child_hash)
                    child_depth = int(child_depths_cpu[pos])
                    if child_hash in seen_in_batch:
                        continue
                    previous_depth = best_depth_by_hash.get(child_hash)
                    if previous_depth is not None and previous_depth <= child_depth:
                        continue
                    seen_in_batch.add(child_hash)
                    best_depth_by_hash[child_hash] = child_depth
                    keep_positions.append(pos)

                if keep_positions:
                    keep_idx = torch.tensor(keep_positions, dtype=torch.int64, device=self.device)
                    kept_states_cpu = child_states.index_select(0, keep_idx).to("cpu")
                    kept_depths = child_depths.index_select(0, keep_idx).to("cpu")
                    kept_parents = child_parent_ids.index_select(0, keep_idx).to("cpu")
                    kept_moves = child_moves.index_select(0, keep_idx).to("cpu")
                    kept_q = candidate_values.index_select(0, keep_idx).to("cpu")
                    kept_hashes = [int(child_hashes[pos]) for pos in keep_positions]

                    for local_idx in range(kept_states_cpu.size(0)):
                        depth = int(kept_depths[local_idx].item())
                        q_value = float(kept_q[local_idx].item())
                        priority = depth + weight * q_value
                        node_idx = len(node_states)
                        node_states.append(kept_states_cpu[local_idx].clone())
                        parents.append(int(kept_parents[local_idx].item()))
                        node_moves.append(int(kept_moves[local_idx].item()))
                        depths.append(depth)
                        hashes.append(kept_hashes[local_idx])
                        heapq.heappush(heap, (priority, q_value, next_serial, node_idx))
                        next_serial += 1

                    heap = self._trim_heap(heap, max_frontier)

                stats["expanded"] += len(selected)
                stats["max_frontier"] = max(stats["max_frontier"], len(heap))
                stats["nodes"] = len(node_states)
                pbar.update(len(selected))
                if self.verbose:
                    pbar.set_postfix(
                        frontier=len(heap),
                        nodes=len(node_states),
                        scored_states=stats["scored_states"],
                    )
        finally:
            pbar.close()

        stats["max_frontier"] = max(stats["max_frontier"], len(heap))
        stats["nodes"] = len(node_states)
        stats["elapsed"] = time.time() - started
        return None, stats


class QDepthBucketSearcher(QBestFirstSearcher):
    """Q-guided multi-queue search with depth buckets plus a global heap."""

    def _make_bucket_stats(self):
        return {
            "expanded": 0,
            "expanded_bucket": 0,
            "expanded_global": 0,
            "generated": 0,
            "scored_states": 0,
            "scored_actions": 0,
            "max_global_frontier": 0,
            "max_bucket_frontier": 0,
            "max_frontier": 0,
            "nodes": 0,
            "max_depth_seen": 0,
            "elapsed": 0.0,
        }

    def _pop_global(self, heap, depths, hashes, best_depth_by_hash, expanded_nodes, max_depth, count):
        selected = []
        while heap and len(selected) < count:
            _, _, _, node_idx = heapq.heappop(heap)
            if expanded_nodes[node_idx]:
                continue
            if depths[node_idx] >= max_depth:
                continue
            if best_depth_by_hash.get(hashes[node_idx], depths[node_idx]) < depths[node_idx]:
                continue
            selected.append(node_idx)
        return selected

    def _pop_bucket(
        self,
        buckets,
        current_depth,
        depths,
        hashes,
        best_depth_by_hash,
        expanded_nodes,
        max_depth,
        count,
        depth_quota=None,
        bucket_depth_expanded=None,
    ):
        depth = int(current_depth)
        while depth < max_depth:
            if depth_quota is not None and bucket_depth_expanded is not None:
                if bucket_depth_expanded.get(depth, 0) >= depth_quota:
                    buckets.pop(depth, None)
                    depth += 1
                    continue
            heap = buckets.get(depth)
            if heap:
                break
            depth += 1

        if depth >= max_depth:
            return [], depth, None

        selected = []
        heap = buckets[depth]
        selected_depth = depth
        if depth_quota is not None and bucket_depth_expanded is not None:
            remaining_quota = max(0, int(depth_quota) - int(bucket_depth_expanded.get(depth, 0)))
            count = min(int(count), remaining_quota)

        while heap and len(selected) < count:
            _, _, _, node_idx = heapq.heappop(heap)
            if expanded_nodes[node_idx]:
                continue
            if depths[node_idx] != depth:
                continue
            if depths[node_idx] >= max_depth:
                continue
            if best_depth_by_hash.get(hashes[node_idx], depths[node_idx]) < depths[node_idx]:
                continue
            selected.append(node_idx)

        if not heap:
            buckets.pop(depth, None)
            depth += 1
        elif depth_quota is not None and bucket_depth_expanded is not None:
            if bucket_depth_expanded.get(selected_depth, 0) + len(selected) >= depth_quota:
                buckets.pop(selected_depth, None)
                depth = selected_depth + 1

        return selected, depth, selected_depth

    def _bucket_size(self, buckets):
        return sum(len(heap) for heap in buckets.values())

    def get_solution_q_depth_bucket(
        self,
        state,
        *,
        weight=1.0,
        max_expansions=100000,
        max_depth=200,
        pop_batch_size=1024,
        top_actions_per_parent=None,
        bucket_batches=3,
        global_batches=1,
        bucket_depth_quota=None,
        max_frontier=None,
        incumbent_length=None,
    ):
        """Search with shallow depth buckets interleaved with global Q priority.

        Bucket batches expand the best Q-scored nodes at the current shallowest
        frontier depth. Global batches expand nodes by ``g + weight*q``. This
        keeps some beam-like layer coverage while still letting strong Q choices
        skip ahead.
        """
        started = time.time()
        stats = self._make_bucket_stats()

        if torch.equal(state, self.V0):
            stats["elapsed"] = time.time() - started
            return torch.empty((0,), dtype=torch.int64), stats

        weight = float(weight)
        max_expansions = int(max_expansions)
        max_depth = int(max_depth)
        pop_batch_size = int(pop_batch_size)
        bucket_batches = int(bucket_batches)
        global_batches = int(global_batches)
        if top_actions_per_parent is not None:
            top_actions_per_parent = int(top_actions_per_parent)
        if bucket_depth_quota is not None:
            bucket_depth_quota = int(bucket_depth_quota)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be a finite non-negative number")
        if max_expansions <= 0:
            raise ValueError("max_expansions must be > 0")
        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        if pop_batch_size <= 0:
            raise ValueError("pop_batch_size must be > 0")
        if bucket_batches < 0 or global_batches < 0 or bucket_batches + global_batches <= 0:
            raise ValueError("at least one of bucket_batches/global_batches must be > 0")
        if top_actions_per_parent is not None and top_actions_per_parent <= 0:
            raise ValueError("top_actions_per_parent must be > 0")
        if bucket_depth_quota is not None and bucket_depth_quota <= 0:
            raise ValueError("bucket_depth_quota must be > 0")

        root_state = state.detach().to("cpu").clone()
        root_hash = int(state2hash(state.unsqueeze(0).to(self.device), self.hash_vec, self.batch_size).item())

        node_states = [root_state]
        parents = [-1]
        node_moves = [-1]
        depths = [0]
        hashes = [root_hash]
        expanded_nodes = [False]
        best_depth_by_hash = {root_hash: 0}

        global_heap = [(0.0, 0.0, 0, 0)]
        buckets = {0: [(0.0, 0.0, 0, 0)]}
        bucket_depth_expanded = {}
        current_bucket_depth = 0
        next_serial = 1
        batch_index = 0
        cycle = bucket_batches + global_batches

        stats["max_global_frontier"] = 1
        stats["max_bucket_frontier"] = 1
        stats["max_frontier"] = 2
        stats["nodes"] = 1

        pbar = tqdm(total=max_expansions, disable=not self.verbose, desc="q-depth-bucket")

        try:
            while (global_heap or buckets) and stats["expanded"] < max_expansions:
                prefer_bucket = (batch_index % cycle) < bucket_batches if bucket_batches > 0 else False
                batch_index += 1
                room = max_expansions - stats["expanded"]
                batch_limit = min(pop_batch_size, room)

                if prefer_bucket:
                    selected, current_bucket_depth, selected_bucket_depth = self._pop_bucket(
                        buckets,
                        current_bucket_depth,
                        depths,
                        hashes,
                        best_depth_by_hash,
                        expanded_nodes,
                        max_depth,
                        batch_limit,
                        depth_quota=bucket_depth_quota,
                        bucket_depth_expanded=bucket_depth_expanded,
                    )
                    source = "bucket"
                    if not selected and global_batches > 0:
                        selected = self._pop_global(
                            global_heap,
                            depths,
                            hashes,
                            best_depth_by_hash,
                            expanded_nodes,
                            max_depth,
                            batch_limit,
                        )
                        source = "global"
                        selected_bucket_depth = None
                else:
                    selected = self._pop_global(
                        global_heap,
                        depths,
                        hashes,
                        best_depth_by_hash,
                        expanded_nodes,
                        max_depth,
                        batch_limit,
                    )
                    source = "global"
                    selected_bucket_depth = None
                    if not selected and bucket_batches > 0:
                        selected, current_bucket_depth, selected_bucket_depth = self._pop_bucket(
                            buckets,
                            current_bucket_depth,
                            depths,
                            hashes,
                            best_depth_by_hash,
                            expanded_nodes,
                            max_depth,
                            batch_limit,
                            depth_quota=bucket_depth_quota,
                            bucket_depth_expanded=bucket_depth_expanded,
                        )
                        source = "bucket"

                if not selected:
                    break

                for node_idx in selected:
                    expanded_nodes[node_idx] = True

                if source == "bucket":
                    stats["expanded_bucket"] += len(selected)
                    if selected_bucket_depth is not None:
                        bucket_depth_expanded[selected_bucket_depth] = (
                            bucket_depth_expanded.get(selected_bucket_depth, 0) + len(selected)
                        )
                else:
                    stats["expanded_global"] += len(selected)

                parent_states = torch.stack([node_states[idx] for idx in selected]).to(self.device)
                parent_node_ids = torch.tensor(selected, dtype=torch.int64, device=self.device)
                parent_depths = torch.tensor([depths[idx] for idx in selected], dtype=torch.int64, device=self.device)
                parent_last_moves = torch.tensor([node_moves[idx] for idx in selected], dtype=torch.int64, device=self.device)

                q_values = self.pred_q(parent_states).to(torch.float32)
                stats["scored_states"] += int(parent_states.size(0))
                valid_mask = self._valid_action_mask(parent_states, parent_last_moves)
                stats["scored_actions"] += int(valid_mask.sum().item())
                q_values = q_values.masked_fill(~valid_mask, float("inf"))

                if top_actions_per_parent is None or top_actions_per_parent >= self.n_gens:
                    flat = torch.nonzero(torch.isfinite(q_values).reshape(-1), as_tuple=False).view(-1)
                    q_flat = q_values.reshape(-1)
                    candidate_values = q_flat.index_select(0, flat)
                    candidate_flat = flat
                else:
                    k = min(int(top_actions_per_parent), self.n_gens)
                    candidate_values_2d, candidate_moves_2d = torch.topk(q_values, k=k, largest=False, sorted=True)
                    finite = torch.isfinite(candidate_values_2d)
                    if not finite.any():
                        stats["expanded"] += len(selected)
                        pbar.update(len(selected))
                        continue
                    rows = torch.arange(parent_states.size(0), device=self.device).unsqueeze(1).expand(-1, k)
                    candidate_flat = (rows[finite] * self.n_gens) + candidate_moves_2d[finite]
                    candidate_values = candidate_values_2d[finite]

                if candidate_flat.numel() == 0:
                    stats["expanded"] += len(selected)
                    pbar.update(len(selected))
                    continue

                parent_rows = torch.div(candidate_flat, self.n_gens, rounding_mode="floor")
                child_moves = torch.remainder(candidate_flat, self.n_gens)
                child_depths = parent_depths.index_select(0, parent_rows) + 1

                if incumbent_length is not None:
                    depth_mask = child_depths < int(incumbent_length)
                    parent_rows = parent_rows[depth_mask]
                    child_moves = child_moves[depth_mask]
                    child_depths = child_depths[depth_mask]
                    candidate_values = candidate_values[depth_mask]
                    if parent_rows.numel() == 0:
                        stats["expanded"] += len(selected)
                        pbar.update(len(selected))
                        continue

                child_parent_ids = parent_node_ids.index_select(0, parent_rows)
                child_states = self.apply_move(parent_states.index_select(0, parent_rows), child_moves)
                stats["generated"] += int(child_states.size(0))

                solved = (child_states == self.V0).all(dim=1)
                if solved.any():
                    solved_pos = int(torch.nonzero(solved, as_tuple=False)[0].item())
                    moves_seq = self._reconstruct_moves(
                        int(child_parent_ids[solved_pos].item()),
                        int(child_moves[solved_pos].item()),
                        parents,
                        node_moves,
                    )
                    stats["expanded"] += len(selected)
                    stats["max_global_frontier"] = max(stats["max_global_frontier"], len(global_heap))
                    bucket_size = self._bucket_size(buckets)
                    stats["max_bucket_frontier"] = max(stats["max_bucket_frontier"], bucket_size)
                    stats["max_frontier"] = max(stats["max_frontier"], len(global_heap) + bucket_size)
                    stats["nodes"] = len(node_states)
                    stats["elapsed"] = time.time() - started
                    pbar.update(len(selected))
                    return moves_seq, stats

                child_hashes = state2hash(child_states, self.hash_vec, self.batch_size).to("cpu").tolist()
                child_depths_cpu = child_depths.to("cpu").tolist()

                keep_positions = []
                seen_in_batch = set()
                for pos, child_hash in enumerate(child_hashes):
                    child_hash = int(child_hash)
                    child_depth = int(child_depths_cpu[pos])
                    if child_hash in seen_in_batch:
                        continue
                    previous_depth = best_depth_by_hash.get(child_hash)
                    if previous_depth is not None and previous_depth <= child_depth:
                        continue
                    seen_in_batch.add(child_hash)
                    best_depth_by_hash[child_hash] = child_depth
                    keep_positions.append(pos)

                if keep_positions:
                    keep_idx = torch.tensor(keep_positions, dtype=torch.int64, device=self.device)
                    kept_states_cpu = child_states.index_select(0, keep_idx).to("cpu")
                    kept_depths = child_depths.index_select(0, keep_idx).to("cpu")
                    kept_parents = child_parent_ids.index_select(0, keep_idx).to("cpu")
                    kept_moves = child_moves.index_select(0, keep_idx).to("cpu")
                    kept_q = candidate_values.index_select(0, keep_idx).to("cpu")
                    kept_hashes = [int(child_hashes[pos]) for pos in keep_positions]

                    for local_idx in range(kept_states_cpu.size(0)):
                        depth = int(kept_depths[local_idx].item())
                        q_value = float(kept_q[local_idx].item())
                        priority = depth + weight * q_value
                        node_idx = len(node_states)
                        node_states.append(kept_states_cpu[local_idx].clone())
                        parents.append(int(kept_parents[local_idx].item()))
                        node_moves.append(int(kept_moves[local_idx].item()))
                        depths.append(depth)
                        hashes.append(kept_hashes[local_idx])
                        expanded_nodes.append(False)
                        heapq.heappush(global_heap, (priority, q_value, next_serial, node_idx))
                        heapq.heappush(buckets.setdefault(depth, []), (q_value, priority, next_serial, node_idx))
                        stats["max_depth_seen"] = max(stats["max_depth_seen"], depth)
                        next_serial += 1

                    global_heap = self._trim_heap(global_heap, max_frontier)

                stats["expanded"] += len(selected)
                bucket_size = self._bucket_size(buckets)
                stats["max_global_frontier"] = max(stats["max_global_frontier"], len(global_heap))
                stats["max_bucket_frontier"] = max(stats["max_bucket_frontier"], bucket_size)
                stats["max_frontier"] = max(stats["max_frontier"], len(global_heap) + bucket_size)
                stats["nodes"] = len(node_states)
                pbar.update(len(selected))
                if self.verbose:
                    pbar.set_postfix(
                        global_frontier=len(global_heap),
                        bucket_frontier=bucket_size,
                        depth=current_bucket_depth,
                    )
        finally:
            pbar.close()

        bucket_size = self._bucket_size(buckets)
        stats["max_global_frontier"] = max(stats["max_global_frontier"], len(global_heap))
        stats["max_bucket_frontier"] = max(stats["max_bucket_frontier"], bucket_size)
        stats["max_frontier"] = max(stats["max_frontier"], len(global_heap) + bucket_size)
        stats["nodes"] = len(node_states)
        stats["elapsed"] = time.time() - started
        return None, stats
