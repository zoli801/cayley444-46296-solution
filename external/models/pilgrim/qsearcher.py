import torch
import math
from collections import deque
from tqdm import tqdm

from .model import batch_process
from .searcher import Searcher
from .utils import state2hash


class QSearcher(Searcher):
    @staticmethod
    def _sorted_isin(values, sorted_reference):
        """Memory-light membership test assuming sorted_reference is unique and sorted."""
        if sorted_reference.numel() == 0:
            return torch.zeros(values.size(0), dtype=torch.bool, device=values.device)

        pos = torch.searchsorted(sorted_reference, values)
        valid = pos < sorted_reference.numel()
        if not valid.any():
            return valid

        check_pos = pos.clamp_max(sorted_reference.numel() - 1)
        matched = sorted_reference.index_select(0, check_pos) == values
        return valid & matched

    def pred_q(self, states):
        """Predict per-action child-state costs for states using the model."""
        pred = batch_process(self.model, states, self.device, self.batch_size)
        if pred.ndim != 2 or pred.size(1) != self.n_gens:
            raise ValueError(
                f"QSearcher expects model output shape [batch, {self.n_gens}], got {tuple(pred.shape)}"
            )
        return pred

    def _valid_action_mask(self, states, last_moves):
        mask = torch.ones((states.size(0), self.n_gens), dtype=torch.bool, device=self.device)
        if last_moves is not None and self.inverse_moves is not None:
            valid_last = last_moves >= 0
            if valid_last.any():
                rows = torch.nonzero(valid_last, as_tuple=False).view(-1)
                mask[rows, self.inverse_moves[last_moves.index_select(0, rows)]] = False
        return mask

    def _find_solved_child(self, states, valid_mask, q_values):
        if self.goal_test is not None:
            return None
        if self.solved_predecessors is None or self.solved_predecessor_hashes is None:
            return None

        parent_hashes = state2hash(states, self.hash_vec, self.batch_size)
        solved_mask = (parent_hashes.unsqueeze(1) == self.solved_predecessor_hashes.unsqueeze(0)) & valid_mask
        if not solved_mask.any():
            return None

        flat_matches = torch.nonzero(solved_mask.view(-1), as_tuple=False).view(-1)
        parents = torch.div(flat_matches, self.n_gens, rounding_mode="floor")
        moves = torch.remainder(flat_matches, self.n_gens)
        exact_mask = (
            states.index_select(0, parents) == self.solved_predecessors.index_select(0, moves)
        ).all(dim=1)
        if not exact_mask.any():
            return None

        solved_flat = flat_matches[torch.nonzero(exact_mask, as_tuple=False).view(-1)[0]]
        solved_parent = torch.div(solved_flat, self.n_gens, rounding_mode="floor").view(1)
        solved_move = torch.remainder(solved_flat, self.n_gens).view(1)
        solved_value = q_values[solved_parent, solved_move]
        return (
            self.V0.unsqueeze(0),
            self.V0_hash,
            solved_value.view(1),
            solved_move,
            solved_parent,
        )

    def _candidate_hashes(self, states, flat_indices):
        hashes = torch.empty(flat_indices.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, flat_indices.size(0), self.batch_size):
            batch_flat = flat_indices[i : i + self.batch_size]
            batch_parent = torch.div(batch_flat, self.n_gens, rounding_mode="floor")
            batch_moves = torch.remainder(batch_flat, self.n_gens)
            parent_states = states.index_select(0, batch_parent)
            hashes[i : i + self.batch_size] = self.hash_after_move(parent_states, batch_moves)
        return hashes

    def _first_unique_positions(self, hashes):
        positions = torch.arange(hashes.size(0), dtype=torch.int64, device=self.device)
        unique_hashes, inverse = torch.unique(hashes, return_inverse=True)
        first_pos = torch.full((unique_hashes.size(0),), hashes.size(0), dtype=torch.int64, device=self.device)
        first_pos.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
        return positions[first_pos.index_select(0, inverse) == positions]

    def _select_unique_candidates(self, states, q_flat, states_bad_hashed, B, total_valid):
        if total_valid <= 0:
            return None, None, None

        current_k = min(int(total_valid), max(int(B) * 2, 16384))
        final_unique_count = 0

        while True:
            candidate_values, candidate_indices = torch.topk(
                q_flat,
                k=current_k,
                largest=False,
                sorted=True,
            )
            del candidate_values
            candidate_hashes = self._candidate_hashes(states, candidate_indices)
            if states_bad_hashed.numel() > 0:
                bad_mask = self._sorted_isin(candidate_hashes, states_bad_hashed)
                candidate_indices = candidate_indices[~bad_mask]
                candidate_hashes = candidate_hashes[~bad_mask]

            if candidate_indices.numel() == 0:
                if current_k == total_valid:
                    return None, None, None
            else:
                unique_pos = self._first_unique_positions(candidate_hashes)
                final_unique_count = int(unique_pos.numel())
                if final_unique_count >= B or current_k == total_valid:
                    selected_pos = unique_pos[:B]
                    return (
                        candidate_indices.index_select(0, selected_pos),
                        candidate_hashes.index_select(0, selected_pos),
                        final_unique_count,
                    )

            if current_k == total_valid:
                return None, None, None
            current_k = min(int(total_valid), current_k * 2)

    def do_greedy_step(self, states, states_bad_hashed, last_moves=None, B=1000):
        """Perform one beam step using parent-state Q outputs instead of child-state V inference."""
        q_values = self.pred_q(states)
        valid_mask = self._valid_action_mask(states, last_moves)
        solved_result = self._find_solved_child(states, valid_mask, q_values)
        if solved_result is not None:
            self.counter[0, 0] += int(valid_mask.sum().item())
            self.counter[0, 1] += 1
            self.counter[1, 0] += 1
            self.counter[1, 1] += 1
            self.counter[2, 0] += 1
            self.counter[2, 1] += 1
            return solved_result

        q_flat = q_values.masked_fill(~valid_mask, float("inf")).reshape(-1)
        total_valid = int(valid_mask.sum().item())
        self.counter[0, 0] += total_valid
        self.counter[0, 1] += 1

        if total_valid == 0:
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_moves = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float32, device=self.device)
            return empty_states, empty_hashes, empty_values, empty_moves, empty_moves

        keep, keep_hashes, unique_count = self._select_unique_candidates(states, q_flat, states_bad_hashed, B, total_valid)
        self.counter[1, 0] += 0 if unique_count is None else unique_count
        self.counter[1, 1] += 1

        if keep is None or keep.numel() == 0:
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_moves = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float32, device=self.device)
            return empty_states, empty_hashes, empty_values, empty_moves, empty_moves

        keep_parent = torch.div(keep, self.n_gens, rounding_mode="floor")
        keep_moves = torch.remainder(keep, self.n_gens)
        selected_states = torch.empty((keep.size(0), self.state_size), dtype=states.dtype, device=self.device)
        for i in range(0, keep.size(0), self.batch_size):
            batch_keep = keep[i : i + self.batch_size]
            batch_parent = keep_parent[i : i + self.batch_size]
            batch_moves = keep_moves[i : i + self.batch_size]
            selected_states[i : i + self.batch_size] = self.apply_move(
                states.index_select(0, batch_parent),
                batch_moves,
            )

        self.counter[2, 0] += keep.size(0)
        self.counter[2, 1] += 1

        return (
            selected_states,
            keep_hashes,
            q_flat[keep],
            keep_moves,
            keep_parent,
        )


class QVRerankSearcher(QSearcher):
    """Q shortlist followed by scalar V reranking."""

    def __init__(self, *args, rerank_model, qshort_alpha=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rerank_model = rerank_model.to(self.device)
        self.qshort_alpha = float(qshort_alpha)
        if self.qshort_alpha < 1.0:
            raise ValueError("qshort_alpha must be >= 1")

    def pred_v(self, states):
        pred = batch_process(self.rerank_model, states, self.device, self.batch_size)
        if pred.ndim != 1:
            pred = pred.view(-1)
        return pred

    def do_greedy_step(self, states, states_bad_hashed, last_moves=None, B=1000):
        q_values = self.pred_q(states)
        valid_mask = self._valid_action_mask(states, last_moves)
        solved_result = self._find_solved_child(states, valid_mask, q_values)
        if solved_result is not None:
            self.counter[0, 0] += int(valid_mask.sum().item())
            self.counter[0, 1] += 1
            self.counter[1, 0] += 1
            self.counter[1, 1] += 1
            self.counter[2, 0] += 1
            self.counter[2, 1] += 1
            return solved_result

        q_flat = q_values.masked_fill(~valid_mask, float("inf")).reshape(-1)
        total_valid = int(valid_mask.sum().item())
        self.counter[0, 0] += total_valid
        self.counter[0, 1] += 1

        if total_valid == 0:
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_moves = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float32, device=self.device)
            return empty_states, empty_hashes, empty_values, empty_moves, empty_moves

        shortlist_size = min(int(total_valid), max(int(B), math.ceil(int(B) * self.qshort_alpha)))
        keep, keep_hashes, unique_count = self._select_unique_candidates(
            states,
            q_flat,
            states_bad_hashed,
            shortlist_size,
            total_valid,
        )
        self.counter[1, 0] += 0 if unique_count is None else unique_count
        self.counter[1, 1] += 1

        if keep is None or keep.numel() == 0:
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_moves = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float32, device=self.device)
            return empty_states, empty_hashes, empty_values, empty_moves, empty_moves

        keep_parent = torch.div(keep, self.n_gens, rounding_mode="floor")
        keep_moves = torch.remainder(keep, self.n_gens)
        shortlist_states = torch.empty((keep.size(0), self.state_size), dtype=states.dtype, device=self.device)
        for i in range(0, keep.size(0), self.batch_size):
            batch_parent = keep_parent[i : i + self.batch_size]
            batch_moves = keep_moves[i : i + self.batch_size]
            shortlist_states[i : i + self.batch_size] = self.apply_move(
                states.index_select(0, batch_parent),
                batch_moves,
            )

        v_values = self.pred_v(shortlist_states).to(torch.float32)
        solved_mask = self.goal_mask(shortlist_states)
        v_values = torch.where(solved_mask, v_values.new_zeros(()), v_values)
        selected = torch.topk(v_values, k=min(int(B), v_values.numel()), largest=False, sorted=False).indices

        self.counter[2, 0] += selected.size(0)
        self.counter[2, 1] += 1

        return (
            shortlist_states.index_select(0, selected),
            keep_hashes.index_select(0, selected),
            v_values.index_select(0, selected),
            keep_moves.index_select(0, selected),
            keep_parent.index_select(0, selected),
        )


class QCPUOffloadSearcher(QSearcher):
    """Q beam search with beam storage and candidate buffers on CPU.

    This mode is meant for very large beams that do not fit in VRAM. It keeps
    the model and per-batch Q inference on GPU, while states, traceback rows,
    candidate buffers, and optional visited hashes live on CPU.
    """

    def __init__(
        self,
        *args,
        cpu_candidate_buffer_factor=2.0,
        cpu_batch_topk=None,
        cpu_batch_topk_factor=1.25,
        cpu_candidate_prune_margin=1.05,
        cpu_dedup_candidates=True,
        cpu_visited_mode="none",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.goal_test is not None:
            raise ValueError("goal_test is not implemented for CPU-offload beam search")
        self.cpu_candidate_buffer_factor = float(cpu_candidate_buffer_factor)
        if self.cpu_candidate_buffer_factor < 1.0:
            raise ValueError("cpu_candidate_buffer_factor must be >= 1.0")
        self.cpu_batch_topk = None if cpu_batch_topk is None else int(cpu_batch_topk)
        if self.cpu_batch_topk is not None and self.cpu_batch_topk <= 0:
            raise ValueError("cpu_batch_topk must be > 0")
        self.cpu_batch_topk_factor = float(cpu_batch_topk_factor)
        if self.cpu_batch_topk_factor <= 0.0:
            raise ValueError("cpu_batch_topk_factor must be > 0")
        self.cpu_candidate_prune_margin = float(cpu_candidate_prune_margin)
        if self.cpu_candidate_prune_margin < 1.0:
            raise ValueError("cpu_candidate_prune_margin must be >= 1.0")
        self.cpu_dedup_candidates = bool(cpu_dedup_candidates)
        if cpu_visited_mode not in {"none", "recent", "all"}:
            raise ValueError("cpu_visited_mode must be one of: none, recent, all")
        self.cpu_visited_mode = cpu_visited_mode
        self.cpu_device = torch.device("cpu")
        self.all_moves_cpu = self.all_moves.to("cpu")
        self.V0_cpu = self.V0.to("cpu")
        self.inverse_moves_cpu = None if self.inverse_moves is None else self.inverse_moves.to("cpu")

    @staticmethod
    def _cpu_sorted_isin(values, sorted_reference):
        if sorted_reference is None or sorted_reference.numel() == 0:
            return torch.zeros(values.size(0), dtype=torch.bool)
        pos = torch.searchsorted(sorted_reference, values)
        valid = pos < sorted_reference.numel()
        if not valid.any():
            return valid
        check_pos = pos.clamp_max(sorted_reference.numel() - 1)
        return valid & (sorted_reference.index_select(0, check_pos) == values)

    def _cpu_hash_after_move_from_device_batch(self, batch_states, parent_idx, moves):
        hashes = torch.empty(parent_idx.size(0), dtype=torch.int64, device=self.device)
        for start in range(0, parent_idx.size(0), self.batch_size):
            end = start + self.batch_size
            chunk_parent = parent_idx[start:end]
            chunk_moves = moves[start:end]
            parent_states = batch_states.index_select(0, chunk_parent)
            hashes[start:end] = self.hash_after_move(parent_states, chunk_moves)
        return hashes

    def _prune_cpu_candidates(self, candidate_buffer, limit):
        if self.cpu_dedup_candidates and "hashes" in candidate_buffer and candidate_buffer["hashes"].numel() > 0:
            order = torch.argsort(candidate_buffer["values"], stable=True)
            candidate_buffer = {key: value.index_select(0, order) for key, value in candidate_buffer.items()}
            unique_hashes, inverse = torch.unique(candidate_buffer["hashes"], return_inverse=True)
            positions = torch.arange(candidate_buffer["hashes"].size(0), dtype=torch.int64)
            first_pos = torch.full((unique_hashes.size(0),), candidate_buffer["hashes"].size(0), dtype=torch.int64)
            first_pos.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
            keep_unique = torch.zeros(candidate_buffer["hashes"].size(0), dtype=torch.bool)
            keep_unique[first_pos] = True
            keep = torch.nonzero(keep_unique, as_tuple=False).view(-1)
            candidate_buffer = {key: value.index_select(0, keep) for key, value in candidate_buffer.items()}

        if candidate_buffer["values"].numel() <= limit:
            return candidate_buffer
        _, keep = torch.topk(candidate_buffer["values"], k=int(limit), largest=False, sorted=False)
        return {key: value.index_select(0, keep) for key, value in candidate_buffer.items()}

    def _append_cpu_candidates(self, candidate_buffer, values, parents, moves, hashes, limit, prune_limit=None):
        incoming = {
            "values": values.to("cpu", dtype=torch.float16),
            "parents": parents.to("cpu", dtype=torch.int32),
            "moves": moves.to("cpu", dtype=torch.int8),
        }
        if hashes is not None:
            incoming["hashes"] = hashes.to("cpu", dtype=torch.int64)
        if candidate_buffer is None:
            candidate_buffer = incoming
        else:
            candidate_buffer = {
                key: torch.cat((candidate_buffer[key], incoming[key]))
                for key in incoming
            }
        if prune_limit is None:
            prune_limit = limit
        if candidate_buffer["values"].numel() > prune_limit:
            candidate_buffer = self._prune_cpu_candidates(candidate_buffer, limit)
        return candidate_buffer

    def _cpu_local_topk_size(self, B, states_count):
        if self.cpu_batch_topk is not None:
            return self.cpu_batch_topk
        batches = max(1, math.ceil(int(states_count) / self.batch_size))
        return max(1, math.ceil((int(B) / batches) * self.cpu_batch_topk_factor))

    def _select_cpu_candidates(self, states_cpu, last_moves_cpu, visited_hashed_cpu, B):
        candidate_buffer = None
        buffer_limit = max(int(B), int(int(B) * self.cpu_candidate_buffer_factor))
        prune_limit = max(buffer_limit, int(buffer_limit * self.cpu_candidate_prune_margin))
        local_topk_target = self._cpu_local_topk_size(B, states_cpu.size(0))
        total_valid = 0

        for start in range(0, states_cpu.size(0), self.batch_size):
            end = min(start + self.batch_size, states_cpu.size(0))
            batch_states_cpu = states_cpu[start:end]
            batch_states = batch_states_cpu.to(self.device, non_blocking=True)
            q_values = self.pred_q(batch_states)
            valid_mask = torch.ones((batch_states.size(0), self.n_gens), dtype=torch.bool, device=self.device)

            if last_moves_cpu is not None and self.inverse_moves is not None:
                batch_last = last_moves_cpu[start:end].to(self.device, non_blocking=True)
                valid_last = batch_last >= 0
                if valid_last.any():
                    rows = torch.nonzero(valid_last, as_tuple=False).view(-1)
                    inverse_idx = batch_last.index_select(0, rows).to(torch.int64)
                    valid_mask[rows, self.inverse_moves[inverse_idx]] = False

            solved_result = self._find_solved_child(batch_states, valid_mask, q_values)
            if solved_result is not None:
                _, _, solved_value, solved_move, solved_parent = solved_result
                solved_parent = solved_parent + start
                return {
                    "solved": True,
                    "states": self.V0_cpu.unsqueeze(0),
                    "hashes": self.V0_hash.to("cpu"),
                    "values": solved_value.to("cpu", dtype=torch.float16).view(1),
                    "moves": solved_move.to("cpu", dtype=torch.int8).view(1),
                    "parents": solved_parent.to("cpu", dtype=torch.int32).view(1),
                    "total_valid": total_valid + int(valid_mask.sum().item()),
                    "unique_count": 1,
                }

            q_flat = q_values.masked_fill(~valid_mask, float("inf")).reshape(-1)
            finite_mask = torch.isfinite(q_flat)
            local_total = int(finite_mask.sum().item())
            total_valid += local_total
            if local_total == 0:
                continue

            local_k = min(local_topk_target, q_flat.numel())
            local_values, local_flat = torch.topk(q_flat, k=local_k, largest=False, sorted=False)
            keep_finite = torch.isfinite(local_values)
            local_values = local_values[keep_finite]
            local_flat = local_flat[keep_finite]
            if local_flat.numel() == 0:
                continue

            local_parent = torch.div(local_flat, self.n_gens, rounding_mode="floor")
            local_moves = torch.remainder(local_flat, self.n_gens)
            need_hashes = self.cpu_dedup_candidates or (visited_hashed_cpu is not None and visited_hashed_cpu.numel() > 0)
            local_hashes = None
            if need_hashes:
                local_hashes = self._cpu_hash_after_move_from_device_batch(batch_states, local_parent, local_moves)

            if visited_hashed_cpu is not None and visited_hashed_cpu.numel() > 0:
                bad_mask = self._cpu_sorted_isin(local_hashes.to("cpu"), visited_hashed_cpu)
                if bad_mask.any():
                    keep = ~bad_mask.to(self.device)
                    local_values = local_values[keep]
                    local_parent = local_parent[keep]
                    local_moves = local_moves[keep]
                    local_hashes = local_hashes[keep]

            if local_values.numel() == 0:
                continue

            global_parent = local_parent + start
            candidate_buffer = self._append_cpu_candidates(
                candidate_buffer,
                local_values,
                global_parent,
                local_moves,
                local_hashes,
                buffer_limit,
                prune_limit,
            )

        if candidate_buffer is None or candidate_buffer["values"].numel() == 0:
            return {
                "solved": False,
                "states": torch.empty((0, self.state_size), dtype=self.state_dtype),
                "hashes": torch.empty((0,), dtype=torch.int64),
                "values": torch.empty((0,), dtype=torch.float16),
                "moves": torch.empty((0,), dtype=torch.int8),
                "parents": torch.empty((0,), dtype=torch.int32),
                "total_valid": total_valid,
                "unique_count": 0,
            }

        candidate_buffer = self._prune_cpu_candidates(candidate_buffer, int(B))
        parents = candidate_buffer["parents"]
        moves = candidate_buffer["moves"]
        next_states = torch.empty((parents.size(0), self.state_size), dtype=self.state_dtype)
        for start in range(0, parents.size(0), self.batch_size):
            end = start + self.batch_size
            parent_states = states_cpu.index_select(0, parents[start:end].to(torch.int64))
            move_idx = moves[start:end].to(torch.int64)
            next_states[start:end] = torch.gather(parent_states, 1, self.all_moves_cpu[move_idx])

        if "hashes" in candidate_buffer:
            result_hashes = candidate_buffer["hashes"]
        else:
            result_hashes = torch.empty((0,), dtype=torch.int64)

        return {
            "solved": False,
            "states": next_states,
            "hashes": result_hashes,
            "values": candidate_buffer["values"],
            "moves": moves,
            "parents": parents,
            "total_valid": total_valid,
            "unique_count": candidate_buffer["values"].numel(),
        }

    def _make_visited_reference(self, visited_hashed_cpu, recent_hash_rows):
        if self.cpu_visited_mode == "none":
            return None
        if self.cpu_visited_mode == "recent":
            if not recent_hash_rows:
                return None
            return torch.unique(torch.cat(list(recent_hash_rows))).sort().values
        if visited_hashed_cpu is None or visited_hashed_cpu.numel() == 0:
            return None
        return visited_hashed_cpu

    def get_solution(self, state, B=2**12, num_steps=200, num_attempts=10):
        state_cpu = state.to("cpu", dtype=self.state_dtype)
        if torch.equal(state_cpu, self.V0_cpu):
            return torch.empty((0,), dtype=torch.int64), 0

        states_bad_hashed_cpu = torch.empty((0,), dtype=torch.int64)
        for J in range(num_attempts):
            states_cpu = state_cpu.unsqueeze(0).clone()
            tree_move_rows = []
            tree_idx_rows = []
            recent_hash_rows = deque(maxlen=4)
            last_moves_cpu = -torch.ones((1,), dtype=torch.int8)
            root_hash = state2hash(states_cpu.to(self.device), self.hash_vec, self.batch_size).to("cpu")
            visited_hashed_cpu = torch.unique(torch.cat((states_bad_hashed_cpu, root_hash))).sort().values

            pbar = tqdm(range(num_steps), disable=not self.verbose)
            solved = False
            for j in pbar:
                visited_reference = self._make_visited_reference(visited_hashed_cpu, recent_hash_rows)
                result = self._select_cpu_candidates(states_cpu, last_moves_cpu, visited_reference, B)
                self.counter[0, 0] += result["total_valid"]
                self.counter[0, 1] += 1
                self.counter[1, 0] += result["unique_count"]
                self.counter[1, 1] += 1

                states_cpu = result["states"]
                states_hashed_cpu = result["hashes"]
                y_pred_cpu = result["values"]
                moves_cpu = result["moves"]
                parents_cpu = result["parents"]
                if states_cpu.size(0) == 0:
                    break

                tree_move_rows.append(moves_cpu.to(torch.int8).clone())
                tree_idx_rows.append(parents_cpu.to(torch.int32).clone())
                last_moves_cpu = moves_cpu.to(torch.int8)
                recent_hash_rows.append(states_hashed_cpu)
                if self.cpu_visited_mode == "all":
                    visited_hashed_cpu = torch.unique(torch.cat((visited_hashed_cpu, states_hashed_cpu))).sort().values

                self.counter[2, 0] += states_cpu.size(0)
                self.counter[2, 1] += 1
                if self.verbose and y_pred_cpu.numel() > 0:
                    pbar.set_description(
                        f"  y_min = {y_pred_cpu.min().item():.1f}, "
                        f"y_mean = {y_pred_cpu.float().mean().item():.1f}, "
                        f"y_max = {y_pred_cpu.max().item():.1f}"
                    )

                solved_mask = (states_cpu == self.V0_cpu).all(dim=1)
                if result["solved"] or solved_mask.any():
                    solved = True
                    break
            else:
                if self.cpu_visited_mode == "all":
                    states_bad_hashed_cpu = visited_hashed_cpu

            if solved:
                break
            if self.cpu_visited_mode == "all":
                states_bad_hashed_cpu = torch.unique(torch.cat((states_bad_hashed_cpu, visited_hashed_cpu))).sort().values

        solved_mask = (states_cpu == self.V0_cpu).all(dim=1)
        if not solved_mask.any():
            return None, J

        pos = int(torch.nonzero(solved_mask, as_tuple=True)[0][0].item())
        moves_rev = []
        for move_row, idx_row in zip(reversed(tree_move_rows), reversed(tree_idx_rows)):
            moves_rev.append(int(move_row[pos].item()))
            pos = int(idx_row[pos].item())
        moves_seq = torch.tensor(moves_rev, dtype=torch.int64)
        moves_seq = self.normalize_moves_seq(moves_seq)
        return moves_seq.flip((0,)), J
