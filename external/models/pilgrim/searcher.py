import torch
import time
from collections import deque
from tqdm import tqdm
from .utils import state2hash
from .model import batch_process


class Searcher:
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

    def __init__(
        self,
        model,
        all_moves,
        V0,
        device=None,
        verbose=0,
        move_names=None,
        inverse_moves=None,
        normalize_path=True,
        batch_size=None,
        hash_seed=0,
        state_dtype=None,
        tail_bfs_depth=0,
        move_order=4,
        goal_test=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.all_moves = all_moves
        self.state_dtype = V0.dtype if state_dtype is None else state_dtype
        self.V0 = V0.to(device=self.device, dtype=self.state_dtype)
        if batch_size is None:
            batch_size = 2**14
        self.batch_size = int(batch_size)
        self.n_gens = all_moves.size(0)
        self.state_size = all_moves.size(1)
        hash_generator = torch.Generator(device="cpu")
        hash_generator.manual_seed(int(hash_seed))
        self.hash_vec = torch.randint(0, int(1e15), (self.state_size,), generator=hash_generator, dtype=torch.int64).to(self.device)
        self.verbose = verbose
        self.counter = torch.zeros((3, 2), dtype=torch.int64)
        self.move_names = list(move_names) if move_names is not None else None
        self.inverse_moves = None if inverse_moves is None else torch.as_tensor(inverse_moves, dtype=torch.int64, device=self.device)
        self.normalize_path = bool(normalize_path)
        self.move_order = int(move_order)
        self.goal_test = goal_test
        if self.move_order < 2:
            raise ValueError("move_order must be >= 2")
        self.move_to_idx = None if self.move_names is None else {name: idx for idx, name in enumerate(self.move_names)}
        self.inverse_all_moves = torch.empty_like(self.all_moves)
        move_rows = torch.arange(self.n_gens, device=self.device).unsqueeze(1).expand(-1, self.state_size)
        state_cols = torch.arange(self.state_size, device=self.device).unsqueeze(0).expand(self.n_gens, -1)
        self.inverse_all_moves[move_rows, self.all_moves] = state_cols
        self.hash_after_move_weights = self.hash_vec.index_select(0, self.inverse_all_moves.reshape(-1)).view(self.n_gens, self.state_size)
        self.V0_hash = state2hash(self.V0.unsqueeze(0), self.hash_vec, 1)
        self.solved_predecessors = None
        self.solved_predecessor_hashes = None
        if self.inverse_moves is not None:
            solved_batch = self.V0.unsqueeze(0).expand(self.n_gens, -1)
            self.solved_predecessors = torch.gather(solved_batch, 1, self.all_moves[self.inverse_moves])
            self.solved_predecessor_hashes = state2hash(self.solved_predecessors, self.hash_vec, self.n_gens)
        self.tail_bfs_depth = int(tail_bfs_depth)
        self.tail_bfs_hashes = None
        self.tail_bfs_codes = None
        self.tail_bfs_base = self.n_gens + 1
        self.tail_bfs_cpu_device = torch.device("cpu")
        self.tail_bfs_all_moves = None
        self.tail_bfs_v0 = None
        if self.tail_bfs_depth < 0:
            raise ValueError("tail_bfs_depth must be >= 0")
        if self.tail_bfs_depth > 0:
            if self.inverse_moves is None:
                raise ValueError("tail_bfs_depth requires inverse_moves")
            self._build_tail_bfs(self.tail_bfs_depth)

    def goal_mask(self, states):
        if self.goal_test is None:
            return states.eq(self.V0.unsqueeze(0)).all(dim=1)
        result = self.goal_test(states)
        result = torch.as_tensor(result, dtype=torch.bool, device=states.device).flatten()
        if result.numel() != states.size(0):
            raise ValueError("goal_test must return one boolean per state")
        return result
    
    def get_unique_states(self, states, states_bad_hashed):
        """Filter unique states by removing duplicates based on hash."""
        idx1 = torch.arange(states.size(0), dtype=torch.int64, device=states.device)
        hashed = state2hash(states, self.hash_vec, self.batch_size)
        mask1  = ~self._sorted_isin(hashed, states_bad_hashed)
        if not mask1.any():
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=states.device)
            empty_idx = torch.empty((0,), dtype=torch.int64, device=states.device)
            return empty_states, empty_idx
        hashed = hashed[mask1]
        hashed_sorted, idx2 = torch.sort(hashed)
        mask2 = torch.concat((torch.tensor([True], device=states.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        return states[mask1][idx2[mask2]], idx1[mask1][idx2[mask2]] 
    
    def get_unique_hashed_states_idx(self, hashed, states_bad_hashed):
        """Filter unique hashed states by removing duplicates."""
        idx1 = torch.arange(hashed.size(0), dtype=torch.int64, device=hashed.device)
        mask1 = ~self._sorted_isin(hashed, states_bad_hashed)
        if not mask1.any():
            return torch.empty((0,), dtype=torch.int64, device=hashed.device)
        hashed = hashed[mask1]
        hashed_sorted, idx2 = torch.sort(hashed)
        mask2 = torch.concat((torch.tensor([True], device=hashed.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        return idx1[mask1][idx2[mask2]]
    
    def get_neighbors(self, states):
        """Return neighboring states for each state in the batch."""
        neighbors = torch.empty(states.size(0), self.n_gens, self.state_size, device=self.device, dtype=states.dtype)
        for i in range(0, states.size(0), self.batch_size):
            batch_states = states[i:i + self.batch_size]
            neighbors[i:i + self.batch_size] = torch.gather(
                batch_states.unsqueeze(1).expand(batch_states.size(0), self.n_gens, self.state_size), 
                2, 
                self.all_moves.unsqueeze(0).expand(batch_states.size(0), self.n_gens, self.state_size)
            )
        return neighbors
    
    def apply_move(self, states, moves):
        moved_states = torch.empty(states.size(0), self.state_size, device=self.device, dtype=states.dtype)
        for i in range(0, states.size(0), self.batch_size):
            moved_states[i:i+self.batch_size] = torch.gather(states[i:i+self.batch_size], 1, self.all_moves[moves[i:i+self.batch_size]])
        return moved_states

    def hash_after_move(self, states, moves):
        hashed = torch.empty(states.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, states.size(0), self.batch_size):
            batch_states = states[i:i+self.batch_size].to(torch.int64)
            batch_moves = moves[i:i+self.batch_size]
            batch_weights = self.hash_after_move_weights.index_select(0, batch_moves)
            hashed[i:i+self.batch_size] = torch.sum(batch_states * batch_weights, dim=1)
        return hashed

    def _build_tail_bfs(self, depth):
        """Build a hash table of states within depth moves from solved.

        Each stored code decodes to moves that solve the matching state.  The
        table is hash-based; the existing solved-state exact check still handles
        distance 0/1, and this optional tail table is meant as a pragmatic
        post-processing accelerator for shallow solved neighborhoods.
        """
        cpu = self.tail_bfs_cpu_device
        all_moves = self.all_moves.to(cpu)
        inverse_moves = self.inverse_moves.to(cpu)
        hash_vec = self.hash_vec.to(cpu)
        v0 = self.V0.to(cpu)
        self.tail_bfs_all_moves = all_moves
        self.tail_bfs_v0 = v0

        all_hash_rows = []
        all_code_rows = []
        visited = state2hash(v0.unsqueeze(0), hash_vec, self.batch_size)
        frontier = v0.unsqueeze(0)
        frontier_codes = torch.zeros((1,), dtype=torch.int64, device=cpu)
        frontier_last = -torch.ones((1,), dtype=torch.int64, device=cpu)

        move_template = torch.arange(self.n_gens, device=cpu, dtype=torch.int64)
        for _ in range(depth):
            next_states_rows = []
            next_hash_rows = []
            next_code_rows = []
            next_last_rows = []

            for start in range(0, frontier.size(0), self.batch_size):
                end = min(start + self.batch_size, frontier.size(0))
                batch_states = frontier[start:end]
                batch_codes = frontier_codes[start:end]
                batch_last = frontier_last[start:end]

                parent_idx = torch.arange(batch_states.size(0), device=cpu).repeat_interleave(self.n_gens)
                moves = move_template.repeat(batch_states.size(0))
                valid = (batch_last[parent_idx] < 0) | (moves != inverse_moves[batch_last[parent_idx]])
                parent_idx = parent_idx[valid]
                moves = moves[valid]
                if moves.numel() == 0:
                    continue

                child_states = torch.gather(batch_states.index_select(0, parent_idx), 1, all_moves[moves])
                child_hashes = state2hash(child_states, hash_vec, self.batch_size)
                keep = ~self._sorted_isin(child_hashes, visited)
                if not keep.any():
                    continue

                child_states = child_states[keep]
                child_hashes = child_hashes[keep]
                moves = moves[keep]
                parent_idx = parent_idx[keep]
                child_codes = (
                    inverse_moves[moves].to(torch.int64)
                    + 1
                    + self.tail_bfs_base * batch_codes.index_select(0, parent_idx)
                )

                order = torch.argsort(child_hashes)
                child_hashes = child_hashes.index_select(0, order)
                child_states = child_states.index_select(0, order)
                child_codes = child_codes.index_select(0, order)
                moves = moves.index_select(0, order)
                unique = torch.concat((
                    torch.tensor([True], device=cpu),
                    child_hashes[1:] != child_hashes[:-1],
                ))

                next_states_rows.append(child_states[unique])
                next_hash_rows.append(child_hashes[unique])
                next_code_rows.append(child_codes[unique])
                next_last_rows.append(moves[unique])

            if not next_hash_rows:
                break

            layer_states = torch.cat(next_states_rows)
            layer_hashes = torch.cat(next_hash_rows)
            layer_codes = torch.cat(next_code_rows)
            layer_last = torch.cat(next_last_rows)
            order = torch.argsort(layer_hashes)
            layer_states = layer_states.index_select(0, order)
            layer_hashes = layer_hashes.index_select(0, order)
            layer_codes = layer_codes.index_select(0, order)
            layer_last = layer_last.index_select(0, order)
            unique = torch.concat((
                torch.tensor([True], device=cpu),
                layer_hashes[1:] != layer_hashes[:-1],
            ))
            layer_states = layer_states[unique]
            layer_hashes = layer_hashes[unique]
            layer_codes = layer_codes[unique]
            layer_last = layer_last[unique]

            all_hash_rows.append(layer_hashes)
            all_code_rows.append(layer_codes)
            visited = torch.unique(torch.cat((visited, layer_hashes)))
            frontier = layer_states
            frontier_codes = layer_codes
            frontier_last = layer_last

        if all_hash_rows:
            hashes = torch.cat(all_hash_rows)
            codes = torch.cat(all_code_rows)
            order = torch.argsort(hashes)
            hashes = hashes.index_select(0, order)
            codes = codes.index_select(0, order)
            unique = torch.concat((
                torch.tensor([True], device=cpu),
                hashes[1:] != hashes[:-1],
            ))
            self.tail_bfs_hashes = hashes[unique]
            self.tail_bfs_codes = codes[unique]

    def _decode_tail_code(self, code):
        moves = []
        code = int(code)
        while code:
            digit = code % self.tail_bfs_base
            if digit:
                moves.append(digit - 1)
            code //= self.tail_bfs_base
        return torch.tensor(moves, dtype=torch.int64)

    def _tail_solves_state(self, state_cpu, tail_moves):
        current = state_cpu
        for move in tail_moves.tolist():
            current = current.index_select(0, self.tail_bfs_all_moves[move])
        return torch.equal(current, self.tail_bfs_v0)

    def _find_tail_bfs_hit(self, states, states_hashed):
        if self.tail_bfs_hashes is None or self.tail_bfs_hashes.numel() == 0:
            return None
        states_hashed = states_hashed.to(self.tail_bfs_cpu_device)
        pos = torch.searchsorted(self.tail_bfs_hashes, states_hashed)
        valid = pos < self.tail_bfs_hashes.numel()
        if not valid.any():
            return None
        check_pos = pos.clamp_max(self.tail_bfs_hashes.numel() - 1)
        matched = valid & (self.tail_bfs_hashes.index_select(0, check_pos) == states_hashed)
        if not matched.any():
            return None
        state_positions = torch.nonzero(matched, as_tuple=True)[0]
        matched_states = states.index_select(0, state_positions.to(states.device)).to(self.tail_bfs_cpu_device)
        for local_pos, state_pos_tensor in enumerate(state_positions.tolist()):
            table_pos = int(check_pos[state_pos_tensor].item())
            tail_moves = self._decode_tail_code(int(self.tail_bfs_codes[table_pos].item()))
            if self._tail_solves_state(matched_states[local_pos], tail_moves):
                return int(state_pos_tensor), tail_moves
        return None

    def _trace_beam_moves(self, tree_idx, tree_move, depth_idx, leaf_pos):
        tree_idx, tree_move = tree_idx[:depth_idx+1].flip((0,)), tree_move[:depth_idx+1].flip((0,))
        path = [tree_idx[0, leaf_pos].item()]
        for k in range(1, depth_idx+1):
            path.append(tree_idx[k, path[-1]].item())

        return torch.tensor(
            [
                int(tree_move[k, path[k - 1]].item()) if k > 0 else int(tree_move[k, leaf_pos].item())
                for k in range(depth_idx + 1)
            ],
            dtype=torch.int64,
        ).flip((0,))
    
    def do_greedy_step(self, states, states_bad_hashed, last_moves=None, B=1000):
        """Perform a greedy step to find the best neighbors."""
        idx0 = torch.arange(states.size(0), device=self.device).repeat_interleave(self.n_gens)
        moves = torch.arange(self.n_gens, device=self.device).repeat(states.size(0))

        if last_moves is not None and self.inverse_moves is not None:
            parent_last_moves = last_moves[idx0]
            valid_mask = (parent_last_moves < 0) | (moves != self.inverse_moves[parent_last_moves])
            idx0 = idx0[valid_mask]
            moves = moves[valid_mask]

        self.counter[0, 0] += moves.size(0); self.counter[0, 1] += 1;

        if moves.numel() == 0:
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_moves = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float16, device=self.device)
            return empty_states, empty_hashes, empty_values, empty_moves, empty_moves

        neighbors_hashed = torch.empty(moves.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, moves.size(0), self.batch_size):
            batch_idx = idx0[i:i+self.batch_size]
            batch_moves = moves[i:i+self.batch_size]
            neighbors = self.apply_move(states[batch_idx], batch_moves)
            neighbors_hashed[i:i+self.batch_size] = state2hash(neighbors, self.hash_vec, self.batch_size)
        idx1 = self.get_unique_hashed_states_idx(neighbors_hashed, states_bad_hashed)
        self.counter[1, 0] += idx1.size(0); self.counter[1, 1] += 1;

        if idx1.numel() == 0:
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_moves = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float16, device=self.device)
            return empty_states, empty_hashes, empty_values, empty_moves, empty_moves
        
        value = torch.empty(idx1.size(0), dtype=torch.float16, device=self.device)
        for i in range(0, idx1.size(0), self.batch_size):
            batch_states = self.apply_move(states[idx0[idx1[i:i+self.batch_size]]], moves[idx1[i:i+self.batch_size]])
            value[i:i+self.batch_size] = self.pred_d(batch_states)[0]
        idx2 = torch.topk(value, k=min(int(B), value.numel()), largest=False, sorted=False).indices
        self.counter[2, 0] += idx2.size(0); self.counter[2, 1] += 1;

        next_states = torch.empty(idx2.size(0), self.state_size, dtype=states.dtype, device=self.device)
        for i in range(0, idx2.size(0), self.batch_size):
            next_states[i:i+self.batch_size] = self.apply_move(
                states[idx0[idx1[idx2[i:i+self.batch_size]]]],
                moves[idx1[idx2[i:i+self.batch_size]]],
            )

        return (
            next_states,
            neighbors_hashed[idx1[idx2]],
            value[idx2],
            moves[idx1[idx2]],
            idx0[idx1[idx2]],
        )
    
    def check_stagnation(self, states_log):
        """Check if the process is in a stagnation state."""
        return torch.isin(torch.concat(list(states_log)[2:]), torch.concat(list(states_log)[:2])).all().item()

    def normalize_moves_seq(self, moves_seq):
        """Normalize consecutive same-face turns to a shorter equivalent sequence."""
        if not self.normalize_path or self.move_names is None or moves_seq.numel() == 0:
            return moves_seq

        normalized = []
        pending_face = None
        pending_turns = 0

        def split_move(move_name):
            if move_name.startswith("-"):
                return move_name[1:], -1
            if move_name.endswith("'"):
                return move_name[:-1], -1
            return move_name, 1

        def inverse_name(face):
            dash_name = f"-{face}"
            if dash_name in self.move_to_idx:
                return dash_name
            apostrophe_name = f"{face}'"
            if apostrophe_name in self.move_to_idx:
                return apostrophe_name
            raise KeyError(f"inverse move name not found for {face!r}")

        def flush(face, turns):
            if face is None:
                return
            turns %= self.move_order
            if turns == 0:
                return
            backward_turns = self.move_order - turns
            if turns <= backward_turns:
                normalized.extend([face] * turns)
            else:
                normalized.extend([inverse_name(face)] * backward_turns)

        for move_idx in moves_seq.tolist():
            move_name = self.move_names[move_idx]
            face, delta = split_move(move_name)

            if pending_face is not None and face != pending_face:
                flush(pending_face, pending_turns)
                pending_turns = 0

            pending_face = face
            pending_turns += delta

        flush(pending_face, pending_turns)

        if not normalized:
            return torch.empty((0,), dtype=torch.int64)
        return torch.tensor([self.move_to_idx[name] for name in normalized], dtype=torch.int64)

    
    def get_solution(self, state, B=2**12, num_steps=200, num_attempts=10):
        """Main solution-finding loop that attempts to solve the cube."""
        state = state.to(device=self.device, dtype=self.state_dtype)
        if bool(self.goal_mask(state.unsqueeze(0))[0]):
            empty_moves = torch.empty((0,), dtype=torch.int64)
            return empty_moves, 0

        states_bad_hashed = torch.tensor([], dtype=torch.int64, device=self.device)
        for J in range(num_attempts):
            states = state.unsqueeze(0).clone()
            tree_move_dtype = torch.int8 if self.n_gens <= torch.iinfo(torch.int8).max else torch.int16
            tree_move = -torch.ones((num_steps, B), dtype=tree_move_dtype)
            tree_idx = -torch.ones((num_steps, B), dtype=torch.int32)
            states_hash_log = deque(maxlen=4)
            last_moves = -torch.ones((1,), dtype=torch.int64, device=self.device)
            visited_hashed = torch.concat((states_bad_hashed, state2hash(states, self.hash_vec)))
            visited_hashed = torch.unique(visited_hashed)
            
            if self.verbose:
                pbar = tqdm(range(num_steps))
            else:
                pbar = range(num_steps)
            for j in pbar:
                states, states_hashed, y_pred, moves, idx = self.do_greedy_step(states, visited_hashed, last_moves, B)
                if states.size(0) == 0:
                    break
                if self.verbose:
                    pbar.set_description(
                        f"  y_min = {y_pred.min().item():.1f}, y_mean = {y_pred.mean().item():.1f}, y_max = {y_pred.max().item():.1f}"
                    )
                last_moves = moves
                states_hash_log.append(states_hashed)
                visited_hashed = torch.unique(torch.concat((visited_hashed, states_hashed)))
                leaves_num = states.size(0)
                tree_move[j, :leaves_num] = moves.to(device=tree_move.device, dtype=tree_move.dtype)
                tree_idx[j, :leaves_num] = idx.to(device=tree_idx.device, dtype=tree_idx.dtype)

                tail_hit = self._find_tail_bfs_hit(states, states_hashed)
                if tail_hit is not None:
                    tail_pos, tail_moves = tail_hit
                    prefix = self._trace_beam_moves(tree_idx, tree_move, j, tail_pos)
                    moves_seq = torch.cat((prefix, tail_moves))
                    moves_seq = self.normalize_moves_seq(moves_seq)
                    return moves_seq, J

                if self.goal_mask(states).any():
                    break
                elif (j > 3 and self.check_stagnation(states_hash_log)):
                    states_bad_hashed = torch.concat((states_bad_hashed, torch.concat(list(states_hash_log))))
                    states_bad_hashed = torch.unique(states_bad_hashed)
                    break
            else:
                states_bad_hashed = visited_hashed

            if self.goal_mask(states).any():
                break
            states_bad_hashed = torch.unique(torch.concat((states_bad_hashed, visited_hashed)))
        
        final_goal_mask = self.goal_mask(states)
        if not final_goal_mask.any():
            return None, J

        goal_pos = torch.nonzero(final_goal_mask, as_tuple=True)[0][0].item()
        moves_seq = self._trace_beam_moves(tree_idx, tree_move, j, goal_pos)
        moves_seq = self.normalize_moves_seq(moves_seq)
        return moves_seq, J
    
    def pred_d(self, states):
        """Predict values for states using the model."""
        pred = batch_process(self.model, states, self.device, self.batch_size)
#         pred[(states == self.V0).all(dim=-1)] = 0
        return pred.unsqueeze(0)
