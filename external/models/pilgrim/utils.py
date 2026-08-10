import torch


def load_torch_file(path, **kwargs):
    try:
        return torch.load(path, **kwargs)
    except Exception as exc:
        try:
            with open(path, "rb") as handle:
                prefix = handle.read(64)
        except OSError:
            raise
        if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
            raise RuntimeError(
                f"{path} is a Git LFS pointer, not the downloaded checkpoint. "
                "Install git-lfs and run `git lfs pull` in this repository."
            ) from exc
        raise


def parse_generator_spec(data):
    if "moves" in data and "move_names" in data:
        return data["moves"], data["move_names"]
    if "actions" in data and "names" in data:
        return data["actions"], data["names"]
    raise KeyError("generator spec must contain either moves/move_names or actions/names")


def generate_inverse_moves(moves):
    inverse_moves = [0] * len(moves)
    move_to_idx = {move: idx for idx, move in enumerate(moves)}
    for i, move in enumerate(moves):
        if move.startswith("-"):
            inverse_name = move[1:]
        elif f"-{move}" in move_to_idx:
            inverse_name = f"-{move}"
        elif move.endswith("'"):
            inverse_name = move[:-1]
        else:
            inverse_name = move + "'"
        inverse_moves[i] = move_to_idx[inverse_name]
    return inverse_moves


def generate_random_walk_states(V0, all_moves, inverse_moves, num_states, depth, device, seed=0):
    if num_states <= 0:
        raise ValueError("num_states must be > 0")
    if depth < 0:
        raise ValueError("depth must be >= 0")

    states = V0.repeat(num_states, 1)
    if depth == 0:
        return states

    inverse_moves_cpu = torch.as_tensor(inverse_moves, dtype=torch.int64, device="cpu")
    previous_moves = torch.full((num_states,), -1, dtype=torch.int64)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    for _ in range(depth):
        next_moves = torch.randint(all_moves.size(0), (num_states,), generator=rng, dtype=torch.int64)
        invalid = (previous_moves >= 0) & (next_moves == inverse_moves_cpu[previous_moves.clamp_min(0)])
        while invalid.any():
            next_moves[invalid] = torch.randint(
                all_moves.size(0),
                (int(invalid.sum().item()),),
                generator=rng,
                dtype=torch.int64,
            )
            invalid = (previous_moves >= 0) & (next_moves == inverse_moves_cpu[previous_moves.clamp_min(0)])

        next_moves_device = next_moves.to(device)
        states = torch.gather(states, 1, all_moves[next_moves_device])
        previous_moves = next_moves

    return states


def state2hash(states, hash_vec, batch_size=2**14):
    num_batches = (states.size(0) + batch_size - 1) // batch_size
    result = torch.empty(states.size(0), dtype=torch.int64, device=states.device)

    for i in range(num_batches):
        batch = states[i * batch_size : (i + 1) * batch_size].to(torch.int64)
        batch_hash = torch.sum(hash_vec * batch, dim=1)
        result[i * batch_size : (i + 1) * batch_size] = batch_hash
    return result
