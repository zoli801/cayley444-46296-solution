# cube444 Q-beam TPU assets

Weights and puzzle data for a Q-native JAX/SPMD beam on the CayleyPy 4x4x4 **colour** cube.

| file | what |
|---|---|
| `s3.npz` | PieceTransformer Q head, 3,383,064 params. 57 tokens (8 corners + 24 wings + 24 centres + CLS), d_model 256, 4 layers, 8 heads, ff 1024, ReLU, CLS pooling. Q-Bellman refined. |
| `mlp_x16.npz` | PairQMLP Q head, 47M params, hd1 1536 / hd2 2304 / 4 residual blocks. BatchNorm is pre-folded into per-channel affine (valid in eval mode only). |
| `p002.json` | the 24 quarter-turn generators and their names |
| `solved_state.npy` | the 96-facelet solved colouring |
| `test.csv` | the 1043 competition states |
| `cube4_submission_46662.csv` | the community floor, used as the merge base and the scoring reference |

Both heads emit **24 values per state**, one per action, with `Q(s,a) = d(apply(s,a))`.
Two contracts follow:

* **Q-native** — score `B` parents, take a GLOBAL flat top-`alpha*B` over all
  `(parent, action)` pairs. Never per-parent.
* **V-like** — `min_a Q(s,a) = d(s) - 1`, a drop-in scalar scorer. Correct, but 24x the
  forwards, which is fatal for a 57-token transformer at width.

The deployed blend is `Q = 0.6 * s3 + 0.4 * mlp_x16`.

## Gotchas that cost real time

1. **`num_classes` is 6, not `state_size`.** Every torch constructor in the original bundle
   defaults it to `state_size`; that silently builds a 96-way embedding and a different,
   worse model. It does not error.
2. **This is a colour cube.** A state is a colouring, so there is no `invert_state` — no
   NISS, no inverse frames, no bidirectional search, and 24 rotation frames rather than 48.
   Symmetry needs a *recolour*: `sym(s,R) = color_map_R[s[rotation_R]]`. The
   slot-permutation-only form yields a legal-looking colouring and is silently wrong.
3. **The npz weights are pre-transposed** to `(in, out)` so the JAX side is a plain
   `h @ w`, and `nn.MultiheadAttention`'s packed `in_proj_weight` is already split into
   q/k/v. Do not transpose again.
4. **BatchNorm is folded.** `mlp_x16.npz` stores `(scale, shift)` per BN, computed from the
   running statistics. Exact in eval mode, meaningless for training.
