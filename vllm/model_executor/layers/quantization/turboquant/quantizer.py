# SPDX-License-Identifier: Apache-2.0
"""TurboQuant quantizer utilities.

Provides WHT (Walsh-Hadamard Transform) rotation primitives used by the
vLLM serving path. The Hadamard matrix is shared across all layers (cached);
each layer gets a unique random sign-flip vector for decorrelation.

Triton kernels (triton_turboquant_store.py / triton_turboquant_decode.py)
handle all quantization, packing, and dequantization on GPU.
"""

from functools import lru_cache
import torch

# ── WHT rotation (default) ───────────────────────────────────────────

@lru_cache(maxsize=8)
def generate_hadamard_matrix(d: int) -> torch.Tensor:
    """Generate normalized D×D Hadamard matrix (cached, shared across layers).

    Uses Sylvester construction: H(2d) = [[H(d), H(d)], [H(d), -H(d)]].
    Requires d to be a power of 2. Result is normalized by 1/sqrt(d).
    """
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / (d ** 0.5)


def generate_random_signs(
    d: int,
    seed: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate deterministic random ±1 sign-flip vector for WHT rotation.

    Each layer uses a unique seed for decorrelation.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, (d,), generator=gen, dtype=torch.float32,
                         device="cpu")
    if device is not None:
        return (bits * 2 - 1).to(device)
    return bits * 2 - 1


# ── Legacy QR rotation (kept for backward compatibility) ─────────────

def generate_rotation_matrix(
    d: int, seed: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate Haar-distributed random orthogonal matrix via QR decomposition.

    Deprecated: WHT rotation (generate_hadamard_matrix + generate_random_signs)
    is preferred — better Gaussianization, less memory, same GPU performance.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity for determinism
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)
