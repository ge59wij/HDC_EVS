import torch
import torchhd
import numpy as np

grid_width = 5
grid_height = 4
k = 2
dims = 40

max_x = ((grid_width + k - 1) // k) * k  # Next multiple of k
max_y = ((grid_height + k - 1) // k) * k

corner_x_positions = list(range(0, max_x + 1, k))
corner_y_positions = list(range(0, max_y + 1, k))

corner_hvs = {}

for x in corner_x_positions:
    for y in corner_y_positions:
        corner_hvs[(x, y)] = torchhd.embeddings.Random(1, dims, "MAP").weight[0]

print("\n[DEBUG] Corner Hypervector Grid (Including Virtual):")
for (x, y), hv in corner_hvs.items():
    print(f"Corner ({x},{y}) → HV Index ({x},{y})")

def get_interpolated_hv(x, y):
    """Retrieve or interpolate a hypervector for (x, y) with boundary handling."""
    print(f"\n[DEBUG] Interpolating Pixel ({x},{y})")

    # Clamp coordinates to grid bounds
    x_clamped = min(max(x, 0), grid_width)
    y_clamped = min(max(y, 0), grid_height)

    # Find surrounding corners
    i = max(0, np.searchsorted(corner_x_positions, x_clamped) - 1)
    j = max(0, np.searchsorted(corner_y_positions, y_clamped) - 1)

    i_next = min(i + 1, len(corner_x_positions) - 1)
    j_next = min(j + 1, len(corner_y_positions) - 1)

    # Get corner coordinates
    x0, x1 = corner_x_positions[i], corner_x_positions[i_next]
    y0, y1 = corner_y_positions[j], corner_y_positions[j_next]

    # Retrieve HVs
    P00 = corner_hvs[(x0, y0)]
    P10 = corner_hvs[(x1, y0)]
    P01 = corner_hvs[(x0, y1)]
    P11 = corner_hvs[(x1, y1)]

    print(f"  Corners: P00({x0},{y0}), P10({x1},{y0}), P01({x0},{y1}), P11({x1},{y1})")

    # interpolation weights
    dx = max(x1 - x0, 1e-9)
    dy = max(y1 - y0, 1e-9)
    lambda_x = (x_clamped - x0) / dx
    lambda_y = (y_clamped - y0) / dy

    print(f"  Weights: λ_x={lambda_x:.2f}, λ_y={lambda_y:.2f}")

    # Compute segment sizes (global indices)
    split_00 = round((1 - lambda_x) * (1 - lambda_y) * dims)
    split_10 = round(lambda_x * (1 - lambda_y) * dims)
    split_01 = round((1 - lambda_x) * lambda_y * dims)
    split_11 = dims - (split_00 + split_10 + split_01)

    # Global indices for each segment (same across all interpolations!!)
    idx_00 = slice(0, split_00)
    idx_10 = slice(split_00, split_00 + split_10)
    idx_01 = slice(split_00 + split_10, split_00 + split_10 + split_01)
    idx_11 = slice(split_00 + split_10 + split_01, dims)

    print(f"  Global Indices:")
    print(f"    P00: {idx_00.start}-{idx_00.stop}")
    print(f"    P10: {idx_10.start}-{idx_10.stop}")
    print(f"    P01: {idx_01.start}-{idx_01.stop}")
    print(f"    P11: {idx_11.start}-{idx_11.stop}")

    # Concatenate using global indices (same for all HVs)
    position_hv = torch.cat([
        P00[idx_00],
        P10[idx_10],
        P01[idx_01],
        P11[idx_11]
    ])

    assert position_hv.shape[0] == dims, "Size mismatch!"
    print(f"  ✅ Success: HV size = {position_hv.shape[0]}")
    return position_hv

test_points = [
    (0, 0),  # Exact corner (uses P00 only)
    (1, 2),  # Inside window (blend of 4 corners)
    (3, 3),  # Actual grid corner (virtual corner)
    (10, 4)  # Virtual corner (x=10 > grid_width=10)
]

print("\n[DEBUG] Testing Interpolation")
for (x, y) in test_points:
    print(f"\n--- Testing ({x},{y}) ---")
    hv = get_interpolated_hv(x, y)