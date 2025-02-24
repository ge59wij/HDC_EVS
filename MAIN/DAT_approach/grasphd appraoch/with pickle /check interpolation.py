import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

grid_width = 4
grid_height = 4
k = 3
dims = 100

max_x = ((grid_width + k - 1) // k) * k
max_y = ((grid_height + k - 1) // k) * k
corner_x_positions = list(range(0, max_x + 1, k))
corner_y_positions = list(range(0, max_y + 1, k))

def generate_unique_hv(base_value):
    """Creates a hypervector where each element is a unique integer for debugging."""
    return torch.arange(base_value, base_value + dims, dtype=torch.int32)

# ---- DEFINE CORNER HVs ----
corner_values = [
    [100, 120, 140],  # Top row (y=0)
    [200, 220, 240],  # Middle row (y=3)
    [300, 320, 340],  # Bottom row (y=6)
]

corner_hvs = {}
for i, x in enumerate(corner_x_positions):
    for j, y in enumerate(corner_y_positions):
        corner_hvs[(x, y)] = generate_unique_hv(corner_values[j][i])

print("\n[DEBUG] Corner Hypervectors:")
for (x, y), hv in corner_hvs.items():
    print(f"Corner ({x},{y}) : HV: {hv.tolist()}")

interpolated_hvs = {}

def get_interpolated_hv(x, y):
    """Retrieve or interpolate a hypervector for (x, y) with full debug output."""
    x_clamped = min(max(x, 0), grid_width)
    y_clamped = min(max(y, 0), grid_height)

    # Check if the point is a corner
    if (x_clamped, y_clamped) in corner_hvs:
        hv = corner_hvs[(x_clamped, y_clamped)]
        interpolated_hvs[(x, y)] = hv.clone()
        print(f"\n[DEBUG] Pixel ({x},{y}) is a corner, using HV: {hv.tolist()}")
        return hv.clone()

    # Check if on vertical edge (x is a corner)
    if x_clamped in corner_x_positions:
        j = max(0, np.searchsorted(corner_y_positions, y_clamped) - 1)
        j_next = min(j + 1, len(corner_y_positions) - 1)
        y0, y1 = corner_y_positions[j], corner_y_positions[j_next]
        dy = max(y1 - y0, 1e-9)
        lambda_y = (y_clamped - y0) / dy

        P0 = corner_hvs[(x_clamped, y0)]
        P1 = corner_hvs[(x_clamped, y1)]

        split_0 = int((1 - lambda_y) * dims)
        split_1 = dims - split_0

        print(
            f"\n[DEBUG] Interpolating Pixel ({x},{y}) on vertical edge between ({x_clamped},{y0}) and ({x_clamped},{y1})")
        print(f"  Weights: lambda_y={lambda_y:.2f}, splits: {split_0} from P0, {split_1} from P1")
        interpolated = torch.cat([P0[:split_0], P1[-split_1:]])

        interpolated_hvs[(x, y)] = interpolated
        return interpolated

    # Check if on horizontal edge (y is a corner)
    if y_clamped in corner_y_positions:
        i = max(0, np.searchsorted(corner_x_positions, x_clamped) - 1)
        i_next = min(i + 1, len(corner_x_positions) - 1)
        x0, x1 = corner_x_positions[i], corner_x_positions[i_next]
        dx = max(x1 - x0, 1e-9)
        lambda_x = (x_clamped - x0) / dx

        P0 = corner_hvs[(x0, y_clamped)]
        P1 = corner_hvs[(x1, y_clamped)]

        split_0 = int((1 - lambda_x) * dims)
        split_1 = dims - split_0

        print(
            f"\n[DEBUG] Interpolating Pixel ({x},{y}) on horizontal edge between ({x0},{y_clamped}) and ({x1},{y_clamped})")
        print(f"  Weights: lamba_x={lambda_x:.2f}, splits: {split_0} from P0, {split_1} from P1")
        interpolated = torch.cat([P0[:split_0], P1[-split_1:]])

        interpolated_hvs[(x, y)] = interpolated
        return interpolated

    # ---- FIX INSIDE WINDOW INTERPOLATION ----
    i = max(0, np.searchsorted(corner_x_positions, x_clamped) - 1)
    j = max(0, np.searchsorted(corner_y_positions, y_clamped) - 1)
    i_next = min(i + 1, len(corner_x_positions) - 1)
    j_next = min(j + 1, len(corner_y_positions) - 1)

    x0, x1 = corner_x_positions[i], corner_x_positions[i_next]
    y0, y1 = corner_y_positions[j], corner_y_positions[j_next]

    P00 = corner_hvs[(x0, y0)]
    P10 = corner_hvs[(x1, y0)]
    P01 = corner_hvs[(x0, y1)]
    P11 = corner_hvs[(x1, y1)]

    dx = max(x1 - x0, 1e-9)
    dy = max(y1 - y0, 1e-9)
    lambda_x = (x_clamped - x0) / dx
    lambda_y = (y_clamped - y0) / dy

    split_00 = int((1 - lambda_x) * (1 - lambda_y) * dims)
    split_10 = int(lambda_x * (1 - lambda_y) * dims)
    split_01 = int((1 - lambda_x) * lambda_y * dims)
    split_11 = dims - (split_00 + split_10 + split_01)

    interpolated = torch.cat([
        P00[:split_00], P10[:split_10], P01[:split_01], P11[:split_11]
    ])

    print(
        f"\n[DEBUG] Interpolating Pixel ({x},{y}) inside cell with corners: P00({x0},{y0}), P10({x1},{y0}), P01({x0},{y1}), P11({x1},{y1})")
    print(f"  Weights: lamba_x={lambda_x:.2f}, lambba_y={lambda_y:.2f}")
    print(f"  Chunks: {split_00} from P00, {split_10} from P10, {split_01} from P01, {split_11} from P11")
    print(f"  Interpolated HV: {interpolated.tolist()}")

    interpolated_hvs[(x, y)] = interpolated
    return interpolated

for x in range(grid_width + 1):
    for y in range(grid_height + 1):
        get_interpolated_hv(x, y)

