import torch
import torchhd
import numpy as np
import math

grid_width = 4
grid_height = 4
k = 6
dims = 10000

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

    # Calculate interpolation weights with clamping
    dx = max(x1 - x0, 1e-9)
    dy = max(y1 - y0, 1e-9)
    lambda_x = (x_clamped - x0) / dx
    lambda_y = (y_clamped - y0) / dy
    lambda_x = min(max(lambda_x, 0.0), 1.0)
    lambda_y = min(max(lambda_y, 0.0), 1.0)

    print(f"  Weights: λ_x={lambda_x:.2f}, λ_y={lambda_y:.2f}")

    # Compute segment sizes
    split_00 = int((1 - lambda_x) * (1 - lambda_y) * dims)
    split_10 = int(lambda_x * (1 - lambda_y) * dims)
    split_01 = int((1 - lambda_x) * lambda_y * dims)
    split_11 = dims - (split_00 + split_10 + split_01)

    # Ensure non-negative splits
    split_00 = max(split_00, 0)
    split_10 = max(split_10, 0)
    split_01 = max(split_01, 0)
    split_11 = max(split_11, 0)

    # Concatenate leading dimensions from each corner
    position_hv = torch.cat([
        P00[:split_00],
        P10[:split_10],
        P01[:split_01],
        P11[:split_11]
    ])
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

# ---- TEST INTERPOLATION ON SAMPLE PIXELS ----
test_points = [
    (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6),
    (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
    (2,0), (2,1), (2,2), (2,3), (2,4), (2,5), (2,6),
    (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6),
    (4,0), (4,1), (4,2), (4,3), (4,4), (4,5), (4,6),
    (5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6),
    (6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6) ]

print("\n[DEBUG] Checking Interpolated Hypervectors")
interpolated_hvs = {}

for (x, y) in test_points:
    print(f"\n--- Testing ({x},{y}) ---")
    interpolated_hvs[(x, y)] = get_interpolated_hv(x, y)

# ---- COSINE SIMILARITY CHECKS ----
print("\n[DEBUG] Cosine Similarity Comparisons")

def cosine_similarity(hv1, hv2):
    return torch.nn.functional.cosine_similarity(hv1.unsqueeze(0), hv2.unsqueeze(0)).item()

print("\n[SIMILARITY] Inside the same window:")
sim_1 = cosine_similarity(interpolated_hvs[(0, 0)], interpolated_hvs[(0, 1)])
print(f"  Pixel (0,0) vs (0,1): {sim_1:.4f}")
sim_2 = cosine_similarity(interpolated_hvs[(0, 0)], interpolated_hvs[(0, 2)])
print(f"  Pixel (0,0) vs (0,2): {sim_2:.4f}")
sim_3 = cosine_similarity(interpolated_hvs[(0, 0)], interpolated_hvs[(1, 1)])
print(f"  Pixel (0,0) vs (1,1): {sim_2:.4f}")

# eighboring windows
print("\n[SIMILARITY] Neighboring Windows:")
sim_3 = cosine_similarity(interpolated_hvs[(0, 2)], interpolated_hvs[(0,4)])
print(f"  Pixel (0,2) vs (0,4): {sim_3:.4f}")

#diagonal neighboring windows
print("\n[SIMILARITY] Diagonal Windows:")
sim_4 = cosine_similarity(interpolated_hvs[(2, 2)], interpolated_hvs[(4, 4)])
print(f"  Pixel (2,2) vs (4,4): {sim_4:.4f}")

#near boundary
print("\n[SIMILARITY] Boundary Handling:")
sim_5 = cosine_similarity(interpolated_hvs[(2,3)], interpolated_hvs[(2,2)])
print(f"  Pixel (2,3) vs (2,2): {sim_5:.4f}")
sim_5 = cosine_similarity(interpolated_hvs[(2,4)], interpolated_hvs[(2,3)])
print(f"  Pixel (2,4) vs (2,4): {sim_5:.4f}")
