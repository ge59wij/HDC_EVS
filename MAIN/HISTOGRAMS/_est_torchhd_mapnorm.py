import torch
import torchhd

# Set dimensions
DIMS = 10000  # Large enough to see the impact

# Generate random bipolar hypervectors
hv1 = torchhd.random(1, DIMS, "MAP").squeeze(0)
hv2 = torchhd.random(1, DIMS, "MAP").squeeze(0)
hv3 = torchhd.random(1, DIMS, "MAP").squeeze(0)
hv4 = torchhd.random(1, DIMS, "MAP").squeeze(0)
hv5 = torchhd.random(1, DIMS, "MAP").squeeze(0)

# Stack them together
hv_stack = torch.stack([hv1, hv2, hv3, hv4, hv5])

print("\nOriginal Hypervectors:")
print(f"hv1 (first 10 elements): {hv1[:10]}")
print(f"hv2 (first 10 elements): {hv2[:10]}")

# Binding operation (pairwise)
hv_bind = torchhd.bind(hv1, hv2)
print("\nBinding Result:")
print(f"hv_bind (first 10 elements): {hv_bind[:10]}")

# Bundling operation (two vectors)
hv_bundle = torchhd.bundle(hv1, hv2)
print("\nBundling Result (Two Vectors):")
print(f"hv_bundle (first 10 elements): {hv_bundle[:10]}")

# Bundling multiple hypervectors using multiset
hv_multi_bundle = torchhd.multiset(hv_stack)

print("\nBundling Multiple HVs Using `multiset()`:")
print(f"hv_multi_bundle (first 10 elements): {hv_multi_bundle[:10]}")

# Normalization
hv_bind_norm = torchhd.normalize(hv_bind)
hv_bundle_norm = torchhd.normalize(hv_bundle)
hv_multi_bundle_norm = torchhd.normalize(hv_multi_bundle)

print("\nAfter Normalization:")
print(f"hv_bind_norm (first 10 elements): {hv_bind_norm[:10]}")
print(f"hv_bundle_norm (first 10 elements): {hv_bundle_norm[:10]}")
print(f"hv_multi_bundle_norm (first 10 elements): {hv_multi_bundle_norm[:10]}")

# Magnitude Check
print("\nVector Magnitude Before and After Normalization:")
print(f"hv_bind norm: {torch.norm(hv_bind).item():.4f} -> {torch.norm(hv_bind_norm).item():.4f}")
print(f"hv_bundle norm: {torch.norm(hv_bundle).item():.4f} -> {torch.norm(hv_bundle_norm).item():.4f}")
print(f"hv_multi_bundle norm: {torch.norm(hv_multi_bundle).item():.4f} -> {torch.norm(hv_multi_bundle_norm).item():.4f}")


dims = 5000

# Generate random HVs
hv1 = torchhd.random(1, dims, "MAP").squeeze(0)
hv2 = torchhd.random(1, dims, "MAP").squeeze(0)

# Create "neutral" HVs
identity_hv = torchhd.identity(1, dims).squeeze(0)  # Identity vector
zero_hv = torch.zeros(dims)  # Zero vector

# Test Binding
bind_with_identity = torchhd.bind(hv1, identity_hv)
bind_with_zero = torchhd.bind(hv1, zero_hv)

# Test Bundling
bundle_with_identity = torchhd.bundle(hv1, identity_hv)
bundle_with_zero = torchhd.bundle(hv1, zero_hv)

# Display results
print("\n[TEST] Does Identity HV Affect Operations?")
print("Bind with identity (should be same as hv1):", torch.allclose(bind_with_identity, hv1, atol=1e-5))
print("Bind with zero HV (should be different):", torch.allclose(bind_with_zero, hv1, atol=1e-5))
print("Bundle with identity (should be same as hv1):", torch.allclose(bundle_with_identity, hv1, atol=1e-5))
print("Bundle with zero HV (should change hv1):", torch.allclose(bundle_with_zero, hv1, atol=1e-5))
