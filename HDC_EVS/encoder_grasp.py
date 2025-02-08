import torch
import torchhd
import math

def generate_time_hvs_correlated(max_time, dims, chunk_size=20):
    # Number of chunks
    n_chunks = math.ceil(max_time / chunk_size) + 1
    # Random HVs for each chunk boundary
    boundary_hvs = [torchhd.random(1, dims, "MAP")[0] for _ in range(n_chunks)]
    # boundary_hvs[i] is the HV for time i*chunk_size
    time_hvs = []
    for t in range(max_time):
        # Figure out which chunk boundaries we are between
        lower_i = t // chunk_size  # Integer division
        upper_i = lower_i + 1
        if upper_i >= n_chunks:
            upper_i = n_chunks - 1

        lower_boundary = lower_i * chunk_size
        upper_boundary = upper_i * chunk_size

        if lower_boundary == upper_boundary:
            # Edge case if chunk_size=1 or last chunk
            time_hvs.append(boundary_hvs[lower_i])
            continue

        alpha = (upper_boundary - t) / (upper_boundary - lower_boundary)
        # alpha in [0..1], how close t is to the lower boundary vs the upper boundary

        hv_lower = boundary_hvs[lower_i]
        hv_upper = boundary_hvs[upper_i]
        hv_t = alpha * hv_lower + (1 - alpha) * hv_upper

        # Optionally, re-binarize or re-sign to keep it in {±1}
        hv_t = torch.sign(hv_t)

        time_hvs.append(hv_t)

    return time_hvs  # List of length max_time


def generate_position_hvs(height, width, dims):
    """
    Generate a random hypervector for each pixel position (x,y).
    For simplicity, each (x,y) gets a unique random HV.
    """
    pos_hvs = torchhd.random(height * width, dims, "MAP")
    return pos_hvs  # Shape [height * width, dims]


def generate_polarity_hvs(dims):
    """
    2 polarities: ON, OFF. We'll just do random HV for ON, and use its negative for OFF.
    """
    hv_on = torchhd.random(1, dims, "MAP")[0]
    hv_off = -hv_on  # Or torchhd.random_hv(1, dims, bipolar=True)[0]
    return hv_on, hv_off


class GraspHDEventEncoder:
    """
    storeall seeds and can encode a single [T,2,H,W] sample
    into a single spatio-temporal HV.
    """
    def __init__(self, height, width, max_time, dims, chunk_size, device):
        self.height = height
        self.width = width
        self.dims = dims
        self.max_time = max_time
        self.device = torch.device(device)

        # 1) Polarity HVs
        self.hv_on, self.hv_off = generate_polarity_hvs(dims)
        # 2) Position HVs
        self.pos_hvs = generate_position_hvs(height, width, dims).to(self.device)
        # 3) Time HVs (correlated)
        self.time_hvs = torch.stack(
            generate_time_hvs_correlated(max_time, dims, chunk_size)
        ).to(self.device)

    def encode_sample(self, sample_tensor):
        """
        Encodes a single sample into a hypervector.
        sample_tensor shape: [T, 2, height, width]
          sample_tensor[t, 0, x, y] => #events for (on)
          sample_tensor[t, 1, x, y] => #events for (off)
        Assumes T <= self.max_time. If T < max_time, that's okay. If T > max_time,
        either clamp or generate bigger time HVs.
        """
        T = sample_tensor.shape[0]
        final_hv = torch.zeros(self.dims, device=self.device)

        for t in range(T):
            # Build "spatial HV" for this time slice
            slice_on = sample_tensor[t, 0].to(self.device)  # Shape [H, W]
            slice_off = sample_tensor[t, 1].to(self.device)  # Shape [H, W]

            # Reshape position HVs to [H, W, dims]
            pos_hvs_reshaped = self.pos_hvs.view(self.height, self.width, self.dims)

            # Compute HV contributions from ON and OFF events
            hv_on = (slice_on.unsqueeze(-1) * pos_hvs_reshaped) * self.hv_on.to(self.device)
            hv_off = (slice_off.unsqueeze(-1) * pos_hvs_reshaped) * self.hv_off.to(self.device)

            # Sum over spatial dimensions
            spatial_hv_t = hv_on.sum(dim=(0, 1)) + hv_off.sum(dim=(0, 1))

            # Bind with time HV
            hv_time = self.time_hvs[t]
            hv_t = spatial_hv_t * hv_time  # Elementwise multiply => binding

            # Accumulate temporal contributions
            final_hv += hv_t

        # (Optional) binarize final HV
        final_hv = torch.sign(final_hv)
        return final_hv



'''
if __name__ == "__main__":
    # Quick test
    encoder = GraspHDEventEncoder(
        height=120, width=160, max_time=400, dims=10000, chunk_size=20, device="cuda"
    )

    # Suppose we have T=180, shape [180, 2, 120, 160]
    fake_sample = torch.randint(low=0, high=2, size=(180, 2, 120, 160)).float()

    hv = encoder.encode_sample(fake_sample)
    print("Encoded HV shape:", hv.shape)  # [10000]
    print(hv)
    print("Number of +1 entries:", torch.sum(hv == 1).item())
    print("Number of -1 entries:", torch.sum(hv == -1).item())
'''