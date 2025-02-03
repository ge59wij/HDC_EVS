import torch
import torch.nn.functional as F
import torchhd

#grasphd
class GestureHDEventEncoder:
    def __init__(self, height, width, max_time, dims, device):
        self.height = height
        self.width = width
        self.max_time = max_time
        self.dims = dims
        self.device = device
        # Seed hypervectors for pixel positions: shape [height, width, dims]
        self.P = torch.randint(0, 2, (height, width, dims), device=self.device, dtype=torch.int8).float()
        self.P[self.P == 0] = -1  # bipolar
        # Seed hypervectors for polarity: shape [2, dims]
        self.I = torch.randint(0, 2, (2, dims), device=self.device, dtype=torch.int8).float()
        self.I[self.I == 0] = -1
        self.T0 = torch.randint(0, 2, (dims,), device=self.device, dtype=torch.int8).float()
        self.T0[self.T0 == 0] = -1
    def _permute_time(self, t):
        # Simple permutation: circular shift by t positions
        return torch.roll(self.T0, shifts=t, dims=0)



    def encode_sample(self, sample):
        sample = sample.to(self.device)
        T = sample.shape[0]
        accumulator = torch.zeros(self.dims, device=self.device)
        for t in range(T):
            # Get time-specific hypervector
            T_t = self._permute_time(t)
            # For each polarity and pixel, bind pixel and polarity HVs, weighted by event count.
            # sample[t] shape: [2, height, width]
            spatial_sum = torch.zeros(self.dims, device=self.device)
            for i in range(2):
                # Get event counts for polarity i: shape [height, width]
                counts = sample[t, i].to(self.device)  # Ensure counts is on the same device
                # Broadcast counts to shape [height, width, dims] and multiply with P and I[i]
                # Binding: elementwise multiplication
                bound = counts.unsqueeze(-1) * (self.P * self.I[i])
                # Sum over spatial dimensions
                spatial_sum += bound.sum(dim=(0, 1))
            # Bind the spatial sum with the time hypervector (elementwise multiplication)
            E_t = spatial_sum * T_t
            accumulator += E_t
        # Binarize
        encoded = torch.sign(accumulator)
        encoded[encoded == 0] = 1
        return encoded
