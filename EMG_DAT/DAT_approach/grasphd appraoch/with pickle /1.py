import torch
import torchhd
import random
from tqdm import tqdm
from pickle_encoder_for_grasp import load_pickle_dataset, plot_with_parameters, analyze_similarities
from grasphdencoding import *



class BaseEncoder:
    def __init__(self, height, width, dims, time_subwindow, k, device):
        self.height = height
        self.width = width
        self.dims = dims
        self.time_subwindow = time_subwindow
        self.k = k
        self.device = torch.device(device) if isinstance(device, str) else device

class Encoder1(BaseEncoder):
    def __init__(self, height, width, dims, time_subwindow, k, device):
        super().__init__(height, width, dims, time_subwindow, k, device)
        print("Initializing Encoder1...")
        self.H_I_plus = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_minus = -self.H_I_plus
        self.corner_hvs = self._generate_corner_hvs()
        self.precomputed_positions = self._precompute_position_hvs()
    def _generate_corner_hvs(self):
        num_rows = self.height // self.k + 1
        num_cols = self.width // self.k + 1
        return torchhd.random(num_rows * num_cols, self.dims, "MAP", device=self.device).reshape(num_rows, num_cols,
                                                                                                 self.dims)
    def _precompute_position_hvs(self):
        precomputed = {}
        for x in range(self.width):
            for y in range(self.height):
                i, j = x // self.k, y // self.k
                i_next, j_next = min(i + 1, self.corner_hvs.shape[0] - 1), min(j + 1, self.corner_hvs.shape[1] - 1)
                P00, P01, P10, P11 = self.corner_hvs[i, j], self.corner_hvs[i, j_next], self.corner_hvs[i_next, j], \
                self.corner_hvs[i_next, j_next]
                alpha_x, alpha_y = (x % self.k) / float(self.k - 1), (y % self.k) / float(self.k - 1)
                precomputed[(x, y)] = (
                        (1 - alpha_x) * (1 - alpha_y) * P00 +
                        alpha_x * (1 - alpha_y) * P10 +
                        (1 - alpha_x) * alpha_y * P01 +
                        alpha_x * alpha_y * P11
                )
        return precomputed

    def get_position_hv(self, x, y):
        return self.precomputed_positions[(x, y)]

    def encode_temporal(self, events, class_id):
        if not events:
            raise ValueError("No events provided for encoding.")

        print(f"Encoding {len(events)} events for class {class_id}...")

        E_temporal = torch.zeros(self.dims, device=self.device)

        for event in events:
            t, (x, y), polarity = event
            P_xy = self.get_position_hv(x, y)
            I_hv = self.H_I_plus if polarity == 1 else self.H_I_minus
            Ei = torchhd.bind(P_xy, I_hv)

            E_temporal = torchhd.bundle(E_temporal, Ei)

        return torchhd.normalize(E_temporal)
class Encoder1Seed(GraspHDseedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device):
        super().__init__(height, width, dims, time_subwindow, k=k, device=torch.device(device))

    def encode_temporal(self, events, class_id):
        if not events:
            raise ValueError("No events provided for encoding.")
        print(f"Encoding {len(events)} events for class {class_id}...")
        E_temporal = torch.zeros(self.dims, device=self.device)
        for event in events:
            t, (x, y), polarity = event
            P_xy = self.get_position_hv(x, y)
            I_hv = self.H_I_plus if polarity == 1 else self.H_I_minus
            T_ti = self.get_time_hv(t)
            Ei = torchhd.bind(torchhd.bind(P_xy, T_ti), I_hv)
            E_temporal = torchhd.bundle(E_temporal, Ei)
        return torchhd.normalize(E_temporal)


# ------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
split_name = "val"

max_samples = 20
DIMS = 6000
K = 4
Timewindow = 10000

print(f"\nUsing Device: {device}")
print(f"Configuration -> K: {K}, DIMS: {DIMS}, Samples: {max_samples}, Time Subwindow: {Timewindow}")

dataset = load_pickle_dataset(dataset_path, split=split_name, max_samples=max_samples)
random.shuffle(dataset)


encoder = Encoder1(height=480, width=640, dims=DIMS, time_subwindow=Timewindow, k =K, device=device)
encoded_vectors, class_labels = [], []

for sample_id, (events, class_id) in tqdm(enumerate(dataset), total=len(dataset), desc="Encoding Samples"):
    formatted_events = [(float(event["t"]), (int(event["x"]), int(event["y"])), int(event["p"])) for event in events]
    encoded_sample = encoder.encode_temporal(formatted_events, class_id)
    encoded_vectors.append(encoded_sample.to(device).squeeze())
    class_labels.append(class_id)

print("\nEncoding Complete. Generating similarity heatmap...\n")
encoded_matrix = torch.stack(encoded_vectors)
similarity_matrix = torchhd.cosine_similarity(encoded_matrix, encoded_matrix)
plot_with_parameters(similarity_matrix, class_labels, K, Timewindow, DIMS, max_samples)
analyze_similarities(similarity_matrix, class_labels, Timewindow, K, DIMS, max_samples)
