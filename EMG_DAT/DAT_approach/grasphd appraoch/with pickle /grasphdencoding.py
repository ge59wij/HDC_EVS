import torch
import torchhd
from grasphdencoding_seedhvs import GraspHDseedEncoder
#todo: normalization

class GraspHDEventEncoder(GraspHDseedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device):
        device = torch.device(device) if isinstance(device, str) else device
        print(f"Initializing encoder")
        super().__init__(height, width, dims, time_subwindow, k=k, device=device)

    def encode_temporal(self, events, class_id):
        """Binds spatial, polarity, and timestamp hypervectors for each event."""
        if not events:
            raise ValueError("No events provided for encoding.")
        print(f"Temporal Encoding: Processing {len(events)} events on {self.device}...")

        last_timestamp = events[-1][0]
        self._generate_time_hvs(last_timestamp)

        E_temporal = None

        for event_index, event in enumerate(events):
            t, (x, y), polarity = event
            P_xy = self.get_position_hv(x, y)
            I_hv = self.H_I_plus if polarity == 1 else self.H_I_minus
            T_ti = self.get_time_hv(t)

            Ei = torchhd.bind(torchhd.bind(P_xy,T_ti), I_hv)  #multibundle?
            Ei = torchhd.ensure_vsa_tensor(Ei, "MAP")

            if E_temporal is None:
                E_temporal = Ei  # First event initializes E_temporal
            else:
                E_temporal = torchhd.bundle(E_temporal, Ei)  # Incremental bundling


            #Debuggingevery 100,000 events
            if (event_index + 1) % 100000 == 0:
                print(f"Intermediate Check after {event_index + 1} events: E_temp: {E_temporal}")

        print(f"\nTemporal Encoding Complete | Class ID: {class_id} | Output Shape: {E_temporal.shape} | Device: {E_temporal.device}")
        print("Sample HV: ", E_temporal)
        return E_temporal



     #Spatial encoding??? this, summed with E temp, last step, resulting vector is HV sample.
def encode_spatial(self, events):

        ###Spatial Encoding:
        #For each triggered event (x, y, polarity), compute H_spatial as:
        #H_spatial = sum_x sum_y (P(x,y) bind I)
        #- (binding) dissimilar
        #- (bundle/Summing) preserves similarities for the accumulated event representations.

    print(f'spatial encoding processing  {len(events)} events...') #debug
    spatial_encoded_event = []
    for event in events:
    t, (x, y), polarity = event
    P_xy = self.get_position_hv(x, y)  # Get position HV
    I_hv = self.H_I_plus if polarity == 1 else self.H_I_minus  # Get polarity HV
    spatial_encoded_event = torchhd.bind(P_xy, I_hv)
    spatial_encoded_event.append(spatial_encoded_event)
    H_spatial = torchhd.multibundle(torch.stack(spatial_encoded_event))
    print(f'Spatial Encoding Complete | Output Shape: {H_spatial.shape}')
    return H_spatial  # Final spatial encoding representing all events, one sample.
