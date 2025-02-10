import torch
import torchhd
import torchhd.functional as functional

from grasphdencoding_seedhvs import GraspHDseedEncoder
#todo: load from pickle files. batches like emg. done
#todo: search if normalizing all at each step makes sense.

class GraspHDEventEncoder(GraspHDseedEncoder):
    def __init__(self, height, width, dims, k, device):
        """
        Initialize the encoder for event-based data.

        Args:
            height (int): Sensor height (number of rows).
            width (int): Sensor width (number of columns).
            k (int): Grid size for spatial hypervectors.
            device.
        """
        #device = device #if torch.cuda.is_available() else "cpu"  # Default to CPU if no device is specified
        #self.device = torch.device(device)
        device = torch.device(device) if isinstance(device, str) else device
        # Ensure the parent class is initialized with the correct device
        print(f"Initializing encoder")
        super().__init__(height, width, dims, time_subwindow=5000, k=k, device=device) ###############50000 μs  Fewer timestamp seed hypervectors
        self.time_hvs = None

    def encode_temporal(self, events, class_id):
        """
        Temporal encoding binds spatial, polarity, and timestamp hypervectors for each event
        and bundles them across time to create a composite hypervector.
        Args:
            events (list): A list of events, where each event is a tuple (t, (x, y), polarity).
        Returns:
            torch.Tensor
        """
        if not events:
            raise ValueError("No events provided for encoding.")
        print(f"Temporal Encoding: Processing {len(events)} events on {self.device}...")

        last_timestamp = events[-1][0]  # Last event timestamp
        time_hvs = self._generate_time_hvs(last_timestamp)  # Generate dynamic timestamp HVs ##### improvement: cache them, only generate if not already, aka longer samples.

        #E_list = []

        E_temporal = torch.zeros(self.dims, device=self.device)  # Initialize HV accumulator instead of list for memory ## this good?

        for event_index, event in enumerate(events):
            t, (x, y), polarity = event  # Unpack event data

            # Get spatial, polarity, and timestamp hypervectors
            P_xy = self.get_position_hv(x, y)  # Position HV
            I_hv = self.H_I_plus if polarity == 1 else self.H_I_minus  # Polarity HV
            T_ti = self.get_time_hv(t, time_hvs)  # Timestamp HV
            #todo: normalize all here?

            # Debug first event
            if event_index < 1:
                self.debug_event(event, time_hvs)


            Ei = torchhd.bind(torchhd.bind(P_xy, I_hv), T_ti)
            E_temporal = torchhd.bundle(E_temporal, Ei)


            # Perform intermediate checks every 100000 events
            if (event_index + 1) % 100000 == 0:
                mean_val = E_temporal.float().mean().item()
                std_val = E_temporal.float().std().item()
                print(f"Intermediate Check after {event_index + 1} events: Mean = {mean_val:.4f}, Std = {std_val:.4f}")
        # Bundle all event hypervectors into a single composite hypervector
        #E_temporal = torchhd.multibundle(torch.stack(E_list))

        #E_temporal = torchhd.ensure_vsa_tensor(E_temporal, "MAP", )
        torchhd.normalize(E_temporal)
        print(f"Temporal Encoding Complete | Class ID: {class_id} | Output Shape: {E_temporal.shape} | Device: {E_temporal.device}", '\n', E_temporal)

        return E_temporal

    def debug_event(self, event, time_hvs):
        """
        Debug a single event and its intermediate hypervectors.

        Args:
            event (tuple): A single event (t, (x, y), polarity).
            time_hvs (torch.Tensor): The precomputed time hypervectors for this sample.

        Returns:
            None
        """
        t, (x, y), polarity = event
        P_xy = self.get_position_hv(x, y)
        I_hv = self.H_I_plus if polarity == 1 else self.H_I_minus
        T_ti = self.get_time_hv(t, time_hvs)

        # Convert MAPTensor to standard tensor for numerical operations
        P_xy_torch = P_xy.to_dense().float()
        I_hv_torch = I_hv.to_dense().float()
        T_ti_torch = T_ti.to_dense().float()

        print(f"\n--- Debugging Event: (t={t}, x={x}, y={y}, polarity={polarity}) ---")
        print(f"Position HV: {P_xy_torch.shape}, First 5 values: {P_xy_torch[:5]}")
        print(
            f"Polarity HV: {I_hv_torch.shape}, Mean: {I_hv_torch.mean().item():.4f}, Std: {I_hv_torch.std().item():.4f}")
        print(f"Time HV: {T_ti_torch.shape}, Mean: {T_ti_torch.mean().item():.4f}, Std: {T_ti_torch.std().item():.4f}")

        Ei = torchhd.bind(torchhd.bind(P_xy, I_hv), T_ti)
        Ei_torch = Ei.to_dense().float()
        print(f"Event HV (Ei): {Ei_torch.shape}, Mean: {Ei_torch.mean().item():.4f}, Std: {Ei_torch.std().item():.4f}")


"""
     Statial encoding??? H spatial doesnt seem to be used so.
        def encode_spatial(self, events):
        
        ###Spatial Encoding:
        #For each triggered event (x, y, polarity), compute H_spatial as:
        #H_spatial = sum_x sum_y (P(x,y) bind I)
        #- (binding) dissimilar
        #- (bundle/Summing) preserves similarities for the accumulated event representations.
        
        print(f'spatial encoding processing  {len(events)} events...') #debug
        encoded_events = []
        for event in events:
            t, (x, y), polarity = event

            P_xy = self.get_position_hv(x, y)  # Get position HV

            I_hv = self.H_I_plus if polarity == 1 else self.H_I_minus  # Get polarity HV

            encoded_event = torchhd.bind(P_xy, I_hv)
            encoded_events.append(encoded_event)

        H_spatial = torchhd.multibundle(torch.stack(encoded_events))

        print(f'Spatial Encoding Complete | Output Shape: {H_spatial.shape}')
        return H_spatial  # Final spatial encoding representing all events
    """


