import torch
import torchhd
from MAIN.DAT_approach.grasphd_appraoch.with_pickle.grasphdencoding_seedhvs import seedEncoder
import numpy as np
from collections import defaultdict
import gc

np.set_printoptions(suppress=True, precision=8)


class Raw_events_HDEncoder(seedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device, max_time, time_method, WINDOW_SIZE_MS, OVERLAP_MS):
        super().__init__(height, width, dims, k, time_subwindow, device, max_time, time_method, WINDOW_SIZE_MS, OVERLAP_MS)

    def encode_eventhd(self, events, class_id):
        """Batch event encoding with multiple time encoding methods."""
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")

        if len(events) == 0:
            return torchhd.empty(1, self.dims).squeeze(0)

        TIME_BIN_SIZE = self.time_subwindow

        # **Extract Event Data**
        events = np.array(events)
        event_times = torch.tensor(events[:, 0], dtype=torch.int32, device=self.device)
        event_x = torch.tensor(events[:, 1], dtype=torch.int32, device=self.device)
        event_y = torch.tensor(events[:, 2], dtype=torch.int32, device=self.device)
        event_polarity = torch.tensor(events[:, 3], dtype=torch.int8, device=self.device)

        # **Fetch Position Hypervectors**
        P_xy = torch.stack([self.get_position_hv(int(x), int(y)) for x, y in zip(event_x, event_y)])

        # **Assign Polarity HVs**
        H_I_on_expanded = self.H_I_on.expand(len(event_polarity), self.dims)
        H_I_off_expanded = self.H_I_off.expand(len(event_polarity), self.dims)
        I_p = torch.where(event_polarity.unsqueeze(-1) == 1, H_I_on_expanded, H_I_off_expanded)

        # **Compute Spatial Encoding**
        H_spatial = torchhd.bind(P_xy, I_p)

        # **Group Events by Time Bin**
        time_bins = ((event_times // TIME_BIN_SIZE) * TIME_BIN_SIZE).long()
        temporal_dict = defaultdict(list)
        for i in range(len(time_bins)):
            temporal_dict[time_bins[i].item()].append(H_spatial[i])

        # **Process Time Bins**
        sorted_time_bins = sorted(temporal_dict.keys())
        permuted_time_hvs = []
        bundled_time_hvs = []

        for index, time_bin in enumerate(sorted_time_bins):
            spatial_hvs = torch.stack(temporal_dict[time_bin])
            SE_t = (torchhd.multibundle(spatial_hvs))

            if self.time_interpolation_method == "event_hd_timepermutation":
                H_timebin = torchhd.permute(SE_t, shifts=index)
                permuted_time_hvs.append(H_timebin.detach())

            elif self.time_interpolation_method == "event_hd_timeinterpolation":
                T_t = self.get_time_hv(time_bin)
                H_timebin = torchhd.bind(SE_t, T_t)
                bundled_time_hvs.append(H_timebin.detach())

            elif self.time_interpolation_method == "stem_hd":
                T_t = self.get_time_hv(time_bin)
                H_timebin = SE_t.clone()
                H_timebin[self.dims // 2:] = T_t[-self.dims // 2:]
                bundled_time_hvs.append(H_timebin.detach())

        # **Final Temporal Bundling**
        if self.time_interpolation_method == "event_hd_timepermutation":
            return torchhd.multibundle(torch.stack(permuted_time_hvs)) if permuted_time_hvs else torchhd.empty(1, self.dims).squeeze(0)
            #H_spatiotemporal = torchhd.normalize(torchhd.multibundle(torch.stack(permuted_time_hvs))) if permuted_time_hvs else torchhd.empty(1, self.dims).squeeze(0)

        elif self.time_interpolation_method in ["event_hd_timeinterpolation", "stem_hd"]:
            H_spatiotemporal = torchhd.multibundle(torch.stack(bundled_time_hvs)) if bundled_time_hvs else torchhd.empty(1, self.dims).squeeze(0)

        # **Clear Unused Tensors to Free Memory**
        del P_xy, I_p, H_spatial, event_x, event_y, event_polarity, event_times, temporal_dict
        gc.collect()

        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
        return H_spatiotemporal


    def process_windows(self, full_events, class_id):
        """Splits event sequence into sliding windows and encodes them."""
        event_hvs = []
        total_events = len(full_events)

        if total_events == 0:
            print(f"[INFO] Skipping empty sample | Class: {class_id}")
            return []

        first_t, last_t = full_events[0][0], full_events[-1][0]
        total_duration = last_t - first_t

        expected_windows = max(1, (total_duration - self.OVERLAP_MS) // max(1, (self.WINDOW_SIZE_MS - self.OVERLAP_MS)) + 1)

        print(f"\n[INFO] Encoding Sample | Class: {class_id} | Total Events: {total_events}")
        print(f"      - First Timestamp: {first_t}, Last Timestamp: {last_t}")
        print(f"      - Total Duration: {total_duration} ms")
        print(f"      - Expected Windows: {expected_windows}")

        start_time = first_t
        end_time = start_time + self.WINDOW_SIZE_MS
        window_index = 0
        skipped_windows = 0
        window_events = []

        for t, x, y, polarity in full_events:
            if t >= end_time:
                if window_events:
                    adjusted_events = [(t - start_time, x, y, p) for t, x, y, p in window_events]
                    if len(adjusted_events) >= 5:
                        print(f"[DEBUG] Window {window_index}: Start={start_time}, End={end_time}, Total Events={len(adjusted_events)}")
                        window_hv = self.encode_eventhd(adjusted_events, class_id)

                        if window_hv is not None:
                            event_hvs.append(window_hv)

                    else:
                        skipped_windows += 1

                #self.clear_temporary_cache()  # **Clear memory per window**

                start_time += (self.WINDOW_SIZE_MS - self.OVERLAP_MS)
                end_time = start_time + self.WINDOW_SIZE_MS
                window_index += 1
                window_events = []

            window_events.append((t, x, y, polarity))

        if window_events:
            adjusted_events = [(t - start_time, x, y, p) for t, x, y, p in window_events]
            if len(adjusted_events) >= 5:
                print(f"[DEBUG] Final Window {window_index}: Start={start_time}, End={end_time}, Events={len(adjusted_events)}")
                window_hv = self.encode_eventhd(adjusted_events, class_id)

                if window_hv is not None:
                    event_hvs.append(window_hv)
            else:
                skipped_windows += 1

        print(f"[INFO] Sample {class_id} - Created: {len(event_hvs)} windows | Skipped: {skipped_windows}")
        return event_hvs

'''
import torch
import torchhd
from MAIN.DAT_approach.grasphd_appraoch.with_pickle.grasphdencoding_seedhvs import seedEncoder
import numpy as np
from collections import defaultdict
import gc

np.set_printoptions(suppress=True, precision=8)


class Raw_events_HDEncoder(seedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device, max_time, time_method, WINDOW_SIZE_MS, OVERLAP_MS):
        super().__init__(height, width, dims, k, time_subwindow, device, max_time, time_method, WINDOW_SIZE_MS, OVERLAP_MS)

    def encode_eventhd(self, events, class_id):
        """Batch event encoding with multiple time encoding methods."""
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")

        if len(events) == 0:
            return torchhd.empty(1, self.dims).squeeze(0)

        TIME_BIN_SIZE = self.time_subwindow
        BATCH_SIZE = 100000  # Set batch size to process smaller chunks

        events = np.array(events)
        num_batches = len(events) // BATCH_SIZE + (len(events) % BATCH_SIZE > 0)

        all_hvs = []  # Store processed HVs in batches

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(events))
            batch_events = events[start_idx:end_idx]

            event_times = torch.tensor(batch_events[:, 0], dtype=torch.int32, device=self.device)
            event_x = torch.tensor(batch_events[:, 1], dtype=torch.int32, device=self.device)
            event_y = torch.tensor(batch_events[:, 2], dtype=torch.int32, device=self.device)
            event_polarity = torch.tensor(batch_events[:, 3], dtype=torch.int8, device=self.device)

            # Fetch Position Hypervectors
            P_xy = torch.stack([self.get_position_hv(int(x), int(y)) for x, y in zip(event_x, event_y)])

            # Assign Polarity HVs
            H_I_on_expanded = self.H_I_on.expand(len(event_polarity), self.dims)
            H_I_off_expanded = self.H_I_off.expand(len(event_polarity), self.dims)
            I_p = torch.where(event_polarity.unsqueeze(-1) == 1, H_I_on_expanded, H_I_off_expanded)

            # Compute Spatial Encoding
            H_spatial = torchhd.bind(P_xy, I_p)

            # Group Events by Time Bin
            time_bins = ((event_times // TIME_BIN_SIZE) * TIME_BIN_SIZE).long()
            temporal_dict = defaultdict(list)
            for i in range(len(time_bins)):
                temporal_dict[time_bins[i].item()].append(H_spatial[i])

            # Process Time Bins
            sorted_time_bins = sorted(temporal_dict.keys())
            permuted_time_hvs = []
            bundled_time_hvs = []

            for index, time_bin in enumerate(sorted_time_bins):
                spatial_hvs = torch.stack(temporal_dict[time_bin])
                SE_t = torchhd.normalize(torchhd.multibundle(spatial_hvs))

                if self.time_interpolation_method == "event_hd_timepermutation":
                    H_timebin = torchhd.permute(SE_t, shifts=index)
                    permuted_time_hvs.append(H_timebin.detach())

                elif self.time_interpolation_method == "event_hd_timeinterpolation":
                    T_t = self.get_time_hv(time_bin)
                    H_timebin = torchhd.bind(SE_t, T_t)
                    bundled_time_hvs.append(H_timebin.detach())

                elif self.time_interpolation_method == "stem_hd":
                    T_t = self.get_time_hv(time_bin)
                    H_timebin = SE_t.clone()
                    H_timebin[self.dims // 2:] = T_t[-self.dims // 2:]
                    bundled_time_hvs.append(H_timebin.detach())

            # Store batch results
            if self.time_interpolation_method == "event_hd_timepermutation":
                batch_hv = torchhd.normalize(torchhd.multibundle(torch.stack(permuted_time_hvs))) if permuted_time_hvs else torchhd.empty(1, self.dims).squeeze(0)

            elif self.time_interpolation_method in ["event_hd_timeinterpolation", "stem_hd"]:
                batch_hv = torchhd.normalize(torchhd.multibundle(torch.stack(bundled_time_hvs))) if bundled_time_hvs else torchhd.empty(1, self.dims).squeeze(0)

            all_hvs.append(batch_hv.detach().cpu())  # Move to CPU to free GPU memory

            # Clear unused tensors and force memory cleanup
            del event_times, event_x, event_y, event_polarity, P_xy, I_p, H_spatial, temporal_dict
            gc.collect()

        # Merge all processed batches
        if all_hvs:
            stacked_hvs = torch.stack(all_hvs)  # Ensure tensors are stacked properly
            H_spatiotemporal = torchhd.normalize(torchhd.multibundle(stacked_hvs))
        else:
            H_spatiotemporal = torchhd.empty(1, self.dims).squeeze(0)  # Return empty HV if no valid encodings
        del all_hvs
        gc.collect()

        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
        return H_spatiotemporal

    def clear_temporary_cache(self):
        """Clears all temporary computed hypervectors while keeping the cache structure."""
        print("[DEBUG] Cleared temporary caches.")

    def process_windows(self, full_events, class_id):
        """Splits event sequence into sliding windows and encodes them."""
        event_hvs = []
        total_events = len(full_events)

        if total_events == 0:
            print(f"[INFO] Skipping empty sample | Class: {class_id}")
            return []

        first_t, last_t = full_events[0][0], full_events[-1][0]
        total_duration = last_t - first_t

        expected_windows = max(1, (total_duration - self.OVERLAP_MS) // max(1, (self.WINDOW_SIZE_MS - self.OVERLAP_MS)) + 1)

        print(f"\n[INFO] Encoding Sample | Class: {class_id} | Total Events: {total_events}")
        print(f"      - First Timestamp: {first_t}, Last Timestamp: {last_t}")
        print(f"      - Total Duration: {total_duration} ms")
        print(f"      - Expected Windows: {expected_windows}")

        start_time = first_t
        end_time = start_time + self.WINDOW_SIZE_MS
        window_index = 0
        skipped_windows = 0
        window_events = []

        for t, x, y, polarity in full_events:
            if t >= end_time:
                if window_events:
                    adjusted_events = [(t - start_time, x, y, p) for t, x, y, p in window_events]
                    if len(adjusted_events) >= 5:
                        print(f"[DEBUG] Window {window_index}: Start={start_time}, End={end_time}, Total Events={len(adjusted_events)}")
                        window_hv = self.encode_eventhd(adjusted_events, class_id)

                        if window_hv is not None:
                            event_hvs.append(window_hv)

                    else:
                        skipped_windows += 1

                self.clear_temporary_cache()  # **Clear memory per window**

                start_time += (self.WINDOW_SIZE_MS - self.OVERLAP_MS)
                end_time = start_time + self.WINDOW_SIZE_MS
                window_index += 1
                window_events = []

            window_events.append((t, x, y, polarity))

        if window_events:
            adjusted_events = [(t - start_time, x, y, p) for t, x, y, p in window_events]
            if len(adjusted_events) >= 5:
                print(f"[DEBUG] Final Window {window_index}: Start={start_time}, End={end_time}, Events={len(adjusted_events)}")
                window_hv = self.encode_eventhd(adjusted_events, class_id)

                if window_hv is not None:
                    event_hvs.append(window_hv)
            else:
                skipped_windows += 1

        print(f"[INFO] Sample {class_id} - Created: {len(event_hvs)} windows | Skipped: {skipped_windows}")
        return event_hvs

'''