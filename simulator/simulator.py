# simulator/simulator.py
import numpy as np
import random
import threading
import pandas as pd
import os
from bisect import bisect_left
from profiling.profiling_class import ProfilingData

# ---------- Contention CSV loading ----------
contention_csv_path = os.path.join("simulator", "data", "contention.csv")
df_contention = pd.read_csv(contention_csv_path)

def get_contention_data(n_yolos_inference, n_llama_inference, n_bart_inference):
    row = df_contention[
        (df_contention['n_yolos'] == n_yolos_inference) &
        (df_contention['n_llama'] == n_llama_inference) &
        (df_contention['n_bart'] == n_bart_inference)
    ]
    if row.empty:
        return {"Llama contention ": 0.0, "Yolos contention": 0.0, "Bart contention": 0.0}
    return {
        "Llama contention ": float(row.iloc[0]['Llama contention']),
        "Yolos contention": float(row.iloc[0]['Yolos contention']),
        "Bart contention": float(row.iloc[0]['Bart contention'])
    }

# ---------- Bandwidth Tracker (unchanged) ----------
class BandwidthTracker:
    def __init__(self, bandwidth_data):
        valid_data = [(float(t), float(bw)) for t, bw in bandwidth_data if bw is not None and not pd.isna(bw)]
        if not valid_data:
            raise ValueError("No valid bandwidth data provided")
        valid_data.sort(key=lambda x: x[0])
        self.timestamps = np.array([t for t, _ in valid_data], dtype=float)
        self.bandwidths = np.array([bw for _, bw in valid_data], dtype=float)
        self.min_timestamp = float(self.timestamps[0])
        self.normalized_timestamps = self.timestamps - self.min_timestamp

    def get_bandwidth_at_time(self, time_seconds: float, use_normalized: bool = False) -> float:
        if use_normalized:
            query_time = float(time_seconds)
        else:
            query_time = float(time_seconds - self.min_timestamp)
        if query_time <= self.normalized_timestamps[0]:
            return float(self.bandwidths[0])
        if query_time >= self.normalized_timestamps[-1]:
            return float(self.bandwidths[-1])
        idx = bisect_left(self.normalized_timestamps, query_time)
        t0 = self.normalized_timestamps[idx-1]
        t1 = self.normalized_timestamps[idx]
        b0 = self.bandwidths[idx-1]
        b1 = self.bandwidths[idx]
        ratio = (query_time - t0) / (t1 - t0)
        bandwidth = b0 + ratio * (b1 - b0)
        return float(max(1.0, bandwidth))

def load_bandwidth_data_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['bandwidth_mbps'] = pd.to_numeric(df['bandwidth_mbps'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'bandwidth_mbps'])
    return list(zip(df['timestamp'].astype(float).values, df['bandwidth_mbps'].astype(float).values))

# ---------- Main Simulator with GLOBAL contention counters ----------
class CloudEdgeSimulator:
    # GLOBAL counters shared by all instances
    _global_i = 0
    _global_j = 0
    _global_k = 0
    _lock = threading.Lock()

    def __init__(self, profiling_data: ProfilingData, device_id=None):
        self.profiling = profiling_data
        self.device_id = device_id
        self.bandwidth_tracker = None
        self.cumulative_time_seconds = 0.0
        self.episode_offset = 0.0
        self.episode_start_time = 0.0

        # Load bandwidth data
        bw_data_type = f"bw_data_{profiling_data.simulation_data_type}.csv"
        bandwidth_csv_path = os.path.join("simulator", "data", bw_data_type)
        if os.path.exists(bandwidth_csv_path):
            try:
                bandwidth_data = load_bandwidth_data_from_csv(bandwidth_csv_path)
                if bandwidth_data:
                    self.bandwidth_tracker = BandwidthTracker(bandwidth_data)
            except Exception as e:
                print(f"⚠️ Could not load bandwidth data: {e}")
                print("Falling back to stochastic bandwidth")

    def reset_episode_time(self):
        self.cumulative_time_seconds = 0.0
        self.episode_start_time = 0.0
        if self.bandwidth_tracker:
            max_time = self.bandwidth_tracker.normalized_timestamps[-1]
            self.episode_offset = random.uniform(0, max_time * 0.9)

    def get_current_bandwidth(self) -> float:
        if self.bandwidth_tracker:
            query_time = float(self.cumulative_time_seconds + self.episode_offset)
            bw_mbps = self.bandwidth_tracker.get_bandwidth_at_time(query_time, use_normalized=True)
            return float(bw_mbps / 8.0)   # MBps
        return float(random.uniform(5, 100))

    def get_next_state_cloud_waiting_time(self, next_layer, current_action, isAllCloud=False):
        layer = int(next_layer)
        cloud_nodes = np.where(current_action[:, 1] == 1)[0]
        if len(cloud_nodes) == 0:
            return 0.0

        if not isAllCloud:
            with self._lock:
                # Random walk on GLOBAL counters
                if random.random() < 0.05:
                    self.__class__._global_i = min(self.__class__._global_i + 1, 3)
                elif random.random() < 0.05:
                    self.__class__._global_i = max(self.__class__._global_i - 1, 0)
                if random.random() < 0.05:
                    self.__class__._global_j = min(self.__class__._global_j + 1, 3)
                elif random.random() < 0.05:
                    self.__class__._global_j = max(self.__class__._global_j - 1, 0)
                if random.random() < 0.05:
                    self.__class__._global_k = min(self.__class__._global_k + 1, 3)
                elif random.random() < 0.05:
                    self.__class__._global_k = max(self.__class__._global_k - 1, 0)
            n_yolos = self.__class__._global_i
            n_llama = self.__class__._global_j
            n_bart = self.__class__._global_k
        else:
            n_yolos = n_llama = n_bart = 3

        contention_row = get_contention_data(n_yolos, n_llama, n_bart)
        contention = max(contention_row["Yolos contention"],
                         contention_row["Llama contention "],
                         contention_row["Bart contention"])

        cloud_times = [self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes]
        max_cloud_ms = max(cloud_times)
        new_cloud_pending = contention + max_cloud_ms

        if isAllCloud and len(cloud_nodes) > 0:
            new_cloud_pending = max_cloud_ms * self.profiling.numberOfEdgeDevice

        return new_cloud_pending

    # ---------- The following methods are exactly as in your original code ----------
    def get_next_state(self, current_state, action, surplus, negative_surplus_count, new_cloud_pending):
        bandwidth, _, layer, _, _, _ = current_state
        layer = int(layer)
        new_bandwidth = max(1.0, min(self.get_current_bandwidth(), 100))
        terminal = (layer + 1 >= len(self.profiling.layers))
        next_layer = layer + 1 if not terminal else layer
        next_state = (new_bandwidth, new_cloud_pending, next_layer, action.copy(), surplus, negative_surplus_count)
        return next_state, terminal, new_cloud_pending

    def compute_energy_and_time(self, current_state, current_action, cloud_pending_ms):
        bandwidth, _, layer, prev_action, _, _ = current_state
        layer = int(layer)
        total_energy = 0.0
        profiling = self.profiling
        deps = profiling.dependencies

        # Transmission time (dependency based)
        transmission_times = []
        if prev_action is not None and layer > 0:
            prev_assign = np.asarray(prev_action[:, 1], dtype=int)
            curr_assign = np.asarray(current_action[:, 1], dtype=int)
            for curr_node in range(len(curr_assign)):
                parent_nodes = deps.get((layer, curr_node), [])
                for (p_layer, p_node) in parent_nodes:
                    parent_loc = prev_assign[p_node] if p_layer == layer-1 else 0
                    curr_loc = curr_assign[curr_node]
                    if parent_loc != curr_loc:
                        out_size = profiling.get_output_size(layer, curr_node)
                        trans_time = max((out_size/1024.0)/max(bandwidth,1e-6), profiling.rtt/1000.0)
                        transmission_times.append(trans_time)
        else:
            for i in range(len(current_action)):
                if current_action[i,1] == 1:
                    trans_time = max((profiling.get_input_size()/1024.0)/max(bandwidth,1e-6), profiling.rtt/1000.0)
                    transmission_times.append(trans_time)

        max_transmission = max(transmission_times) if transmission_times else 0.0
        if max_transmission > 0:
            total_energy += profiling.edge_communication_power * max_transmission

        # Edge processing
        edge_times = []
        edge_energies = []
        for i in range(len(current_action)):
            if current_action[i,1] == 0:
                t = profiling.get_node_edge_time(layer, i) / 1000.0
                p = profiling.get_node_edge_power(layer, i)
                edge_times.append(t)
                edge_energies.append(p * t)

        if layer in [3,5]:
            edge_total_time = max(edge_times) if edge_times else 0.0
            total_energy += max(edge_energies) if edge_energies else 0.0
        else:
            edge_total_time = sum(edge_times)
            total_energy += sum(edge_energies)

        # Cloud idle energy
        actual_idle = 0.0
        if np.any(current_action[:,1] == 1):
            cloud_pending_s = cloud_pending_ms / 1000.0
            actual_idle = max(0.0, cloud_pending_s - edge_total_time)
            total_energy += profiling.edge_idle_power * actual_idle

        completion_time_s = edge_total_time + max_transmission + actual_idle
        self.cumulative_time_seconds += completion_time_s
        return total_energy, completion_time_s

    def calculate_reward(self, layer, total_energy, completion_time_s, previous_surplus, negative_surplus_count, isA2C=False):
        fractional_deadline_ms = (self.profiling.get_edge_time_for_layer(layer) / self.profiling.get_total_edge_time()) * self.profiling.deadline
        completion_time_ms = completion_time_s * 1000.0
        effective_deadline_ms = fractional_deadline_ms + previous_surplus
        surplus_ms = effective_deadline_ms - completion_time_ms
        if surplus_ms < 0:
            negative_surplus_count += 1
        energy_penalty = total_energy * 1000.0
        reward = energy_penalty
        if isA2C:
            reward *= 0.15
        reward *= -1.0
        return reward, surplus_ms, negative_surplus_count, fractional_deadline_ms