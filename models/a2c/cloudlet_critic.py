# models/a2c/cloudlet_critic.py
import threading
import time
import random
from models.a2c.critic import TabularCritic

class HolisticCloudlet:
    def __init__(self, window_ms=50, beta=1.0, gamma=0.95, alpha_global=0.1,
                 global_critic_weights=None):
        self.window_ms = window_ms
        self.beta = beta
        self.gamma = gamma
        self.alpha_global = alpha_global
        self.lock = threading.Lock()
        self.energy_reports = []          # (timestamp, device_id, energy)
        self.deadline_reports = []        # (timestamp, device_id, missed)
        self.global_critic = TabularCritic()
        if global_critic_weights is not None:
            self.global_critic.set_weights(global_critic_weights)
        self.replay_buffer = []           # (s_key, a_key, reward, s_next_key)
        self.pending_weights = None
        self.device_agents = []

        # Statistics (accumulated until Ctrl+C)
        self.episode_stats = {}           # device_id -> [sum_energy, sum_misses, count]
        self.lock_stats = threading.Lock()
        self.total_episodes_processed = 0

    # --- Reporting from devices ---
    def send_energy_report(self, device_id, energy, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        with self.lock:
            self.energy_reports.append((timestamp, device_id, energy))
            now = time.time()
            self.energy_reports = [(ts, d, e) for ts, d, e in self.energy_reports if ts > now - 2.0]

    def send_deadline_miss(self, device_id, missed, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        with self.lock:
            self.deadline_reports.append((timestamp, device_id, 1 if missed else 0))
            now = time.time()
            self.deadline_reports = [(ts, d, m) for ts, d, m in self.deadline_reports if ts > now - 2.0]

    def send_trajectory(self, device_id, trajectory):
        with self.lock:
            self.replay_buffer.extend(trajectory)
            if len(self.replay_buffer) > 50000:
                self.replay_buffer = self.replay_buffer[-50000:]

    def send_episode_stats(self, device_id, episode_energy, episode_misses):
        """episode_misses = 1 if global deadline missed, else 0"""
        with self.lock_stats:
            if device_id not in self.episode_stats:
                self.episode_stats[device_id] = [0.0, 0, 0]  # sum_energy, sum_misses, count
            stats = self.episode_stats[device_id]
            stats[0] += episode_energy
            stats[1] += episode_misses
            stats[2] += 1
            self.total_episodes_processed += 1
            # No automatic printing – we'll print only at the end

    # --- Global reward computation ---
    def compute_global_reward(self):
        now = time.time()
        start = now - self.window_ms / 1000.0
        total_energy = 0.0
        total_miss = 0
        with self.lock:
            for ts, _, e in self.energy_reports:
                if ts >= start:
                    total_energy += e
            for ts, _, m in self.deadline_reports:
                if ts >= start:
                    total_miss += m
        return -total_energy - self.beta * total_miss

    def broadcast_global_reward(self):
        r = self.compute_global_reward()
        for agent in self.device_agents:
            agent.set_global_reward(r)

    # --- Global critic training ---
    def update_global_critic(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        for s_key, a_key, reward, s_next_key in batch:
            V = self.global_critic.get(s_key)
            V_next = self.global_critic.get(s_next_key) if s_next_key is not None else 0.0
            target = reward + self.gamma * V_next
            td_error = target - V
            self.global_critic.update(s_key, td_error, self.alpha_global)
        with self.lock:
            self.pending_weights = self.global_critic.get_weights()

    def broadcast_global_weights(self):
        with self.lock:
            if self.pending_weights is not None:
                weights = self.pending_weights
                self.pending_weights = None
                for agent in self.device_agents:
                    agent.local_critic.set_weights(weights)

    def get_global_critic_weights(self):
        with self.lock:
            return self.global_critic.get_weights()

    # --- Final statistics printing (only called at exit) ---
    def print_statistics(self):
        with self.lock_stats:
            if not self.episode_stats:
                print("\n[CLOUDLET] No episodes completed. No statistics to show.")
                return
            print("\n========== Holistic MARL Final Statistics ==========")
            print(f"Total episodes across all devices: {self.total_episodes_processed}")
            total_energy_all = 0.0
            total_misses_all = 0
            total_episodes_all = 0
            for dev_id, (sum_e, sum_m, cnt) in self.episode_stats.items():
                avg_energy = sum_e / cnt if cnt > 0 else 0.0
                miss_rate = (sum_m / cnt) * 100 if cnt > 0 else 0.0
                print(f"Device {dev_id}: episodes={cnt}, avg energy={avg_energy:.4f} J, global deadline miss rate={miss_rate:.2f}%")
                total_energy_all += sum_e
                total_misses_all += sum_m
                total_episodes_all += cnt
            if total_episodes_all > 0:
                avg_energy_overall = total_energy_all / total_episodes_all
                miss_rate_overall = (total_misses_all / total_episodes_all) * 100
                print(f"OVERALL: avg energy={avg_energy_overall:.4f} J, global deadline miss rate={miss_rate_overall:.2f}%")
            print("===================================================\n")

    # --- Background threads ---
    def set_devices(self, agents):
        self.device_agents = agents

    def run_reward_broadcast(self, stop_event, interval_ms=50):
        print("[CLOUDLET] Reward broadcast thread started.")
        while not stop_event.is_set():
            self.broadcast_global_reward()
            time.sleep(interval_ms / 1000.0)

    def run_global_critic_update(self, stop_event, update_interval_sec=5.0):
        print("[CLOUDLET] Global critic update thread started.")
        while not stop_event.is_set():
            self.update_global_critic()
            self.broadcast_global_weights()
            time.sleep(update_interval_sec)