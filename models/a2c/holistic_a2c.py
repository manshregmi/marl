# models/a2c/holistic_a2c.py
import numpy as np
import time
from models.a2c.critic import TabularCritic
from models.a2c.tabular_a2c import TabularActorCriticAgent
from simulator.simulator import CloudEdgeSimulator

class HolisticAgent(TabularActorCriticAgent):
    def __init__(self, device_id, cloudlet, profiling_data, gamma=0.95,
                 alpha_actor=0.02, alpha_critic=0.05,
                 policy_table=None, value_table=None):
        super().__init__(profiling_data, gamma=gamma, alpha_actor=alpha_actor, alpha_critic=alpha_critic)
        self.device_id = device_id
        self.cloudlet = cloudlet
        
        # Override policy table if provided (for resuming training)
        if policy_table is not None:
            self.policy_table = policy_table
            
        # Local critic (replace parent's value_table with TabularCritic wrapper)
        self.local_critic = TabularCritic()
        if value_table is not None:
            self.local_critic.set_weights(value_table)
        else:
            # Copy initial weights from parent's value_table (which is a dict)
            self.local_critic.set_weights(self.value_table)
            
        self.latest_global_reward = 0.0
        self.trajectory_buffer = []   # (s_key, a_key, reward, s_next_key)
        self.episode_energy = 0.0     # accumulate energy per episode
        # Global deadline miss flag (0/1) – set at episode end
        self.global_deadline_miss = 0

        # Use simulator with global contention counters
        self.simulator = CloudEdgeSimulator(profiling_data, device_id=device_id)

    def set_global_reward(self, r):
        self.latest_global_reward = r

    def get_policy_table(self):
        return self.policy_table.copy()

    def get_value_table(self):
        return self.local_critic.get_weights()

    def train(self, current_state):
        # 1) Choose action (uses parent's choose_action)
        action = self.choose_action(current_state)
        a_key = self._action_to_key(action)
        layer = int(current_state[2])

        # 2) Execute in simulator
        next_cloud = self.simulator.get_next_state_cloud_waiting_time(
            next_layer=min(layer+1, len(self.profiling.layers)-1),
            current_action=action,
            isAllCloud=False
        )
        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state, action, next_cloud
        )
        # Get surplus and deadline miss (ignore local reward)
        _, surplus, neg_count, _ = self.simulator.calculate_reward(
            layer, energy, completion_time_s,
            current_state[4], current_state[5], isA2C=False
        )
        deadline_missed = (surplus < 0)   # fractional level miss – not used for global stats
        next_state, terminal, _ = self.simulator.get_next_state(
            current_state, action, surplus, neg_count, next_cloud
        )

        # Accumulate episode energy (for average energy)
        self.episode_energy += energy

        # 3) Send energy report to cloudlet (for global reward computation)
        ts = time.time()
        self.cloudlet.send_energy_report(self.device_id, energy, ts)
        # Note: deadline_missed is NOT sent; global deadline is handled at episode end.

        # 4) Local TD update using broadcast GLOBAL reward
        s_key = self._state_to_key(current_state)
        s_next_key = self._state_to_key(next_state) if not terminal else None
        V = self.local_critic.get(s_key)
        V_next = self.local_critic.get(s_next_key) if s_next_key is not None else 0.0
        delta = self.latest_global_reward + self.gamma * V_next - V

        # Update local critic
        self.local_critic.update(s_key, delta, self.alpha_critic)

        # Update actor (policy_table) using same advantage
        old_pref = self.policy_table.get((s_key, a_key), 0.0)
        self.policy_table[(s_key, a_key)] = np.clip(old_pref + self.alpha_actor * delta, -50.0, 50.0)

        # Store transition (using global reward) for later global critic training
        self.trajectory_buffer.append((s_key, a_key, self.latest_global_reward, s_next_key))

        # 5) If episode ends, send trajectory and global deadline stats
        if terminal:
            total_time_ms = self.simulator.cumulative_time_seconds * 1000.0
            global_missed = 1 if total_time_ms > self.profiling.deadline else 0
            print(f"[DEBUG] Device {self.device_id} EPISODE FINISHED: Energy={self.episode_energy:.4f}J, Total Time={total_time_ms:.2f}ms, Deadline={self.profiling.deadline}ms, Missed={global_missed}")

            self.cloudlet.send_trajectory(self.device_id, self.trajectory_buffer)
            self.cloudlet.send_episode_stats(self.device_id, self.episode_energy, global_missed)

            # Reset for next episode
            self.trajectory_buffer = []
            self.episode_energy = 0.0
            self.global_deadline_miss = 0

        return action, self.latest_global_reward, next_state, terminal, energy, completion_time_s, surplus, deadline_missed