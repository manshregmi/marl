# runner/run_marl.py
import threading
import time
import pickle
import os
from models.a2c.cloudlet_critic import HolisticCloudlet
from models.a2c.holistic_a2c import HolisticAgent
from profiling.initialize.graph1 import get_profiling_data

SAVE_DIR = "saved_tables"

def ensure_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

def load_all_tables(num_devices):
    """Load policy and value tables for all devices and global critic from fixed filenames."""
    policy_tables = {}
    value_tables = {}
    global_weights = None

    for dev_id in range(num_devices):
        policy_file = os.path.join(SAVE_DIR, f"device_{dev_id}_policy.pkl")
        value_file = os.path.join(SAVE_DIR, f"device_{dev_id}_value.pkl")
        if os.path.exists(policy_file) and os.path.exists(value_file):
            with open(policy_file, 'rb') as f:
                policy_tables[dev_id] = pickle.load(f)
            with open(value_file, 'rb') as f:
                value_tables[dev_id] = pickle.load(f)
        else:
            policy_tables[dev_id] = None
            value_tables[dev_id] = None

    global_file = os.path.join(SAVE_DIR, "global_critic.pkl")
    if os.path.exists(global_file):
        with open(global_file, 'rb') as f:
            global_weights = pickle.load(f)

    return policy_tables, value_tables, global_weights

def save_all_tables(agents, cloudlet):
    ensure_save_dir()
    print(f"\n[SAVE] Saving all tables to {SAVE_DIR}/ (overwriting previous saves)")
    for agent in agents:
        dev_id = agent.device_id
        policy_file = os.path.join(SAVE_DIR, f"device_{dev_id}_policy.pkl")
        value_file = os.path.join(SAVE_DIR, f"device_{dev_id}_value.pkl")
        with open(policy_file, 'wb') as f:
            pickle.dump(agent.get_policy_table(), f)
        with open(value_file, 'wb') as f:
            pickle.dump(agent.get_value_table(), f)
        print(f"  Saved Device {dev_id}")
    global_file = os.path.join(SAVE_DIR, "global_critic.pkl")
    with open(global_file, 'wb') as f:
        pickle.dump(cloudlet.get_global_critic_weights(), f)
    print(f"  Saved Global Critic")
    print(f"[SAVE] Done.")

def load_profiling():
    return get_profiling_data(deadline=500, edge_devices=8)

def device_worker(agent, stop_event):
    print(f"[WORKER] Device {agent.device_id} started")
    bw = agent.simulator.get_current_bandwidth()
    state = (bw, 0.0, 0, None, 0.0, 0)
    while not stop_event.is_set():
        action, r, next_state, term, _, _, _, _ = agent.train(state)
        state = next_state
        if term:
            agent.simulator.reset_episode_time()
            bw = agent.simulator.get_current_bandwidth()
            state = (bw, 0.0, 0, None, 0.0, 0)
        time.sleep(0.001)

def main():
    print("[MAIN] Loading profiling data...")
    profiling = load_profiling()
    num_devices = 8

    # Try to load existing tables for resuming training
    print("[MAIN] Checking for existing saved tables...")
    policy_tables, value_tables, global_weights = load_all_tables(num_devices)
    if any(v is not None for v in policy_tables.values()):
        print("[MAIN] Found existing tables. Resuming training.")
    else:
        print("[MAIN] No existing tables found. Starting from scratch.")

    print("[MAIN] Creating cloudlet...")
    cloudlet = HolisticCloudlet(window_ms=50, beta=1.0, gamma=0.95, alpha_global=0.1,
                                global_critic_weights=global_weights)

    agents = []
    for i in range(num_devices):
        agent = HolisticAgent(
            device_id=i,
            cloudlet=cloudlet,
            profiling_data=profiling,
            gamma=0.95,
            alpha_actor=0.02,
            alpha_critic=0.05,
            policy_table=policy_tables.get(i, None),
            value_table=value_tables.get(i, None)
        )
        agents.append(agent)
    cloudlet.set_devices(agents)

    stop_event = threading.Event()

    print("[MAIN] Starting device threads...")
    device_threads = []
    for agent in agents:
        t = threading.Thread(target=device_worker, args=(agent, stop_event))
        t.start()
        device_threads.append(t)

    print("[MAIN] Starting cloudlet reward and critic update threads...")
    t_reward = threading.Thread(target=cloudlet.run_reward_broadcast, args=(stop_event, 50))
    t_critic = threading.Thread(target=cloudlet.run_global_critic_update, args=(stop_event, 5.0))
    t_reward.start()
    t_critic.start()

    print("[MAIN] All threads started. Training will run until Ctrl+C.")
    print("[MAIN] Statistics will be printed every 100 total episodes (across all devices).")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupt received. Saving tables before exit...")
        stop_event.set()
        time.sleep(1)  # give threads a moment to finish current step
        save_all_tables(agents, cloudlet)
        for t in device_threads:
            t.join()
        t_reward.join()
        t_critic.join()
        # Print final statistics
        cloudlet.print_statistics()
        print("[MAIN] Done.")

if __name__ == "__main__":
    main()