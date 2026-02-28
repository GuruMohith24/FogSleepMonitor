import sys
import os
import time

# Ensure we're running with the right path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fog_node.fog_service import FogProcessingNode

def run_test_sample():
    print("=== Starting Live Dashboard Test Sample ===")
    node = FogProcessingNode()
    
    # 1. Warm-up buffer
    current_time = int(time.time() * 1000)
    for i in range(29):
        current_time += 100
        # Start building the buffer with stable sleep
        node.data_buffer.append([current_time, 50, -20, 10, 512])

    # 2. Simulate Stable Sleep sequence
    print("Generating Stable Sleep Signals...")
    for _ in range(30):
        current_time += 100
        node.data_buffer.append([current_time, 15, -10, 5, 510])
        if len(node.data_buffer) > 30:
            node.data_buffer.pop(0)
        node.process_window()
        time.sleep(0.01)
        
    # 3. Sudden Disturbance (Awake/Nightmare)
    print("Generating Sudden Disturbance (High Movement & High Heart-Rate)...")
    for _ in range(30):
        current_time += 100
        # Huge spikes in accelerometer + high heart rate
        node.data_buffer.append([current_time, 3500, -2800, 4100, 780])
        if len(node.data_buffer) > 30:
            node.data_buffer.pop(0)
        node.process_window()
        time.sleep(0.01)
        
    print("=== Test Sample Finished! Check Dashboard ===")

if __name__ == "__main__":
    run_test_sample()
