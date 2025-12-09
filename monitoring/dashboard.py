import json
import matplotlib.pyplot as plt
import os

def plot_dashboard(log_file="monitoring/metrics.json"):
    if not os.path.exists(log_file):
        print("Log file not found.")
        return

    timestamps = []
    fps_values = []
    latencies = []

    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                timestamps.append(data.get('timestamp', 0))
                fps_values.append(data.get('fps', 0))
                latencies.append(data.get('processing_time', 0))
            except:
                continue

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(fps_values, label='FPS')
    plt.title('FPS over Time')
    plt.xlabel('Frame')
    plt.ylabel('FPS')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(latencies, label='Latency (ms)', color='orange')
    plt.title('Latency over Time')
    plt.xlabel('Frame')
    plt.ylabel('ms')
    plt.legend()

    plt.tight_layout()
    plt.savefig('dashboard_snapshot.png')
    print("Dashboard snapshot saved to dashboard_snapshot.png")

if __name__ == "__main__":
    plot_dashboard()
