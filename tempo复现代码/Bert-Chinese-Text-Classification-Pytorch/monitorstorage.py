import GPUtil
import time

def record_gpu_memory(device_id, log_file):
    with open(log_file, 'w') as file:
        while True:
            gpu = GPUtil.getGPUs()[device_id]
            used_memory = gpu.memoryUsed
            total_memory = gpu.memoryTotal
            usage_percentage = (used_memory / total_memory) * 100

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - GPU {device_id}: Used Memory {used_memory}MB, Total Memory {total_memory}MB, Usage {usage_percentage}%\n"
            print(log_entry)
            file.write(log_entry)
            file.flush()

            time.sleep(0.1)  # Adjust the interval as needed

if __name__ == "__main__":
    device_id = 0  # Change this to the desired GPU device ID
    log_file = "gpu_memory_log.txt"
    record_gpu_memory(device_id, log_file)
