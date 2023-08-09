import psutil

def get_running_processes():
    process_list = []
    for process in psutil.process_iter(['pid', 'name']):
        process_list.append((process.info['pid'], process.info['name']))
    return process_list

if __name__ == "__main__":
    running_processes = get_running_processes()
    for pid, name in running_processes:
        print(f"PID: {pid}, Name: {name}")
