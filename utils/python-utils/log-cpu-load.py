import psutil as psu
import sys
import signal
import time
from datetime import datetime
import csv

def generate_signal_handler(file_obj):
    def signal_handler(signum,frame):
        file_obj.close()
        exit(0)
    return signal_handler


if __name__ == "__main__":
    file_path = sys.argv[1]
    delay = float(sys.argv[2])
    log_file = open(file_path,"w",newline='')
    signal.signal(signal.SIGINT, generate_signal_handler(log_file))
    
    cpu_count = psu.cpu_count(logical=False)
    fieldnames = ['time'] + [f"cpu_{i}" for i in range(cpu_count)]
    writer = csv.DictWriter(log_file,fieldnames=fieldnames)

    writer.writeheader()


    while True:
        curr_time = datetime.now()
        time_dir = {'time': curr_time.isoformat()}
        cpu_load = psu.cpu_percent(interval=0.5,percpu=True)
        cpu_dir = {f'cpu_{i}': v for i,v in enumerate(cpu_load)}
        final_dir = {**time_dir, **cpu_dir}
        writer.writerow(final_dir)
        log_file.flush()
        if delay > 1:
            time.sleep(delay)
        else:
            time.sleep(1)


