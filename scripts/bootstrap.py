import logging
import os
import subprocess
import sys
import time
from typing import List

from nvitop import select_devices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def find_available_gpus(memory_threshold: int, utilization_threshold: int, required_gpus: int) -> List[int]:
    devices = select_devices(
        format="index",
        force_index=True,
        max_memory_utilization=memory_threshold,
        max_gpu_utilization=utilization_threshold,
        min_count=required_gpus,
        tolerance=0,
        sort=False
    )

    if len(devices) >= required_gpus:
        return devices[:required_gpus]
    else:
        return []

def run_command(gpu_indices):
    gpu_str = ",".join(map(str, gpu_indices))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str

    logger.info(f"Selected GPUs: {gpu_str}")

    commands = sys.argv[1:]
    if not commands:
        logger.fatal("No command provided to execute.")
        exit(1)

    logger.info(f"Running command: {' '.join(commands)}")

    process = subprocess.Popen(
        commands,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()
    return process.returncode

def main():
    required_gpus = 5
    check_interval = 60  # seconds
    memory_threshold = 1 # percent (%)
    utilization_threshold = 1 # percent (%)

    logger.info(f"GPU monitoring started, waiting for {required_gpus} GPUs to be available...")
    logger.info(f"Conditions: Used VRAM < {memory_threshold}%, GPU utilization < {utilization_threshold}%")

    while True:
        available_gpus = find_available_gpus(
            memory_threshold, 
            utilization_threshold, 
            required_gpus
        )

        if available_gpus:
            exit_code = run_command(available_gpus)
            logger.info(f"Command execution completed with exit code: {exit_code}")
            break
        else:
            logger.info(f"Rechecking in {check_interval} seconds...")
            time.sleep(check_interval)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nUser interrupted, exiting...")
        exit(0)