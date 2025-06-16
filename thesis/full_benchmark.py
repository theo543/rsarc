from os import system, urandom
from time import sleep, time_ns
from pathlib import Path
from sys import argv
from random import randint
import threading
import psutil

def clear_file_cache():
    # Et = Empty Standby List
    assert system("RAMMap64 -Et") == 0, "Failed to run RAMMap64 to clear file cache. Make sure it is installed and in PATH."
    sleep(1)

def gen_random(f: Path, size: int):
    CHUNK = 1024 * 1024 * 64
    with open(f, "wb") as file:
        for _ in range(size // CHUNK):
            file.write(urandom(CHUNK))
        if size % CHUNK != 0:
            file.write(urandom(size % CHUNK))

BLOCK = 16384
PARITY_PERCENT = 20

def corrupt_file(f: Path, p: Path):
    f_size = f.stat().st_size
    p_size = p.stat().st_size
    blocks = (f_size + BLOCK - 1) // BLOCK
    parity_blocks = ((blocks + PARITY_PERCENT - 1) * PARITY_PERCENT) // 100
    max_errors = (blocks * PARITY_PERCENT) // 100
    with open(f, "rb+") as file:
        for _ in range(max_errors // 2):
            block_idx = randint(0, blocks)
            file.seek(block_idx * BLOCK)
            remaining_size = f_size - block_idx * BLOCK
            file.write(urandom(BLOCK)[:remaining_size])
    with open(p, "rb+") as file:
        metadata_size = 100 + (blocks + parity_blocks) * 40
        for _ in range(max_errors // 2):
            block_idx = randint(0, blocks)
            file.seek(metadata_size + block_idx * BLOCK)
            remaining_size = p_size - metadata_size - block_idx * BLOCK
            file.write(urandom(BLOCK)[:remaining_size])

def cpu_measure_thread(stop_event: threading.Event, out: list[int]):
    SAMPLING_RATE = 0.2 # seconds
    SAMPLING_RATE_NS = 2 * 10**8 # 0.2 * 10^9
    CPU_THRESHOLD = 15
    high_usage_time = 0
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=SAMPLING_RATE)
        if cpu_usage > CPU_THRESHOLD:
            high_usage_time += SAMPLING_RATE_NS
    out.append(high_usage_time)

def run_timed(command: str, name: str, log_file: Path):
    stop_event = threading.Event()
    out = []
    cpu_thread = threading.Thread(target=cpu_measure_thread, args=(stop_event, out), daemon=True)
    start = time_ns()
    cpu_thread.start()
    assert system(command) == 0, f"Command failed: {command}"
    stop_event.set()
    cpu_thread.join()
    duration = time_ns() - start
    with open(log_file, "a", encoding="ascii") as log:
        log.write(f"{name},{duration},{out[0]}\n")

def main():
    print(f"Start size = {2 ** int(argv[1])} GB, End size = {2 ** int(argv[2])} GB")

    files = [(1024 * 1024 * 1024 * (2 ** x), f"GB{2 ** x}.bin") for x in range(int(argv[1]), int(argv[2]) + 1)]
    parity_count = [(size // BLOCK) * PARITY_PERCENT // 100 for size, _ in files]

    for (s, f) in files:
        f = Path(f)
        if not f.exists():
            print(f"Generating {f}")
            gen_random(f, s)

    log = Path("end_to_end_benchmark.csv")

    if "encode" in argv:
        for ((s, f), p) in zip(files, parity_count):
            clear_file_cache()
            run_timed(f"cargo run --release encode {f} {f}.rsarc {BLOCK} {p}", f"encode,{s // (1024 * 1024 * 1024)}", log)

    if "decode" in argv:
        for ((s, f), p) in zip(files, parity_count):
            data = Path(f)
            parity = Path(f"{f}.rsarc")
            corrupt_file(data, parity)
            clear_file_cache()
            run_timed(f"cargo run --release repair {data} {parity}", f"decode,{s // (1024 * 1024 * 1024)}", log)

if __name__ == "__main__":
    main()
