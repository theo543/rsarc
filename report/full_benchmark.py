from os import system, urandom
from time import sleep, time_ns
import ctypes
from pathlib import Path

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

def kb(x: int): return x * 1024, f"KB{x}.bin"
def mb(x: int): return kb(x)[0] * 1024, f"MB{x}.bin"
def gb(x: int): return mb(x)[0] * 1024, f"GB{x}.bin"

def main():
    assert ctypes.windll.shell32.IsUserAnAdmin() != 0, "This script must be run as an admin since flushing the file cache using RAMMap64 requires admin privileges."

    files = [gb(2 ** x) for x in range(6, 8)]
    BLOCK = 16384
    PARITY_PERCENT = 20
    parity_count = [(size // BLOCK) * PARITY_PERCENT // 100 for size, _ in files]

    for (s, f) in files:
        f = Path(f)
        if not f.exists():
            print(f"Generating {f}")
            gen_random(f, s)

    log = Path("benchmark.log")

    for ((s, f), p) in zip(files, parity_count):
        clear_file_cache()
        start = time_ns()
        assert system(f"cargo run --release encode {f} {f}.rsarc {BLOCK} {p}") == 0
        with open(log, "a", encoding="ascii") as log_file:
            log_file.write(f"encode {f}: {time_ns() - start} ns\n")

if __name__ == "__main__":
    main()
