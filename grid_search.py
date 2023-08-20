import argparse
from pathlib import Path
import subprocess
import zipfile
import itertools


def main(L, N, S):
    with Path("seeds.txt").open("w") as f:
        for seed in range(100):
            line = f"{seed} {L} {N} {S}"
            print(line, file=f)
    out = subprocess.run("cargo run --release --bin gen seeds.txt", shell=True, check=True, capture_output=True)
    print(out)




if __name__ == "__main__":
    LNs = [(10, 100), (20, 90), (30, 80), (40, 70), (50, 60)]
    Ss = [1, 25, 100, 225, 400, 625, 900]
    for (L, N), S in itertools.product(LNs, Ss):
        main(0, 0, 0)
        break