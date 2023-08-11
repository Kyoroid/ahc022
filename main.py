from __future__ import annotations
import sys
from typing import NamedTuple


class ExitCell(NamedTuple):
    y: int
    x: int


def read0() -> tuple[int, int, int, list[ExitCell]]:
    L, N, S = map(int, input().strip().split())
    exit_cells = []
    for i in range(N):
        exit_cells.append(ExitCell(*input().strip().split()))
    return L, N, S, exit_cells


def write_cells(P: list[list[int]], PMIN=0, PMAX=1000):
    for line in P:
        print(*line)
    sys.stdout.flush()


def write_answer(wormholes: list[int]):
    print(-1, -1, -1)
    for w in wormholes:
        print(w)
    sys.stdout.flush()


class Solver:
    def __init__(self) -> None:
        self.L: int = 0
        self.N: int = 0
        self.S: int = 0
        self.wormholes: list[int] = []
        self.exit_cells = []

    def placement_step(self):
        self.L, self.N, self.S, self.exit_cells = read0()
        self.wormholes = list(range(self.N))
        self.P = [[0 for j in range(self.L)] for i in range(self.L)]
        write_cells(self.P)

    def measurement_step(self):
        pass

    def answer(self):
        write_answer(self.wormholes)

    def run(self):
        self.placement_step()
        self.measurement_step()
        self.answer()


if __name__ == "__main__":
    Solver().run()
