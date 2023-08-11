from __future__ import annotations
import sys
from typing import NamedTuple
import math
import logging

logging.basicConfig(level="ERROR")
logger = logging.getLogger()


class ExitCell(NamedTuple):
    y: int
    x: int


class FieldInitializer:
    def __init__(self, L: int, p_min: int = 0, p_max: int = 1000) -> None:
        self.L = L
        self.p_min = p_min
        self.p_max = p_max

    def _minmax_scale(self, v, v_min, v_max):
        std = (v - v_min) / (v_max - v_min)
        return std * (self.p_max - self.p_min) + self.p_min

    def get_zero_field(self) -> list[list[int]]:
        return [[0 for y in range(self.L)] for x in range(self.L)]

    def get_cosine_field(self) -> list[list[int]]:
        p = self.get_zero_field()
        period = 2 * math.pi / self.L
        for y in range(self.L):
            for x in range(self.L):
                v = self._minmax_scale(
                    math.cos(y * period) + math.cos(x * period), -2, 2
                )
                p[y][x] = round(v)
        return p


def read0() -> tuple[int, int, int, list[ExitCell]]:
    L, N, S = map(int, input().strip().split())
    exit_cells = []
    for i in range(N):
        y, x = map(int, input().strip().split())
        exit_cells.append(ExitCell(y, x))
    return L, N, S, exit_cells


def write_cells(P: list[list[int]], PMIN=0, PMAX=1000):
    for line in P:
        print(*line)
    sys.stdout.flush()


def write_measure(i: int, y: int, x: int):
    print(i, y, x, flush=True)


def read_prediction() -> int:
    p = int(input().strip())
    return p


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
        self.exit_cells = []
        self.P = None
        self.wormholes_p: list[int] = None

    def placement_step(self):
        self.L, self.N, self.S, self.exit_cells = read0()
        field = FieldInitializer(self.L)
        self.P = field.get_cosine_field()
        self.wormholes_p = [0] * self.N
        write_cells(self.P)

    def measure(self, i: int, y: int, x: int) -> int:
        write_measure(i, y, x)
        return read_prediction()

    def measurement_step(self):
        # n_samples回サンプリング
        n_samples = 100
        for i in range(self.N):
            p_sampled = [self.measure(i, 0, 0) for j in range(n_samples)]
            p_e = sum(p_sampled) / n_samples
            p_ss = sum([(p - p_e) * (p - p_e) for p in p_sampled]) / n_samples
            p_std = math.sqrt(p_ss)
            self.wormholes_p[i] = round(p_e)
            logger.info(f"{i}th sample has p_e={p_e:.03f}, p_std={p_std:.03f}")

    def answer(self, wormholes):
        write_answer(wormholes)

    def answer_step(self):
        actual, expect = [], []
        for i in range(self.N):
            exit_cell = self.exit_cells[i]
            p_gt = self.P[exit_cell.y][exit_cell.x]
            actual.append((p_gt, i))
        for i in range(self.N):
            wormhole_p = self.wormholes_p[i]
            expect.append((wormhole_p, i))
        actual.sort()
        expect.sort()
        logger.info(actual)
        logger.info(expect)
        wormholes = [-1] * self.N
        for k in range(self.N):
            # i-th wormhole is j-th exitcell
            i = expect[k][1]
            j = actual[k][1]
            wormholes[i] = j
        self.answer(wormholes)

    def run(self):
        self.placement_step()
        self.measurement_step()
        self.answer_step()


if __name__ == "__main__":
    Solver().run()
