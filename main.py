from __future__ import annotations
import sys
from typing import NamedTuple
import math
import logging

logging.basicConfig(level="INFO")
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


class FeatureExtractor:

    def __init__(self) -> None:
        # カーネルサイズ
        K = 13
        half_K = 13 // 2
        self.D = [(0, 0), (-half_K, 0), (-1, -half_K), (1, half_K), (0, half_K)]

    def extract_feature(self, L: int, P: list[list[int]], y: int, x: int) -> tuple[int, ...]:
        feature = []
        for dy, dx in self.D:
            ny = (y + dy) % L
            nx = (x + dx) % L
            v = P[ny][nx]
            feature.append(v)
        return tuple(feature)


def mse_loss(pred: list[float], target: list[int]) -> float:
    n = len(pred)
    return sum([(pred[i] - target[i])**2 for i in range(n)]) / n


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
        extractor = FeatureExtractor()
        self.exit_cell_features = []
        for exit_cell in self.exit_cells:
            feature = extractor.extract_feature(self.L, self.P, exit_cell.y, exit_cell.x)
            self.exit_cell_features.append(feature)
            logger.info(feature)
        self.wormhole_features = []
        write_cells(self.P)

    def measure(self, i: int, y: int, x: int) -> int:
        write_measure(i, y, x)
        return read_prediction()

    def measurement_step(self):
        # n_samples回サンプリング
        n_samples = 100 // 20
        # カーネルサイズ
        K = 13
        half_K = K // 2
        extractor = FeatureExtractor()
        for i in range(self.N):

            p_e = [[0 for x in range(K)] for y in range(K)]
            D = [(0, 0), (-half_K, 0), (-1, -half_K), (1, half_K), (0, half_K)]
            for dy, dx in D:
                p_e[dy+half_K][dx+half_K] = sum([self.measure(i, dy, dx) for j in range(n_samples)]) / n_samples
            feature = extractor.extract_feature(K, p_e, half_K, half_K)
            self.wormhole_features.append(feature)

    def answer(self, wormholes):
        write_answer(wormholes)

    def answer_step(self):
        INF = 1e12
        wormholes = [0] * self.N
        for i in range(self.N):
            min_loss = INF
            min_j = 0
            for j in range(self.N):
                loss = mse_loss(self.wormhole_features[i], self.exit_cell_features[j])
                if loss < min_loss:
                    min_loss = loss
                    min_j = j
            wormholes[i] = min_j
        self.answer(wormholes)

    def run(self):
        self.placement_step()
        self.measurement_step()
        self.answer_step()


if __name__ == "__main__":
    Solver().run()
