from __future__ import annotations
import sys
from typing import NamedTuple
import random
import math
import logging
import numpy as np
import scipy.stats

logging.basicConfig(level="INFO")
logger = logging.getLogger()

SEED0 = [79, 90, 11, 72, 16, 74, 69, 24, 58, 48, 23, 15, 70, 80, 57, 51, 22, 6, 50, 37, 45, 7, 12, 61, 29, 94, 89, 87, 5, 43, 81, 26, 8, 56, 10, 0, 31, 44, 9, 21, 68, 93, 36, 40, 62, 65, 91, 85, 86, 49, 13, 71, 27, 84, 25, 35, 47, 28, 42, 75, 17, 88, 67, 64, 78, 83, 46, 77, 32, 4, 60, 19, 53, 14, 18, 20, 54, 41, 2, 66, 34, 59, 76, 63, 55, 38, 52, 92, 39, 3, 1, 33, 30, 82, 73]


class ExitCell(NamedTuple):
    y: int
    x: int


class WaveGenerator:

    def __init__(self, L: int) -> None:
        # 周期カーネル
        theta = np.array([1.0, L / (2 * np.pi)])
        x = np.arange(0, L, dtype=np.int)
        K = np.zeros((L, L))

        for i in range(L):
            for j in range(L):
                K[i, j] = \
                    np.exp(theta[0] * np.cos(np.abs(x[i]-x[j]) / theta[1]))

        lmb, u_t = np.linalg.eig(K)
        lmb, u_t = lmb.real, u_t.real
        eps = 1e-12

        A = np.dot(u_t, np.diag(np.sqrt(lmb + eps)))
        self.L = L
        self.x = x
        self.A = A

    def get_wave(self, dtype=np.float) -> np.ndarray:
        y = np.dot(self.A, scipy.stats.norm.rvs(loc=0, scale=1, size=self.L))
        return y.astype(dtype)


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

    def get_cosine_field(self, y_offset: int = 0, x_offset: int = 0) -> list[list[int]]:
        p = self.get_zero_field()
        period = 2 * math.pi / self.L
        yoff = math.pi * 2 * (y_offset / self.L)
        xoff = math.pi * 2 * (x_offset / self.L)
        for y in range(self.L):
            for x in range(self.L):
                v = self._minmax_scale(
                    math.cos(y * period - yoff) + math.cos(x * period - xoff), -2, 2
                )
                p[y][x] = round(v)
        return p

    def get_random_cosine_field(self):
        wgen = WaveGenerator(self.L)
        wgen = WaveGenerator(self.L)
        z1 = wgen.get_wave()
        z2 = wgen.get_wave()
        vmax = np.max(z1) + np.max(z2)
        vmin = np.min(z1) + np.min(z2)
        p = self.get_zero_field()
        for y in range(self.L):
            for x in range(self.L):
                v = self._minmax_scale(
                    z1[y] + z2[x], vmin, vmax
                ).astype(np.int).item()
                p[y][x] = v
        return p


class FeatureExtractor:
    def __init__(self) -> None:
        # カーネルサイズ
        K = 13
        half_K = K // 2
        self.K = K
        self.half_K = half_K
        self.D = [(0, 0), (0, -half_K), (-half_K, 0), (0, half_K), (half_K, 0)]

    def extract_feature(
        self, L: int, P: list[list[int]], y: int, x: int
    ) -> tuple[int, ...]:
        feature = []
        for dy, dx in self.D:
            ny = (y + dy) % L
            nx = (x + dx) % L
            v = P[ny][nx]
            feature.append(v)
        return tuple(feature)


def mse_loss(pred: list[float], target: list[int]) -> float:
    n = len(pred)
    return sum([(pred[i] - target[i]) ** 2 for i in range(n)]) / n


def read_parameters() -> tuple[int, int, int, list[ExitCell]]:
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


class SolverBase:
    def __init__(self, L: int, N: int, S: int, exit_cells: list[ExitCell], P=None) -> None:
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
        self.P = P
        self.wormholes_p: list[int] = None
        self.wormhole_features = []
        self.exit_cell_features = []
        self.placement_cost = 0
        self.measurement_cost = 0

    def write_cells(self, P: list[list[int]]) -> None:
        raise NotImplementedError

    def measure(self, i: int, y: int, x: int) -> int:
        raise NotImplementedError

    def write_answer(self, wormholes: list[int]) -> None:
        raise NotImplementedError

    def placement_step(self):
        if self.P is None:
            field = FieldInitializer(self.L, p_min=0, p_max=1000)
            P = field.get_cosine_field()
            self.P = P
        P = self.P
        extractor = FeatureExtractor()
        for exit_cell in self.exit_cells:
            feature = extractor.extract_feature(
                self.L, P, exit_cell.y, exit_cell.x
            )
            self.exit_cell_features.append(feature)
        self.write_cells(P)
        placement_cost = 0
        for i in range(self.L):
            i1 = (i + 1) % self.L
            for j in range(self.L):
                j1 = (j + 1) % self.L
                placement_cost += (P[i][j] - P[i][j1]) ** 2 + (P[i][j] - P[i1][j]) ** 2
        self.placement_cost = placement_cost

    def measurement_step(self):
        # n_samples回サンプリング
        n_samples = 10
        extractor = FeatureExtractor()
        # カーネルサイズ
        K = extractor.K
        half_K = extractor.half_K
        measurement_cost = 0
        for i in range(self.N):
            p_e = [[0 for x in range(K)] for y in range(K)]
            for dy, dx in extractor.D:
                p_e[dy + half_K][dx + half_K] = (
                    sum([self.measure(i, dy, dx) for j in range(n_samples)]) / n_samples
                )
                measurement_cost += 100 * (10 + abs(dy) + abs(dx)) * n_samples
            feature = extractor.extract_feature(K, p_e, half_K, half_K)
            self.wormhole_features.append(feature)
        self.measurement_cost = measurement_cost

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
        self.write_answer(wormholes)

    def run(self):
        self.placement_step()
        self.measurement_step()
        self.answer_step()


class Simulator(SolverBase):

    def __init__(self, L: int, N: int, S: int, exit_cells: list[ExitCell], P=None) -> None:
        super().__init__(L, N, S, exit_cells, P)
        # wormhole_id -> exit_cell_id
        self.exit_cell_id = list(random.sample(range(self.N), k=self.N))
        self._score = 0
    
    def write_cells(self, P: list[list[int]]) -> None:
        pass

    def measure(self, i: int, y: int, x: int) -> int:
        j = self.exit_cell_id[i]
        exit_cell = self.exit_cells[j]
        p = self.P[(exit_cell.y + y) % self.L][(exit_cell.x + x) % self.L]
        theta = random.gauss(0, self.S)
        prediction = max(0, min(1000, round(p) + theta))
        return prediction
    
    def write_answer(self, wormholes: list[int]) -> None:
        w = 0
        for i in range(self.N):
            if wormholes[i] != self.exit_cell_id[i]:
                w += 1
        score = 10 ** 14 * 0.8 ** w / (self.measurement_cost + self.placement_cost + 10**5)
        self._score = math.ceil(score)
    
    @property
    def score(self):
        return self._score

class Solver(SolverBase):

    def write_cells(self, P: list[list[int]]) -> None:
        write_cells(P)

    def measure(self, i: int, y: int, x: int) -> int:
        write_measure(i, y, x)
        return read_prediction()

    def write_answer(self, wormholes: list[int]) -> None:
        write_answer(wormholes)


if __name__ == "__main__":
    L, N, S, exit_cells = read_parameters()
    best_sim = None
    best_score = -1
    for i in range(10):
        scores = []
        for j in range(5):
            simulator = Simulator(L, N, S, exit_cells)
            simulator.run()
            score = simulator.score
            scores.append(score)
        avg_score = sum(scores) / 100
    if avg_score > best_score:
        best_score = avg_score
        best_sim = simulator
    best_sim = simulator

    logger.info(f"simulated placement cost = {best_sim.placement_cost}")
    logger.info(f"simulated measurement cost = {best_sim.measurement_cost}")
    logger.info(f"simulated score = {best_sim.score}")
    solver = Solver(L, N, S, exit_cells, P=best_sim.P)
    solver.run()
    logger.info(f"predicted placement cost = {solver.placement_cost}")
    logger.info(f"predicted measurement cost = {solver.measurement_cost}")
