from __future__ import annotations
import sys
from typing import NamedTuple
import random
from pprint import pformat
import math
import logging
import numpy as np
import scipy.stats

logging.basicConfig(level="INFO")
logger = logging.getLogger()

SEED0_GT = [79, 90, 11, 72, 16, 74, 69, 24, 58, 48, 23, 15, 70, 80, 57, 51, 22, 6, 50, 37, 45, 7, 12, 61, 29, 94, 89, 87, 5, 43, 81, 26, 8, 56, 10, 0, 31, 44, 9, 21, 68, 93, 36, 40, 62, 65, 91, 85, 86, 49, 13, 71, 27, 84, 25, 35, 47, 28, 42, 75, 17, 88, 67, 64, 78, 83, 46, 77, 32, 4, 60, 19, 53, 14, 18, 20, 54, 41, 2, 66, 34, 59, 76, 63, 55, 38, 52, 92, 39, 3, 1, 33, 30, 82, 73]


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


class Heatmap:

    def __init__(self, L: int, P: list[list[int]]):
        """ヒートマップを初期化します。
        """
        self.L = L
        self.P = P
        """ヒートマップの正確な温度"""
        placement_cost = 0
        for i in range(L):
            i1 = (i + 1) % L
            for j in range(L):
                j1 = (j + 1) % L
                placement_cost += (P[i][j] - P[i][j1]) ** 2 + (P[i][j] - P[i1][j]) ** 2
        self._placement_cost = placement_cost
    
    @classmethod
    def const_init(cls, L, y_offset: int, x_offset: int, p_min: int=0, p_max: int=1000) -> "Heatmap":
        p = [[0 for x in range(L)] for y in range(L)]
        omega = 2 * math.pi / L
        phi_y = math.pi * 2 * (-y_offset / L)
        phi_x = math.pi * 2 * (-x_offset / L)
        for y in range(L):
            for x in range(L):
                v = cls._minmax_scale(
                    math.cos(y * omega + phi_y) + math.cos(x * omega + phi_x), -2, 2, p_min, p_max
                )
                p[y][x] = round(v)
        return cls(L, p)
    
    @classmethod
    def rand_init(cls, L: int, p_min: int=0, p_max: int=1000) -> "Heatmap":
        y_offset = random.randint(0, L-1)
        x_offset = random.randint(0, L-1)
        return cls.const_init(L, y_offset, x_offset, p_min=p_min, p_max=p_max)

    @classmethod
    def _minmax_scale(cls, src_value, src_min, src_max, dst_min, dst_max):
        std = (src_value - src_min) / (src_max - src_min)
        return std * (dst_max - dst_min) + dst_min
    
    def get_placement_cost(self):
        return self._placement_cost


class Featurizer:

    def __init__(self, offset_yx: list[tuple[int, int]], num_samples: int = 10) -> None:
        self.offset_yx = offset_yx
        self.feature_size = len(offset_yx)
        self.num_samples = num_samples
        measurement_cost = 0
        for dy, dx in self.offset_yx:
            measurement_cost += 100 * (10 + abs(dy) + abs(dx)) * num_samples
        self._measurement_cost = measurement_cost
    
    @classmethod
    def rand_init(cls, window_size: int, feature_size: int, num_samples:int = 10) -> "Featurizer":
        """特徴ベクトルをランダムに生成する
        """
        indices = list(random.sample(range(window_size * window_size), k=feature_size))
        offset_yx = []
        for idx in indices:
            p, q = divmod(idx, window_size)
            dy = p - window_size // 2
            dx = q - window_size // 2
            offset_yx.append((dy, dx))
        return cls(offset_yx, num_samples)

    def get_offset_yx(self) -> list[tuple[int, int]]:
        """特徴生成に用いる相対座標(dy, dx)のリストを返す
        """
        return self.offset_yx

    def get_measurement_cost(self) -> int:
        """この特徴生成器を使う場合の計測コストを返す
        """
        return self._measurement_cost


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
    def __init__(self, L: int, N: int, S: int, exit_cells: list[ExitCell], heatmap: Heatmap, featurizer: Featurizer) -> None:
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
        self.heatmap = heatmap
        self.featurizer = featurizer
        self.max_measurement = 10000
        # p_e (=feature_e)
        self.p_e = None
        """i番目のワームホールの先にある測定対象のセルのうち、j番目のセルの測定値"""
        self.out_idx = None
        """i番目のワームホールがout_idx[i]番目の出口セルに繋がっていると予測する"""

    def write_cells(self, P: list[list[int]]) -> None:
        raise NotImplementedError

    def measure(self, i: int, dy: int, dx: int) -> int:
        """i番目のワームホールと接続している出口セルから(dy, dx)マス移動したところにあるセルを測定する"""
        raise NotImplementedError

    def write_answer(self, out_idx: list[int]) -> None:
        raise NotImplementedError

    def placement_step(self):
        self.write_cells(heatmap.P)

    def measurement_step(self) -> list[list[int]]:
        num_samples = self.featurizer.num_samples
        feature_size = self.featurizer.feature_size
        p_e = [[0.0 for j in range(featurizer.feature_size)] for i in range(self.N)]
        offset_yx = self.featurizer.get_offset_yx()
        for i in range(self.N):
            # i番目のワームホールの特徴量を作る
            for j in range(feature_size):
                dy, dx = offset_yx[j]
                pij_samples = [self.measure(i, dy, dx) for _ in range(num_samples)]
                pij_e = sum(pij_samples) / num_samples
                p_e[i][j] = pij_e
        return p_e

    def answer_step(self) -> list[int]:
        INF = 1e12
        out_idx = [0] * self.N
        """i番目のワームホールがout_idx[i]番目の出口セルに繋がっていると予測する"""

        feature_size = self.featurizer.feature_size
        offset_yx = self.featurizer.get_offset_yx()
        P = self.heatmap.P

        # i番目の出口セルを測定
        exit_cell_features: list[list[int]] = []
        for i in range(self.N):
            cy, cx = self.exit_cells[i]
            feature_ec = [0] * feature_size
            for j in range(feature_size):
                dy, dx = offset_yx[j]
                y, x = (cy + dy) % self.L, (cx + dx) % self.L
                feature_ec[j] = P[y][x]
            exit_cell_features.append(feature_ec)
        
        # i番目のワームホールを測定
        wormhole_features: list[list[float]] = []
        for i in range(self.N):
            feature_wh = self.p_e[i]
            wormhole_features.append(feature_wh)

        for i in range(self.N):
            min_loss = INF
            min_j = 0
            for j in range(self.N):
                loss = mse_loss(wormhole_features[i], exit_cell_features[j])
                if loss < min_loss:
                    min_loss = loss
                    min_j = j
            out_idx[i] = min_j
        self.write_answer(out_idx)
        return out_idx

    def run(self):
        self.placement_step()
        self.p_e = self.measurement_step()
        self.out_idx = self.answer_step()


class SimulatorResult(NamedTuple):
    score_e: float
    score_std: float

class Simulator(SolverBase):

    def __init__(self, L: int, N: int, S: int, exit_cells: list[ExitCell], heatmap: Heatmap, featurizer: Featurizer) -> None:
        super().__init__(L, N, S, exit_cells, heatmap, featurizer)
        self.out_idx_gt = None
        """i番目のワームホールがout_idx_gt[i]番目の出口セルに繋がっている"""
    
    def placement_step(self):
        pass

    
    def write_cells(self, P: list[list[int]]) -> None:
        pass

    def measure(self, i: int, dy: int, dx: int) -> int:
        j = self.out_idx_gt[i]
        exit_cell = self.exit_cells[j]
        P = self.heatmap.P
        p_gt = P[(exit_cell.y + dy) % self.L][(exit_cell.x + dx) % self.L]
        theta = random.gauss(0, self.S)
        p = max(0, min(1000, round(p_gt + theta)))
        return p
    
    def write_answer(self, out_idx: list[int]) -> None:
        pass
    
    def _get_score(self):
        w = sum([p != g for p, g in zip(self.out_idx, self.out_idx_gt)])
        measurement_cost = self.featurizer.get_measurement_cost()
        placement_cost = self.heatmap.get_placement_cost()
        score = math.ceil(10 ** 14 * 0.8 ** w / (measurement_cost + placement_cost + 10**5))
        return w, score
    
    def simulate(self, loop:int=10) -> SimulatorResult:
        scores = []
        for i in range(loop):
            self.out_idx_gt = random.sample(range(self.N), k=self.N)
            self.placement_step()
            self.p_e = self.measurement_step()
            self.out_idx = self.answer_step()
            w_i, score_i = self._get_score()
            scores.append(score_i)
        score_e = sum(scores) / loop
        score_std = math.sqrt(sum([(s - score_e)**2 for s in scores]) / loop)
        return SimulatorResult(score_e, score_std)

class Solver(SolverBase):

    def write_cells(self, P: list[list[int]]) -> None:
        write_cells(P)

    def measure(self, i: int, dy: int, dx: int) -> int:
        write_measure(i, dy, dx)
        return read_prediction()

    def write_answer(self, out_idx: list[int]) -> None:
        write_answer(out_idx)
    
    def answer_step(self) -> list[int]:
        INF = 1e14
        out_idx = [0] * self.N
        """i番目のワームホールがout_idx[i]番目の出口セルに繋がっていると予測する"""

        feature_size = self.featurizer.feature_size
        offset_yx = self.featurizer.get_offset_yx()
        P = self.heatmap.P

        # i番目の出口セルを直接測定
        exit_cell_features: list[list[int]] = []
        for i in range(self.N):
            cy, cx = self.exit_cells[i]
            feature_ec = [0] * feature_size
            for j in range(feature_size):
                dy, dx = offset_yx[j]
                y, x = (cy + dy) % self.L, (cx + dx) % self.L
                feature_ec[j] = P[y][x]
            exit_cell_features.append(feature_ec)
        
        # i番目のワームホールを測定
        wormhole_features: list[list[float]] = []
        for i in range(self.N):
            feature_wh = self.p_e[i]
            wormhole_features.append(feature_wh)

        for i in range(self.N):
            min_loss = INF
            min_j = 0
            for j in range(self.N):
                loss = mse_loss(wormhole_features[i], exit_cell_features[j])
                if loss < min_loss:
                    min_loss = loss
                    min_j = j
            out_idx[i] = min_j
        self.write_answer(out_idx)
        return out_idx

    def run(self):
        self.placement_step()
        self.p_e = self.measurement_step()
        self.out_idx = self.answer_step()


if __name__ == "__main__":
    L, N, S, exit_cells = read_parameters()
    best_simulator = None
    best_score_e = -1
    best_score_std = 1e12
    # (Heatmap, Featurizer) の組み合わせを時間の限り試す
    for t in range(20):
        heatmap = Heatmap.const_init(L, y_offset=0, x_offset=0)
        featurizer = Featurizer.rand_init(window_size=11, feature_size=5, num_samples=10)
        simulator = Simulator(L, N, S, exit_cells, heatmap, featurizer)
        result = simulator.simulate(loop=5)
        logger.info(f"simulator score={result.score_e, result.score_std}")
        if result.score_e > best_score_e:
            best_score_std = result.score_std
            best_score_e = result.score_e
            best_simulator = simulator
            logger.info(f"best heatmap: {simulator.heatmap.P[0]}")
    logger.info(f"best simulator score={best_score_e}")
    solver = Solver(L, N, S, exit_cells, best_simulator.heatmap, best_simulator.featurizer)
    solver.run()
    logger.info(f"best heatmap: {solver.heatmap.P[0]}")
