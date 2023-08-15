from __future__ import annotations
import sys
from typing import NamedTuple
from collections import defaultdict, Counter
import itertools
import pprint
import random
import math
import logging
import numpy as np
from scipy.interpolate import LinearNDInterpolator

logging.basicConfig(level="INFO")
logger = logging.getLogger()


class ExitCell(NamedTuple):
    y: int
    x: int


class FieldInitializer:
    def __init__(
        self,
        L: int,
        N: int,
        S: int,
        exit_cells: list[ExitCell],
        p_min: int = 0,
        p_max: int = 1000,
    ) -> None:
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
        self.p_min = p_min
        self.p_max = p_max

    def _minmax_scale(self, v, v_min, v_max):
        std = (v - v_min) / (v_max - v_min)
        return std * (self.p_max - self.p_min) + self.p_min

    def get_const_field(self, const: int) -> list[list[int]]:
        return [[const for y in range(self.L)] for x in range(self.L)]

    def get_cosine_field(self, y_offset: int = 0, x_offset: int = 0) -> list[list[int]]:
        p = self.get_const_field(0)
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


class Featurizer:
    def __init__(
        self,
        L: int,
        N: int,
        S: int,
        exit_cells: list[ExitCell],
        p_max: int,
        code_radius: int,
        code_length: int,
        code_offset=None,
        code=None,
    ) -> None:
        assert code_length >= 8
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
        self.p_max = p_max
        self.threshold = p_max / 2
        self.code_radius = code_radius
        self.code_length = code_length
        self._offset_yx = code_offset
        self._code = code

    def init_code_offset(self) -> None:
        """2Dコードを盤面に埋め込む際のデータ点の座標をランダムに決める"""
        r = self.code_radius
        rr = r * 2 + 1
        assert self.code_length <= rr * rr, "半径が小さすぎてコードを埋め込めません"
        indice = random.sample(range(rr * rr), k=self.code_length)
        offset_yx = []
        for idx in indice:
            dy, dx = divmod(idx, rr)
            offset_yx.append((dy - r, dx - r))
        self._offset_yx = offset_yx

    @property
    def code_offset(self) -> list[tuple[int, int]]:
        return self._offset_yx

    def init_code(self) -> None:
        code_list = list(itertools.product([0, 1], repeat=self.code_length))
        self._code = code_list[:self.N]

    @property
    def code(self):
        return self._code

    def list_points(self, cy: int, cx: int) -> list[tuple[int, int]]:
        """気温の測定位置の絶対座標を取得する"""
        return [
            ((cy + dy) % self.L, (cx + dx) % self.L) for (dy, dx) in self._offset_yx
        ]

    def value2code(self, value: list[int]) -> list[int]:
        """気温をコードに変換する"""
        return [1 if v > self.threshold else 0 for v in value]

    def value2str(self, value: list[int]) -> str:
        """気温をコード文字列に変換する"""
        code = self.value2code(value)
        return "".join(map(str, code))


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
    def __init__(self, L: int, N: int, S: int, exit_cells: list[ExitCell]) -> None:
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
        self.P = None
        self.p_min = 0
        self.p_max = min(1000, round(self.S*4))
        self.wormholes_p: list[int] = None
        self.exit_cell_features: list[list[int]] = []
        self.wormhole_features: list[list[int]] = []
        self.code_radius = 1
        self.code_length = 8
        self.featurizer: Featurizer = None

    def write_cells(self, P: list[list[int]]) -> None:
        raise NotImplementedError

    def read_parameters(self) -> tuple[int, int, int, list[ExitCell]]:
        raise NotImplementedError

    def measure(self, i: int, y: int, x: int) -> int:
        raise NotImplementedError

    def write_answer(self, wormholes: list[int]) -> None:
        raise NotImplementedError

    def placement_step(self):
        field = FieldInitializer(
            self.L, self.N, self.S, self.exit_cells, p_max=self.p_max
        )
        max_code_quality = -1000000
        best_featurizer = None
        best_P = None
        best_exit_cell_features = None
        n_code_gen = 1000
        for t in range(n_code_gen):
            featurizer = Featurizer(
                self.L,
                self.N,
                self.S,
                self.exit_cells,
                p_max=self.p_max,
                code_radius=self.code_radius,
                code_length=self.code_length,
            )
            P = field.get_const_field(round(featurizer.threshold))
            featurizer.init_code_offset()
            self.exit_cell_features = []
            # コードを生成
            featurizer.init_code()
            code_list = featurizer.code
            for i in range(self.N):
                exit_cell = self.exit_cells[i]
                code = code_list[i]
                point_yx = featurizer.list_points(exit_cell.y, exit_cell.x)
                for j in range(self.code_length):
                    y, x = point_yx[j]
                    P[y][x] = self.p_max if code[j] == 1 else self.p_min
            # コードを読み取ってみる
            code_quality = 0
            exit_cell_features = []
            for i in range(self.N):
                exit_cell = self.exit_cells[i]
                code_expect = code_list[i]
                point_yx = featurizer.list_points(exit_cell.y, exit_cell.x)
                value = [0] * self.code_length
                for j in range(self.code_length):
                    y, x = point_yx[j]
                    value[j] = P[y][x]
                code_actual = featurizer.value2code(value)
                exit_cell_features.append(code_actual)
                for j in range(self.code_length):
                    if code_actual[j] != code_expect[j]:
                        code_quality -= 1
            if code_quality > max_code_quality:
                max_code_quality = code_quality
                best_featurizer = featurizer
                best_exit_cell_features = exit_cell_features
                best_P = P
            logger.info(f"Featurizer {t}: code_quality={code_quality}")
        YX = set()
        for cy, cx in self.exit_cells:
            for y, x in best_featurizer.list_points(cy, cx):
                for i in range(3):
                    for j in range(3):
                        YX.add((y + self.L*i, x + self.L*j))
        YX = list(YX)
        Z = []
        for y, x in YX:
            Z.append(best_P[y%self.L][x%self.L])
        
        interp = LinearNDInterpolator(YX, Z)
        P = field.get_const_field(round(best_featurizer.threshold))
        for y in range(self.L):
            for x in range(self.L):
                z = interp(y+self.L, x+self.L).item()
                P[y][x] = round(z)
        self.featurizer = best_featurizer
        self.exit_cell_features = best_exit_cell_features
        self.P = P
        self.write_cells(self.P)

    def measurement_step(self):
        # n_samples回サンプリング
        n_samples = 10
        # カーネルサイズ
        for i in range(self.N):
            value = [0] * self.code_length
            for j in range(self.code_length):
                dy, dx = self.featurizer.code_offset[j]
                value[j] = (
                    sum([self.measure(i, dy, dx) for _ in range(n_samples)]) / n_samples
                )
            code = self.featurizer.value2code(value)
            self.wormhole_features.append(code)

    def answer_step(self):
        INF = 1e12
        wormholes = [0] * self.N
        for i in range(self.N):
            min_loss = INF
            min_j = 0
            for j in range(self.N):
                feature_pred = self.wormhole_features[i]
                feature_expect = self.exit_cell_features[j]
                loss = mse_loss(feature_pred, feature_expect)
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
    def measure(self, i: int, y: int, x: int) -> int:
        exit_cell = self.exit_cells[i]
        p = self.P[(exit_cell.y + y) % self.L][(exit_cell.x + x) % self.L]
        theta = random.gauss(0, self.S)
        return max(0, min(1000, round(p) + theta))


class Solver(SolverBase):
    def read_parameters(self) -> tuple[int, int, int, list[ExitCell]]:
        return read_parameters()

    def write_cells(self, P: list[list[int]]) -> None:
        write_cells(P)

    def measure(self, i: int, y: int, x: int) -> int:
        write_measure(i, y, x)
        return read_prediction()

    def write_answer(self, wormholes: list[int]) -> None:
        write_answer(wormholes)


if __name__ == "__main__":
    L, N, S, exit_cells = read_parameters()
    Solver(L, N, S, exit_cells).run()
