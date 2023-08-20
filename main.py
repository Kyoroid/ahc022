import sys
import argparse
from abc import ABC
from typing import NamedTuple, List, Union, Tuple
import random
from pprint import pformat
import math
import logging
import time
from collections import Counter

logging.basicConfig(level="ERROR")
logger = logging.getLogger()

SEED0_GT = [79, 90, 11, 72, 16, 74, 69, 24, 58, 48, 23, 15, 70, 80, 57, 51, 22, 6, 50, 37, 45, 7, 12, 61, 29, 94, 89, 87, 5, 43, 81, 26, 8, 56, 10, 0, 31, 44, 9, 21, 68, 93, 36, 40, 62, 65, 91, 85, 86, 49, 13, 71, 27, 84, 25, 35, 47, 28, 42, 75, 17, 88, 67, 64, 78, 83, 46, 77, 32, 4, 60, 19, 53, 14, 18, 20, 54, 41, 2, 66, 34, 59, 76, 63, 55, 38, 52, 92, 39, 3, 1, 33, 30, 82, 73]

class ExitCell(NamedTuple):
    y: int
    x: int


def mse_loss(pred: List[float], target: List[int]) -> float:
    n = len(pred)
    return sum([(pred[i] - target[i]) ** 2 for i in range(n)]) / n


def read_parameters() -> Tuple[int, int, int, List[ExitCell]]:
    L, N, S = map(int, input().strip().split())
    exit_cells = []
    for i in range(N):
        y, x = map(int, input().strip().split())
        exit_cells.append(ExitCell(y, x))
    return L, N, S, exit_cells


def write_p(P: List[List[int]], PMIN=0, PMAX=1000):
    for line in P:
        print(*line)
    sys.stdout.flush()


def write_measure(i: int, y: int, x: int):
    print(i, y, x, flush=True)


def read_prediction() -> int:
    p = int(input().strip())
    return p


def write_answer(wormholes: List[int]):
    print(-1, -1, -1)
    for w in wormholes:
        print(w)
    sys.stdout.flush()


class Heatmap(NamedTuple):
    P: List[List[int]]
    """ヒートマップの正確な温度"""
    min_p: int
    """温度の最小値"""
    max_p: int
    """温度の最大値"""

    def get_placement_cost(self, L: int) -> int:
        """配置コストを返す"""
        placement_cost = 0
        for i in range(L):
            i1 = (i + 1) % L
            for j in range(L):
                j1 = (j + 1) % L
                placement_cost += (self.P[i][j] - self.P[i][j1]) ** 2 + (
                    self.P[i][j] - self.P[i1][j]
                ) ** 2
        return placement_cost


class HeatmapEncoder:
        
    def __init__(self, L: int, N: int, S: int, exit_cells: List[ExitCell], min_p: int=0, max_p: int=1000) -> None:
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
        self.min_p = min_p
        self.max_p = max_p
        self.step = S
        zero_value = [[0 for x in range(L)] for y in range(L)]
        threshold = list(range(min_p, max(min_p+1, max_p + 1 - self.step), self.step))
        self.max_digit = len(threshold)
        # TODO: ここを良い感じにする
        for y in range(L):
            for x in range(L):
                zero_value[y][x] = random.randint(0, self.max_digit - 1)
        self._zero_value = zero_value

    def encode(self, y: int, x: int, bit: int) -> Tuple[int, int]:
        """ビット(0/1)を(量子化値, 量子化温度)に変換する"""
        q_value = bit + self._zero_value[y][x]
        return max(self.min_p, min(self.max_p, q_value * self.step))
    
    def decode(self, p: Union[int, float]) -> int:
        # """温度を特徴量に変換する"""
        # q_value = max(0, min(self.max_digit, round(p) // self.step))
        q_value = p
        return q_value


class FeatureOffset(NamedTuple):
    """特徴オフセット"""

    feature_size: int
    """特徴量の長さ"""
    offset_yx: List[Tuple[int, int]]
    """特徴オフセット(dy, dx)のリスト"""

    def get_coord_yx(self, L: int, exit_cell: ExitCell) -> List[Tuple[int, int]]:
        """出口セルに対応する特徴座標リストを絶対座標として返す"""
        cy, cx = exit_cell.y, exit_cell.x
        return [((cy + dy) % L, (cx + dx) % L) for (dy, dx) in self.offset_yx]

    def get_measurement_cost(self, N: int, repeat_measurement: int) -> int:
        """測定コストを返す"""
        measurement_cost = 0
        for dy, dx in self.offset_yx:
            measurement_cost += 100 * (10 + abs(dy) + abs(dx)) * repeat_measurement
        return measurement_cost * N

    def get_measurement_count(self, N: int, repeat_measurement: int) -> int:
        """測定回数を返す"""
        return N * self.feature_size * repeat_measurement


class FeatureOffsetInitializer:
    """特徴ベクトルの座標を適当に決める"""

    def __init__(
        self,
        L: int,
        N: int,
        S: int,
        exit_cells: List[ExitCell],
        feature_size: int,
        feature_radius: int,
    ):
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
        self.feature_size = feature_size
        self.feature_radius = feature_radius
        points = []
        for dy in range(-feature_radius, feature_radius+1):
            for dx in range(-feature_radius, feature_radius+1):
                #euclid_distance = abs(dy) + abs(dx)
                #if euclid_distance <= feature_radius:
                #    points.append((dy, dx))
                points.append((dy, dx))
        self._points = points

    def random_feature_offset(self) -> FeatureOffset:
        """特徴オフセットをランダムに生成する"""
        offset_yx = random.sample(self._points, self.feature_size)
        return FeatureOffset(self.feature_size, offset_yx)


class HeatmapBuilder:
    """温度を良い感じに決める"""

    def __init__(self, L: int, N: int, S: int, exit_cells: List[ExitCell], feature_initializer: FeatureOffsetInitializer, min_p: int = 0, max_p: int = 1000) -> None:
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
        self.min_p = min_p
        self.max_p = max_p
        self.feature_initializer = feature_initializer

    def _build_heatmap_seed(
        self, heatmap_encoder: HeatmapEncoder, feature_offset: FeatureOffset
    ) -> Tuple[List[List[int]], Counter]:
        encoded_p = [[-1 for x in range(self.L)] for y in range(self.L)]
        feature = [[-1 for j in range(feature_offset.feature_size)] for i in range(self.N)]

        code_dict = Counter()
        for i in range(self.N):
            code = []
            coord_yx = feature_offset.get_coord_yx(self.L, self.exit_cells[i])
            for j in range(feature_offset.feature_size):
                y, x = coord_yx[j]
                if encoded_p[y][x] == -1:
                    p = heatmap_encoder.encode(y, x, random.randint(0, 1))
                    encoded_p[y][x] = p
                code.append(f"{encoded_p[y][x]:04d}")
                feature[i][j] = encoded_p[y][x]
            key = "".join(code)
            code_dict[key] += 1
        return feature, code_dict

    def build_optimized_heatmap(
        self, loop: int = 1000,
    ) -> Tuple[FeatureOffset, Heatmap, HeatmapEncoder]:
        min_score = self.N + 1
        min_code_dict = None
        min_feature_offset = None
        min_feature = None
        min_heatmap_encoder = None
        heatmap_encoder = HeatmapEncoder(self.L, self.N, self.S, self.exit_cells, self.min_p, self.max_p)
        for i in range(loop):
            feature_offset = self.feature_initializer.random_feature_offset()
            feature, code_dict = self._build_heatmap_seed(heatmap_encoder, feature_offset)
            score = self.N - len(code_dict.keys())
            if score < min_score:
                min_score = score
                min_feature = feature
                min_code_dict = code_dict
                min_feature_offset = feature_offset
                min_heatmap_encoder = heatmap_encoder
        points = set()
        for i in range(self.N):
            coord_yx = min_feature_offset.get_coord_yx(self.L, self.exit_cells[i])
            for j in range(min_feature_offset.feature_size):
                y, x = coord_yx[j]
                z = min_feature[i][j]
                points.add((y, x, z))
        fill_value = (self.max_p + self.min_p) // 2
        P = [[fill_value for x in range(self.L)] for y in range(self.L)]
        for i in range(self.N):
            coord_yx = min_feature_offset.get_coord_yx(self.L, self.exit_cells[i])
            for j in range(min_feature_offset.feature_size):
                y, x = coord_yx[j]
                P[y][x] = min_feature[i][j]
        heatmap = Heatmap(P, self.min_p, self.max_p)
        return min_feature_offset, heatmap, min_heatmap_encoder


class HeatmapOptimizer:
    """温度を良い感じに調整する"""

    def __init__(self, L: int, N: int, S: int, exit_cells: List[ExitCell]) -> None:
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
    
    def optimize(self, feature_offset: FeatureOffset, heatmap: Heatmap, loop: int=100000) -> None:
        points = set()
        for i in range(self.N):
            for y, x in feature_offset.get_coord_yx(self.L, self.exit_cells[i]):
                points.add((y, x))
        for t in range(loop):
            y = random.randint(0, self.L-1)
            x = random.randint(0, self.L-1)
            if (y, x) in points:
                continue
            z = 0
            for dy, dx in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                ny, nx = (y + dy) % self.L, (x + dx) % self.L
                z += heatmap.P[ny][nx]
            heatmap.P[y][x] = z // 4


class SolverBase(ABC):

    def __init__(self, L: int, N: int, S: int, exit_cells: List[ExitCell], feature_offset: FeatureOffset, heatmap: Heatmap, heatmap_encoder: HeatmapEncoder, repeat_measurement: int) -> None:
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
        self.feature_offset = feature_offset
        self.heatmap = heatmap
        self.heatmap_encoder = heatmap_encoder
        self.max_measurement = 10000
        self.repeat_measurement = repeat_measurement
        self.feature_e = None
        """feature_e[i][j]: i番目のワームホールの先にある測定対象のセルのうち、j番目のセルの測定値"""
        self.out_idx_e = None
        """out_idx[i]: i番目のワームホールが繋がっている出口セルの予測値"""

    def write_cells(self, P: List[List[int]]) -> None:
        raise NotImplementedError

    def measure(self, i: int, dy: int, dx: int) -> int:
        """i番目のワームホールと接続している出口セルから(dy, dx)マス移動したところにあるセルを測定する"""
        raise NotImplementedError

    def write_answer(self, out_idx: List[int]) -> None:
        raise NotImplementedError

    def placement_step(self):
        self.write_cells(self.heatmap.P)

    def measurement_step(self) -> List[List[float]]:
        feature_e = [[0.0 for j in range(self.feature_offset.feature_size)] for i in range(self.N)]
        for i in range(self.N):
            # i番目のワームホールの特徴量を作る
            offset_yx = self.feature_offset.offset_yx
            for j in range(self.feature_offset.feature_size):
                dy, dx = offset_yx[j]
                p_e = sum([self.measure(i, dy, dx) for _ in range(self.repeat_measurement)]) / self.repeat_measurement
                feature_e[i][j] = self.heatmap_encoder.decode(p_e)
        return feature_e

    def answer_step(self, feature_pred: List[List[float]]) -> List[int]:
        INF = 1e12
        out_idx_e = [0] * self.N
        """i番目のワームホールがout_idx[i]番目の出口セルに繋がっていると予測する"""

        # 特徴量のground truthを生成
        feature_gt: List[List[int]] = []
        for i in range(self.N):
            coord_yx = self.feature_offset.get_coord_yx(self.L, self.exit_cells[i])
            feature_gt_i = [0] * self.feature_offset.feature_size
            for j in range(self.feature_offset.feature_size):
                y, x = coord_yx[j]
                p = self.heatmap.P[y][x]
                feature_gt_i[j] = self.heatmap_encoder.decode(p)
            feature_gt.append(feature_gt_i)

        for i in range(self.N):
            min_loss = INF
            min_j = 0
            for j in range(self.N):
                loss = mse_loss(feature_pred[i], feature_gt[j])
                if loss < min_loss:
                    min_loss = loss
                    min_j = j
            out_idx_e[i] = min_j
        self.write_answer(out_idx_e)
        return out_idx_e

    def run(self):
        self.placement_step()
        feature_e = self.measurement_step()
        out_idx = self.answer_step(feature_e)


class SolutionInfo(NamedTuple):

    score: int
    num_wrong_answers: int
    min_p: int
    max_p: int
    placement_cost: int
    measurement_cost: int
    measurement_count: int


class AvgSolutionInfo(NamedTuple):

    score_avg: float
    score_std: float
    num_wrong_answers_avg: float
    num_wrong_answers_std: float
    min_p: int
    max_p: int
    placement_cost: int
    measurement_cost: int
    measurement_count: int


class Simulator(SolverBase):

    def __init__(self, L: int, N: int, S: int, exit_cells: List[ExitCell], feature_offset: FeatureOffset, heatmap: Heatmap, heatmap_encoder: HeatmapEncoder, repeat_measurement: int) -> None:
        super().__init__(L, N, S, exit_cells, feature_offset, heatmap, heatmap_encoder, repeat_measurement)
        self.out_idx_gt = None
        """i番目のワームホールがout_idx_gt[i]番目の出口セルに繋がっている"""
    
    def placement_step(self):
        pass

    
    def write_cells(self, P: List[List[int]]) -> None:
        pass

    def measure(self, i: int, dy: int, dx: int) -> int:
        j = self.out_idx_gt[i]
        exit_cell = self.exit_cells[j]
        P = self.heatmap.P
        p_gt = P[(exit_cell.y + dy) % self.L][(exit_cell.x + dx) % self.L]
        theta = random.gauss(0, self.S)
        p = max(0, min(1000, round(p_gt + theta)))
        return p
    
    def write_answer(self, out_idx: List[int]) -> None:
        pass
    
    def simulate_once(self) -> SolutionInfo:
        self.out_idx_gt = random.sample(range(self.N), k=self.N)
        self.placement_step()
        feature_pred = self.measurement_step()
        out_idx_pred = self.answer_step(feature_pred)
        num_wrong_answers = sum([p != g for p, g in zip(out_idx_pred, self.out_idx_gt)])
        measurement_cost = self.feature_offset.get_measurement_cost(self.N, repeat_measurement=self.repeat_measurement)
        measurement_count = self.feature_offset.get_measurement_count(self.N, repeat_measurement=self.repeat_measurement)
        placement_cost = self.heatmap.get_placement_cost(self.L)
        score = math.ceil(10 ** 14 * 0.8 ** num_wrong_answers / (measurement_cost + placement_cost + 10**5))
        return SolutionInfo(
            score = score,
            num_wrong_answers=num_wrong_answers,
            min_p=self.heatmap.min_p,
            max_p=self.heatmap.max_p,
            placement_cost=placement_cost,
            measurement_cost=measurement_cost,
            measurement_count=measurement_count,
        )

    def simulate(self, loop: int) -> AvgSolutionInfo:
        min_p = self.heatmap.min_p
        max_p = self.heatmap.max_p
        measurement_cost = self.feature_offset.get_measurement_cost(self.N, repeat_measurement=self.repeat_measurement)
        measurement_count = self.feature_offset.get_measurement_count(self.N, repeat_measurement=self.repeat_measurement)
        placement_cost = self.heatmap.get_placement_cost(self.L)
        score_list = []
        num_wrong_answers_list = []
        for t in range(loop):
            self.out_idx_gt = random.sample(range(self.N), k=self.N)
            self.placement_step()
            feature_pred = self.measurement_step()
            out_idx_pred = self.answer_step(feature_pred)
            num_wrong_answers = sum([p != g for p, g in zip(out_idx_pred, self.out_idx_gt)])
            score = math.ceil(10 ** 14 * 0.8 ** num_wrong_answers / (measurement_cost + placement_cost + 10**5))
            score_list.append(score)
            num_wrong_answers_list.append(num_wrong_answers)
        score_avg=  sum(score_list) / loop
        num_wrong_answers_avg = sum(num_wrong_answers_list) / loop
        score_std = math.sqrt(sum([(v - score_avg)**2 for v in score_list]) / loop)
        num_wrong_answers_std = math.sqrt(sum([(v - num_wrong_answers_avg)**2 for v in num_wrong_answers_list]) / loop)
        return AvgSolutionInfo(
            score_avg =score_avg,
            score_std=score_std,
            num_wrong_answers_avg=num_wrong_answers_avg,
            num_wrong_answers_std=num_wrong_answers_std,
            min_p=min_p,
            max_p=max_p,
            placement_cost=placement_cost,
            measurement_cost=measurement_cost,
            measurement_count=measurement_count,
        )


class Solver(SolverBase):

    def write_cells(self, P: List[List[int]]) -> None:
        write_p(P)

    def measure(self, i: int, dy: int, dx: int) -> int:
        write_measure(i, dy, dx)
        return read_prediction()

    def write_answer(self, out_idx: List[int]) -> None:
        write_answer(out_idx)


def main(feature_sizes: List[int], time_threshold: float=3.5, seed: int = 0):
    random.seed(seed)
    start = time.time()
    L, N, S, exit_cells = read_parameters()
    best_feature_offset, best_heatmap, best_heatmap_encoder = None, None, None
    best_solution_info = None
    best_repeat_measurement = 10
    best_score = -1
    repeat_measurement_dict = [10, 9, 4, 3, 2, 2, 2, 2, 2, 2]
    for feature_size in feature_sizes:
        feature_initializer = FeatureOffsetInitializer(
                L, N, S, exit_cells, feature_size=feature_size, feature_radius=(feature_size + 1) // 2
            )
        prev_max_p = -1
        for factor in range(1, (N.bit_length()+1)):
            end = time.time()
            if end - start > time_threshold:
                break
            max_p = min(1000, S * factor)
            if prev_max_p == max_p:
                continue
            prev_max_p = max_p
            repeat_measurement = repeat_measurement_dict[factor]
            heatmap_builder = HeatmapBuilder(L, N, S, exit_cells, max_p = max_p, feature_initializer=feature_initializer)
            feature_offset, heatmap, heatmap_encoder = heatmap_builder.build_optimized_heatmap(loop=10)
            # 測定回数上限を超える特徴は使わない
            measurement_count = feature_offset.get_measurement_count(N, repeat_measurement=repeat_measurement)
            if measurement_count > 10000:
                continue
            simulator = Simulator(L, N, S, exit_cells, feature_offset, heatmap, heatmap_encoder, repeat_measurement=repeat_measurement)
            solution_info = simulator.simulate(loop=10)
            if solution_info.score_avg > best_score:
                best_score = solution_info.score_avg
                best_solution_info = solution_info
                best_feature_offset = feature_offset
                best_heatmap = heatmap
                best_heatmap_encoder = heatmap_encoder
                best_repeat_measurement = repeat_measurement
    heatmap_optimizer = HeatmapOptimizer(L, N, S, exit_cells)
    heatmap_optimizer.optimize(best_feature_offset, best_heatmap)
    logger.info(f"Best Simulator Result")
    logger.info(f"\tScore: {best_solution_info.score_avg} ± {best_solution_info.score_std}")
    logger.info(f"\tNumber of wrong answers: {best_solution_info.num_wrong_answers_avg} ± {best_solution_info.num_wrong_answers_std}")
    logger.info(f"\tmin_p: {best_solution_info.min_p}")
    logger.info(f"\tmax_p: {best_solution_info.max_p}")
    logger.info(f"\tPlacement cost: {best_solution_info.placement_cost}")
    logger.info(f"\tMeasurement cost: {best_solution_info.measurement_cost}")
    logger.info(f"\tMeasurement count: {best_solution_info.measurement_count}")
    solver = Solver(L, N, S, exit_cells, best_feature_offset, best_heatmap, best_heatmap_encoder, repeat_measurement=best_repeat_measurement)
    solver.run()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 7, 9, 11, 13, 15])
    parser.add_argument("--time_threshold", type=float, default=3.7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(**vars(args))