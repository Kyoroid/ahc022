from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import random
import logging
from typing import NamedTuple, List

logging.basicConfig(level="INFO")
logger = logging.getLogger()

code_length = 5
"""コード長"""
base = 2
"""n進数のコードを生成"""
samples = 10
"""生成するサンプルの数"""


class ExitCell(NamedTuple):
    y: int
    x: int


def read_parameters() -> tuple[int, int, int, list[ExitCell]]:
    L, N, S = map(int, input().strip().split())
    exit_cells = []
    for i in range(N):
        y, x = map(int, input().strip().split())
        exit_cells.append(ExitCell(y, x))
    return L, N, S, exit_cells


@dataclass
class Heatmap:
    P: list[list[int]]
    """ヒートマップの正確な温度"""

    def get_placement_cost(self) -> int:
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


@dataclass
class FeatureOffset:
    """特徴オフセット"""

    feature_size: int
    """特徴量の長さ"""
    offset_yx: list[tuple[int, int]]
    """特徴オフセット(dy, dx)のリスト"""

    def get_coord_yx(self, exit_cell: ExitCell) -> list[tuple[int, int]]:
        """出口セルに対応する特徴座標リストを絶対座標として返す"""
        cy, cx = exit_cell.y, exit_cell.x
        return [((cy + dy) % L, (cx + dx) % L) for (dy, dx) in self.offset_yx]

    def get_measurement_cost(self, repeat_measurement: int) -> int:
        """測定コストを返す"""
        measurement_cost = 0
        for dy, dx in self.offset_yx:
            measurement_cost += 100 * (10 + abs(dy) + abs(dx)) * repeat_measurement
        return measurement_cost


class FeatureOffsetInitializer:
    """特徴ベクトルの座標を適当に決める"""

    def __init__(
        self,
        L: int,
        N: int,
        S: int,
        exit_cells: list[ExitCell],
        feature_size: int = 5,
        window_size: int = 4,
    ):
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
        self.feature_size = feature_size
        self.window_size = window_size

    def random_feature_offset(self) -> FeatureOffset:
        """特徴オフセットをランダムに生成する"""
        indices = list(
            random.sample(
                range(self.window_size * self.window_size), k=self.feature_size
            )
        )
        offset_yx = []
        for idx in indices:
            p, q = divmod(idx, self.window_size)
            dy = p - self.window_size // 2
            dx = q - self.window_size // 2
            offset_yx.append((dy, dx))
        return FeatureOffset(self.feature_size, offset_yx)


class HeatmapBuilder:
    """温度をいい感じに決める"""

    def __init__(self, L: int, N: int, S: int, exit_cells: list[ExitCell]) -> None:
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells

    def get_height(self, y: int, x: int) -> int:
        return min(2, x // 16) + min(2, y // 16)

    def _build_quantized_heatmap(
        self, feature_offset: FeatureOffset
    ) -> tuple[Heatmap, Counter]:
        encoded_p = [[-1 for x in range(self.L)] for y in range(self.L)]
        code_dict = Counter()
        for i in range(self.N):
            code = []
            coord_yx = feature_offset.get_coord_yx(self.exit_cells[i])
            for j in range(feature_offset.feature_size):
                y, x = coord_yx[j]
                if encoded_p[y][x] == -1:
                    encoded_p[y][x] = random.randint(0, 1) + self.get_height(y, x)
                code.append(str(encoded_p[y][x]))
            key = "".join(code)
            code_dict[key] += 1
        for y in range(self.L):
            for x in range(self.L):
                if encoded_p[y][x] == -1:
                    encoded_p[y][x] = 0
        return encoded_p, code_dict

    def build_optimized_heatmap(
        self, loop: int = 1000
    ) -> tuple[Heatmap, FeatureOffset]:
        min_score = self.N + 1
        min_encoded_P = None
        min_code_dict = None
        min_feature_offset = None
        feature_initializer = FeatureOffsetInitializer(
            self.L, self.N, self.S, self.exit_cells
        )
        for i in range(loop):
            feature_offset = feature_initializer.random_feature_offset()
            encoded_P, code_dict = self._build_quantized_heatmap(feature_offset)
            score = self.N - len(code_dict.keys())
            if score < min_score:
                min_score = score
                min_encoded_P = encoded_P
                min_code_dict = code_dict
                min_feature_offset = feature_offset
        logger.info(min_code_dict)
        # temperature scaling
        scale = S
        for y in range(self.L):
            for x in range(self.L):
                min_encoded_P[y][x] = min(1000, min_encoded_P[y][x] * scale)
        # Fill
        heatmap = Heatmap(min_encoded_P)
        return heatmap, min_feature_offset


if __name__ == "__main__":
    L, N, S, exit_cells = read_parameters()
    heatmap_builder = HeatmapBuilder(L, N, S, exit_cells)
    heatmap, feature_offset = heatmap_builder.build_optimized_heatmap(loop=100)
    for line in heatmap.P:
        print(*line)
    print(-1, -1, -1)
    for i in range(N):
        print(0)
