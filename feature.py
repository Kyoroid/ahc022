from __future__ import annotations
from collections import Counter
import random
import logging
from typing import NamedTuple

logging.basicConfig(level="INFO")
logger = logging.getLogger()

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

class FeatureLocator:
    """特徴ベクトルの座標を良い感じに決める"""
    
    def __init__(self, L: int, N: int, S: int, exit_cells: list[ExitCell]):
        self.L = L
        self.N = N
        self.S = S
        self.exit_cells = exit_cells
    
    def build_random_points(self, feature_size: int = 5, window_size: int = 4):
        """特徴ベクトルの座標をランダムに生成する"""
        indices = list(random.sample(range(window_size * window_size), k=feature_size))
        offset_ys = []
        for idx in indices:
            p, q = divmod(idx, window_size)
            dy = p - window_size // 2
            dx = q - window_size // 2
            offset_ys.append((dy, dx))
        return offset_ys

    def build_optimized_points(self, feature_size: int = 5, window_size: int = 4, loop: int = 100):
        min_loss = 10**16
        min_offest_yx = None
        min_counter = None
        for i in range(loop):
            offset_yx = self.build_random_points(feature_size, window_size)
            counter = Counter()
            for cy, cx in exit_cells:
                for dy, dx in offset_yx:
                    y, x = (cy + dy) % L, (cx + dx) % L
                    idx = y * L + x
                    counter[idx] += 1
            frequency = counter.most_common()
            loss = sum([(v - 1)**3 for _, v in frequency])
            if loss < min_loss:
                min_loss = loss
                min_offest_yx = offset_yx
                min_counter = counter
        logger.info(min_offest_yx)
        logger.info([(idx, v) for idx, v in min_counter.most_common() if v > 1])
        return min_offest_yx
            
    

if __name__ == "__main__":
    L, N, S, exit_cells = read_parameters()
    locator = FeatureLocator(L, N, S, exit_cells)
    p = [[0 for x in range(L)] for y in range(L)]
    offset_yx = locator.build_optimized_points()
    for cy, cx in exit_cells:
        for dy, dx in offset_yx:
            y, x = (cy + dy) % L, (cx + dx) % L
            p[y][x] += 1
    for line in p:
        print(*line)
    print(-1, -1, -1)
    for i in range(N):
        print(0)
