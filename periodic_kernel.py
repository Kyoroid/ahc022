import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


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

L = 50
wgen = WaveGenerator(L)
z1 = wgen.get_wave()
z2 = wgen.get_wave()
p = np.zeros((L, L), dtype=np.int)
for i in range(L):
    for j in range(L):
        p[i, j] = z1[i] + z2[j]

plt.imshow(p)
plt.colorbar()
plt.show()