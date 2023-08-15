import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("i", type=int)
    return parser.parse_args()

def main(i: int):
    max_trial = 10
    max_samples = 100
    s = i * i
    for t in range(max_trial):
        x = np.arange(max_samples)
        p_min = np.random.normal(0, s, size=(max_samples, ))
        p_max = np.random.normal(s, s, size=(max_samples, ))
        p_min_e = np.zeros_like(p_min)
        p_max_e = np.zeros_like(p_max)
        for j in range(max_samples):
            p_min_e[j] = np.mean(p_min[:j+1])
            p_max_e[j] = np.mean(p_max[:j+1])
        plt.plot(x, p_min_e)
        plt.plot(x, p_max_e)
    plt.title(f"i={i}")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
