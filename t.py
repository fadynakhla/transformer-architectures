import numpy as np
import time

N = 4096 * 5
X = np.random.randn(N, N).astype(np.float32)
Y = np.random.randn(N, N).astype(np.float32)

flop = N * N * 2 * N
print(flop / 1e9)
st = time.monotonic()
X @ Y
t = time.monotonic() - st
print(flop / 1e9 / t)
