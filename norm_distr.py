import matplotlib.pyplot as plt
import numpy as np
import random

rng = np.random.default_rng()

samples = 1000
s = rng.power(5.0, 1)

s = [random.uniform(0.01, 1.0) ** 2 for x in range(10000)]

count, bins, ignored = plt.hist(s, bins=1000)

plt.show()

print(s)