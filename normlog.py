# import matplotlib.pyplot as plt
# import numpy as np
#
# mu, sigma = 1., 1.
# s = np.random.lognormal(mu, sigma, 1000)
# count, bins, ignored = plt.hist(s, 100, density=True, align='mean')
# x = np.linspace(low(bins), upp(bins), 10000)
# pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#        / (x * sigma * np.sqrt(2 * np.pi)))
# plt.plot(x, pdf, linewidth=2, color='r')
# plt.axis('tight')
# plt.show()
import random

import numpy as np
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

X1 = get_truncated_normal(mean=4, sd=1, low=1, upp=4)
# X2 = get_truncated_normal(mean=5.5, sd=1, low=1, upp=10)
# X3 = get_truncated_normal(mean=8, sd=1, low=1, upp=10)
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(3, sharex=True)
# ax[0].hist(X1.rvs(10000), density=True)
# # ax[1].hist(X2.rvs(10000), density=True)
# # ax[2].hist(X3.rvs(10000), density=True)
# plt.show()
#









import matplotlib.pyplot as plt
# fig, ax = plt.subplots(3, sharex=True)

# ax[0].hist(, density=True)
# ax[1].hist(X2.rvs(10000), density=True)
# ax[2].hist(X3.rvs(10000), density=True)
x = [get_asymmetric_norm(3,10,5) for x in range(10000)]
# print(x)
from collections import Counter
c = Counter(x)

plt.bar(c.keys(), c.values())
plt.show()