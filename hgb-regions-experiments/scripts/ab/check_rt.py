import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_info
HistGradientBoostingRegressor(max_iter=2).fit(np.random.rand(1000, 5), np.random.rand(1000))
print([(p["internal_api"], p["filepath"].split("/envs/")[-1]) for p in threadpool_info() if p["user_api"] == "openmp"])
