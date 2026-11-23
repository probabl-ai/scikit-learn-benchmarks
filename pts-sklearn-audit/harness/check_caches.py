"""Load each cached dataset with the PTS stack (numpy 1.24): caches pickled by numpy 2 fail."""
from sklearn import datasets as d

checks = {
    "20newsgroup_vectorized_py3.pkl": lambda: d.fetch_20newsgroups_vectorized(download_if_missing=False),
    "20news-bydate_py3.pkz": lambda: d.fetch_20newsgroups(download_if_missing=False),
    "RCV1": lambda: d.fetch_rcv1(download_if_missing=False),
    "kddcup99-py3": lambda: d.fetch_kddcup99(download_if_missing=False),
    "covertype": lambda: d.fetch_covtype(download_if_missing=False),
    "cal_housing_py3.pkz": lambda: d.fetch_california_housing(download_if_missing=False),
}
for name, f in checks.items():
    try:
        f()
        print("OK  ", name)
    except Exception as e:
        print("FAIL", name, type(e).__name__, str(e)[:100])
