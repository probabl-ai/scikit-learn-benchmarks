from _clustering import clustering_cases
from _implementations import implementations_for_pixi_env, with_implementations
from _linear import linear_cases
from _trees import apply_sklearnex_cpu_tree_variants, tree_cases


def generate_cases() -> list[dict]:
    implementations = implementations_for_pixi_env(workload="all_models")

    tree_workload = tree_cases("test")
    if any(impl["library"] == "sklearnex" for impl in implementations):
        tree_workload = apply_sklearnex_cpu_tree_variants(tree_workload)

    cases = []
    cases.extend(with_implementations(tree_workload, implementations))
    cases.extend(with_implementations(linear_cases("test"), implementations))
    cases.extend(with_implementations(clustering_cases("test"), implementations))
    return cases
