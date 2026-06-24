from fof_system.data import get_provider
from fof_system.engine.robustness import run_robustness


def test_robustness_runs_multiple_walk_forward_scenarios():
    provider = get_provider("mock")
    result = run_robustness(provider, list(provider.truth), "2019-01-01", "2024-09-30")
    assert set(result["scenario"]) == {"base", "high_cost", "conservative", "loose"}
    assert result["ann_excess"].notna().all()
    assert result["total_transaction_cost"].gt(0).all()
