from telco_multiagent.data.corruptions import (
    CORRUPTION_TO_DEFECT,
    generate_corruptions,
)
from telco_multiagent.data.synthetic import build_synthetic_dataset


def test_corruption_labels_match_recipe():
    ds = build_synthetic_dataset()
    corr = generate_corruptions(ds, seed=1, per_sample=6)
    for rec in corr["train"] + corr["eval"]:
        assert rec.defect_labels == [CORRUPTION_TO_DEFECT[rec.corruption_type]]
        assert rec.is_corrupt is True


def test_train_eval_blueprints_disjoint():
    ds = build_synthetic_dataset()
    corr = generate_corruptions(ds, seed=1, per_sample=6)
    train_bps = {r.blueprint_id for r in corr["train"]}
    eval_bps = {r.blueprint_id for r in corr["eval"]}
    assert train_bps.isdisjoint(eval_bps)


def test_positives_are_clean():
    ds = build_synthetic_dataset()
    corr = generate_corruptions(ds, seed=1)
    for rec in corr["positives"]:
        assert rec.is_corrupt is False
        assert rec.corrupted_sequence == rec.gold_sequence


def test_recipe_hash_versioned():
    ds = build_synthetic_dataset()
    c1 = generate_corruptions(ds, seed=1)
    c2 = generate_corruptions(ds, seed=2)
    h1 = c1["positives"][0].provenance["recipe"]
    h2 = c2["positives"][0].provenance["recipe"]
    assert h1 != h2
