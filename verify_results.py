"""Audit saved experiment results without fitting models or downloading weights.

Usage: python verify_results.py [--raw /path/to/upworthy.csv]
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE_SHA256 = "8368313b060f4015a0c6fb34e6d788163cee29554144aba9390b14922eb9d8ce"
MODELS = ["zero_lift", "training_mean", "tfidf_ridge", "minilm_ridge"]


def check_raw(raw_path, combined):
    """Independently reconstruct selection and content groups from the raw CSV."""
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == SOURCE_SHA256
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    raw = raw.rename(columns={raw.columns[0]: "row_id"})
    assert len(raw) == 22666 and raw.clickability_test_id.nunique() == 4873
    date = raw.created_at.str[:10]
    excluded = set(raw.loc[date.between("2013-06-25", "2014-01-10"), "clickability_test_id"])
    clean = raw.loc[~raw.clickability_test_id.isin(excluded)].copy()
    assert len(excluded) == 1050 and len(clean) == 18388
    assert clean.clickability_test_id.nunique() == 3823

    def tidy(value):
        return re.sub(r"\s+", " ", value, flags=re.ASCII).strip(" ")

    usable = clean.copy()
    for col in ["headline", "excerpt", "eyecatcher_id"]:
        usable[col] = usable[col].map(tidy)
    usable = usable.loc[(usable.headline != "") & (usable.eyecatcher_id != "")]
    winners = {}
    group_count = variant_count = 0
    for (test_id, image_id, excerpt), group in usable.groupby(
        ["clickability_test_id", "eyecatcher_id", "excerpt"], sort=True
    ):
        if group.headline.nunique() < 2:
            continue
        group_count += 1
        variant_count += len(group)
        rows = sorted(group.to_dict("records"), key=lambda r: (r["headline"], int(r["row_id"])))
        for a, b in itertools.combinations(rows, 2):
            if a["headline"] == b["headline"]:
                continue
            payload = dict(test_id=test_id, image_id=image_id, excerpt=excerpt,
                           headline_a=a["headline"], headline_b=b["headline"])
            digest = hashlib.sha256(json.dumps(
                payload, separators=(",", ":"), ensure_ascii=False
            ).encode()).hexdigest()
            candidate = (digest, int(a["row_id"]), int(b["row_id"]))
            if test_id not in winners or candidate < winners[test_id]:
                winners[test_id] = candidate
    assert (group_count, variant_count, len(winners)) == (2003, 8774, 1930)
    raw_by_id = raw.set_index(raw.row_id.astype(int), verify_integrity=True)
    for row in combined.itertuples():
        assert winners[row.test_id] == (row.selection_hash, row.row_id_a, row.row_id_b)
        for side in ("a", "b"):
            source = raw_by_id.loc[getattr(row, f"row_id_{side}")]
            assert tidy(source.headline) == getattr(row, f"headline_{side}")
            assert tidy(source.eyecatcher_id) == row.image_id
            assert tidy(source.excerpt) == (row.excerpt if pd.notna(row.excerpt) else "")
            for count in ["clicks", "impressions"]:
                assert int(source[count]) == getattr(row, f"{count}_{side}")

    parent = {key: key for key in winners}
    def find(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen = {}
    for row in clean.loc[clean.clickability_test_id.isin(winners)].itertuples():
        for field in ["slug", "headline"]:
            value = tidy(getattr(row, field)).lower()
            if not value:
                continue
            key = (field, value)
            if key in seen:
                a, b = find(row.clickability_test_id), find(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = row.clickability_test_id
    for row in combined.itertuples():
        assert find(row.test_id) == row.content_group
    assert len({find(key) for key in parent}) == 1753
    return "PASS: raw checksum, exclusions, deterministic pair selection, counts, content groups"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args()
    splits = {s: pd.read_csv(ROOT / f"data/processed/{s}.csv")
              for s in ["train", "validation", "test"]}
    assert {s: len(d) for s, d in splits.items()} == {"train": 1172, "validation": 397, "test": 361}
    combined = pd.concat(splits.values(), ignore_index=True)
    assert combined.test_id.is_unique
    assert combined.groupby("content_group")["split"].nunique().eq(1).all()
    assert combined.content_group.nunique() == 1753
    for split, frame in splits.items():
        assert frame["split"].eq(split).all()
        assert frame.headline_a.lt(frame.headline_b).all()
        for side in ["a", "b"]:
            clicks, impressions = frame[f"clicks_{side}"], frame[f"impressions_{side}"]
            assert ((impressions > 0) & (clicks >= 0) & (clicks <= impressions)).all()
        label = 100 * (frame.clicks_b / frame.impressions_b - frame.clicks_a / frame.impressions_a)
        np.testing.assert_allclose(label, frame.lift_pp, atol=1e-12, rtol=0)
        for group in frame.content_group:
            bucket = int(hashlib.sha256(("20261106:" + group).encode()).hexdigest(), 16) % 100
            assert split == ("train" if bucket < 60 else "validation" if bucket < 80 else "test")

    predictions = pd.read_csv(ROOT / "results/test_predictions.csv")
    pd.testing.assert_frame_equal(predictions[splits["test"].columns], splits["test"])
    assert np.isfinite(predictions[MODELS].to_numpy()).all()
    np.testing.assert_allclose(predictions.zero_lift, 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(predictions.training_mean, splits["train"].lift_pp.mean(), atol=1e-12, rtol=0)
    actual = predictions.lift_pp.to_numpy()
    scores = pd.read_csv(ROOT / "results/test_metrics.csv").set_index("model")
    for model in MODELS:
        error = predictions[model].to_numpy() - actual
        np.testing.assert_allclose(
            [np.sqrt(np.mean(error ** 2)), np.mean(np.abs(error))],
            # Float32 model predictions were written as short decimal CSV values.
            scores.loc[model, ["rmse_pp", "mae_pp"]], atol=1e-8, rtol=0
        )
    plan = json.loads((ROOT / "results/evaluation_plan.json").read_text())
    valid = pd.read_csv(ROOT / "results/validation_comparison.csv")
    assert valid.sort_values("rmse_pp").iloc[0].model == plan["selected_model"] == "minilm_ridge"
    for feature in ["tfidf", "minilm"]:
        tuning = pd.read_csv(ROOT / f"results/validation_{feature}_tuning.csv")
        best = tuning.sort_values(["rmse_pp", "alpha"], ascending=[True, False]).iloc[0]
        assert best.alpha == plan[f"{feature}_alpha"] == 10.0
        np.testing.assert_allclose(best.rmse_pp, valid.set_index("model").loc[f"{feature}_ridge", "rmse_pp"])

    groups = list(predictions.groupby(plan["bootstrap_unit"], sort=True).indices.values())
    assert len(groups) == 331
    squared_errors = (predictions[["tfidf_ridge", "minilm_ridge"]].to_numpy() - actual[:, None]) ** 2
    rng = np.random.default_rng(plan["bootstrap_seed"])
    draws = []
    for _ in range(plan["bootstrap_repetitions"]):
        rows = np.concatenate([groups[i] for i in rng.choice(len(groups), size=len(groups), replace=True)])
        rmse = np.sqrt(squared_errors[rows].mean(axis=0))
        gain = rmse[0] - rmse[1]
        draws.append([gain, 100 * gain / rmse[0]])
    draws = np.asarray(draws)
    np.testing.assert_allclose(draws, pd.read_csv(ROOT / "results/test_bootstrap_draws.csv"), atol=1e-10, rtol=0)
    summary = pd.read_csv(ROOT / "results/test_bootstrap_summary.csv")
    observed = np.sqrt(squared_errors.mean(axis=0))
    gain = observed[0] - observed[1]
    np.testing.assert_allclose(summary.estimate, [gain, 100 * gain / observed[0]], atol=1e-10, rtol=0)
    np.testing.assert_allclose(summary[["ci_low", "ci_high"]].to_numpy().T,
                               np.quantile(draws, [0.025, 0.975], axis=0), atol=1e-10, rtol=0)
    report = {
        "status": "PASS", "split_rows": {s: len(d) for s, d in splits.items()},
        "test_content_groups": len(groups), "bootstrap_repetitions": len(draws),
        "checks": ["Target arithmetic and arm order", "Unique experiment IDs and group-disjoint hash splits",
                   "Saved test data match predictions", "Baseline predictions and all test metrics",
                   "Validation selection", "All paired bootstrap draws and intervals"],
        "raw_audit": check_raw(args.raw, combined) if args.raw else "Not run; supply --raw to enable",
        "limits": "No encoder inference or model fitting rerun; saved predictions audited. Spark jobs not rerun.",
        "verification_environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
    }
    (ROOT / "results/review_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
