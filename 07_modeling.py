"""Reproduction of the saved experiment; see README.md before rerunning.
Runs CPU encoder inference and overwrites local results.
"""

# %%
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import sys

# Locate this repository from the notebook's working directory.
project_dir = next(
    (path for path in [Path.cwd(), *Path.cwd().parents]
     if (path / "data/processed/train.csv").is_file()), None
)
if project_dir is None:
    raise FileNotFoundError("Open this notebook from the extracted pie-headlines folder.")
print("Project folder:", project_dir)
print("Python:", sys.version.split()[0])
for package in ("numpy", "pandas", "scikit-learn", "torch", "sentence-transformers"):
    try:
        print(package, version(package))
    except PackageNotFoundError:
        print(package, "not installed; see requirements-modeling.txt")


# %%
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Load training and validation data.
data_dir = project_dir / "data/processed"
train = pd.read_csv(data_dir / "train.csv")
valid = pd.read_csv(data_dir / "validation.csv")

# Verify that related content stays within one split.
assert set(train["content_group"]).isdisjoint(valid["content_group"])

# Our prediction target: CTR_B minus CTR_A, in percentage points.
y_train = train["lift_pp"].to_numpy(dtype=float)
y_valid = valid["lift_pp"].to_numpy(dtype=float)

print("Training experiments:", len(train))
print("Validation experiments:", len(valid))
print(f"Training mean lift: {y_train.mean():.4f} percentage points")

# Create one prediction for each validation experiment.
baseline_predictions = {
    "zero_lift": np.zeros(len(valid)),
    "training_mean": np.full(len(valid), y_train.mean()),
}

# Reusable scoring function for later models.
def score_predictions(name, actual, predicted):
    return {
        "model": name,
        "rmse_pp": root_mean_squared_error(actual, predicted),
        "mae_pp": mean_absolute_error(actual, predicted),
    }

baseline_results = pd.DataFrame([
    score_predictions(name, y_valid, predictions)
    for name, predictions in baseline_predictions.items()
])

# Save full-precision results for our eventual post.
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)
baseline_results.to_csv(
    results_dir / "validation_baselines.csv", index=False
)

print(baseline_results.round(4).to_string(index=False))

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import joblib

# Learn vocabulary and word frequencies from TRAINING headlines only.
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000,
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
)

training_headlines = (
    train["headline_a"].tolist() + train["headline_b"].tolist()
)
vectorizer.fit(training_headlines)

# Convert each headline pair into a difference in text features.
def tfidf_difference(frame):
    a = vectorizer.transform(frame["headline_a"])
    b = vectorizer.transform(frame["headline_b"])
    return b - a

X_train_tfidf = tfidf_difference(train)
X_valid_tfidf = tfidf_difference(valid)

# Try a fixed set of regularization strengths.
tfidf_models = {}
tuning_rows = []

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    model = Ridge(
        alpha=alpha, fit_intercept=False, solver="lsqr", tol=1e-6
    )
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_valid_tfidf)
    row = score_predictions("tfidf_ridge", y_valid, predictions)
    row["alpha"] = alpha

    tuning_rows.append(row)
    tfidf_models[alpha] = model

# Select by validation RMSE; prefer stronger shrinkage if tied.
tfidf_tuning = pd.DataFrame(tuning_rows).sort_values(
    ["rmse_pp", "alpha"], ascending=[True, False]
)
best_alpha = float(tfidf_tuning.iloc[0]["alpha"])
best_tfidf = tfidf_models[best_alpha]
tfidf_valid_pred = best_tfidf.predict(X_valid_tfidf)

tfidf_result = score_predictions("tfidf_ridge", y_valid, tfidf_valid_pred)
comparison = pd.concat(
    [baseline_results, pd.DataFrame([tfidf_result])],
    ignore_index=True,
)

# Save results and the fitted model.
models_dir = project_dir / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(
    {"vectorizer": vectorizer, "model": best_tfidf},
    models_dir / "tfidf_ridge.joblib",
)
tfidf_tuning.to_csv(results_dir / "validation_tfidf_tuning.csv", index=False)
comparison.to_csv(results_dir / "validation_comparison.csv", index=False)

print("Training feature matrix:", X_train_tfidf.shape)
print("Selected alpha:", best_alpha)
print(comparison.round(4).to_string(index=False))

# %%
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

data_dir = project_dir / "data/processed"

train = pd.read_csv(data_dir / "train.csv")
valid = pd.read_csv(data_dir / "validation.csv")

torch.set_num_threads(2)

# Pin the pretrained model version for reproducibility.
model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

encoder = SentenceTransformer(
    model_id,
    revision=model_revision,
    device="cpu",
    cache_folder=str(project_dir / "model_cache"),
)

def embedding_difference(frame):
    # Put all A headlines first, followed by all B headlines.
    texts = frame["headline_a"].tolist() + frame["headline_b"].tolist()

    vectors = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    n = len(frame)
    return vectors[n:] - vectors[:n]

X_train_emb = embedding_difference(train)
X_valid_emb = embedding_difference(valid)

# Save features and row identifiers so we can reuse them.
feature_dir = project_dir / "data/features"
feature_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    feature_dir / "minilm_features.npz",
    train=X_train_emb,
    validation=X_valid_emb,
    train_ids=train["test_id"].to_numpy(dtype=str),
    validation_ids=valid["test_id"].to_numpy(dtype=str),
)

config = {
    "model_id": model_id,
    "revision": model_revision,
    "normalize_embeddings": True,
    "feature_direction": "B minus A",
}
config_path = project_dir / "models/embedding_config.json"
config_path.parent.mkdir(exist_ok=True)
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

print("Training embeddings:", X_train_emb.shape)
print("Validation embeddings:", X_valid_emb.shape)

# %%
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

results_dir = project_dir / "results"

train = pd.read_csv(project_dir / "data/processed/train.csv")
valid = pd.read_csv(project_dir / "data/processed/validation.csv")

# Load embeddings and verify that their rows match our experiment IDs.
with np.load(
    project_dir / "data/features/minilm_features.npz",
    allow_pickle=False,
) as saved:
    X_train_emb = saved["train"]
    X_valid_emb = saved["validation"]

    assert np.array_equal(
        saved["train_ids"], train["test_id"].to_numpy(dtype=str)
    ), "Training rows do not match embeddings"

    assert np.array_equal(
        saved["validation_ids"], valid["test_id"].to_numpy(dtype=str)
    ), "Validation rows do not match embeddings"

y_train = train["lift_pp"].to_numpy(dtype=float)
y_valid = valid["lift_pp"].to_numpy(dtype=float)

# Fit on training data; compare settings on validation data.
embedding_models = {}
tuning_rows = []

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    model = Ridge(
        alpha=alpha, fit_intercept=False, solver="lsqr", tol=1e-6
    )
    model.fit(X_train_emb, y_train)
    predictions = model.predict(X_valid_emb)

    tuning_rows.append({
        "model": "minilm_ridge",
        "alpha": alpha,
        "rmse_pp": root_mean_squared_error(y_valid, predictions),
        "mae_pp": mean_absolute_error(y_valid, predictions),
    })
    embedding_models[alpha] = model

embedding_tuning = pd.DataFrame(tuning_rows).sort_values(
    ["rmse_pp", "alpha"], ascending=[True, False]
)
best_embedding_alpha = float(embedding_tuning.iloc[0]["alpha"])
best_embedding_model = embedding_models[best_embedding_alpha]

best_result = embedding_tuning.iloc[0][
    ["model", "rmse_pp", "mae_pp"]
].to_dict()

# Add the result to our existing comparison; allow rerunning this cell.
previous = pd.read_csv(results_dir / "validation_comparison.csv")
previous = previous.loc[previous["model"] != "minilm_ridge"]

comparison = pd.concat(
    [previous, pd.DataFrame([best_result])], ignore_index=True
)

# Save the fitted regression and its encoder configuration.
encoder_config = json.loads(
    (project_dir / "models/embedding_config.json").read_text()
)
joblib.dump(
    {"model": best_embedding_model, "encoder_config": encoder_config},
    project_dir / "models/minilm_ridge.joblib",
)

embedding_tuning.to_csv(
    results_dir / "validation_minilm_tuning.csv", index=False
)
comparison.to_csv(results_dir / "validation_comparison.csv", index=False)

print("Selected embedding alpha:", best_embedding_alpha)
print(comparison.round(4).to_string(index=False))

# %%
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

data_dir = project_dir / "data/processed"
results_dir = project_dir / "results"

# Load the already-fitted models.
tfidf_saved = joblib.load(project_dir / "models/tfidf_ridge.joblib")
minilm_saved = joblib.load(project_dir / "models/minilm_ridge.joblib")
encoder_config = minilm_saved["encoder_config"]

train = pd.read_csv(data_dir / "train.csv")
valid = pd.read_csv(data_dir / "validation.csv")
validation_scores = pd.read_csv(results_dir / "validation_comparison.csv")

# Record our choices before opening the test data.
selected_model = validation_scores.sort_values("rmse_pp").iloc[0]["model"]
evaluation_plan = {
    "selected_model": str(selected_model),
    "selection_metric": "validation RMSE",
    "training_rows": len(train),
    "tfidf_alpha": float(tfidf_saved["model"].alpha),
    "minilm_alpha": float(minilm_saved["model"].alpha),
    "encoder": encoder_config,
    "primary_comparison": "MiniLM versus TF-IDF",
    "bootstrap_unit": "content_group",
    "bootstrap_repetitions": 2000,
    "bootstrap_seed": 20261106,
}
(results_dir / "evaluation_plan.json").write_text(
    json.dumps(evaluation_plan, indent=2), encoding="utf-8"
)

# Open the held-out test data and check group separation.
test = pd.read_csv(data_dir / "test.csv")
development_groups = set(train["content_group"]) | set(valid["content_group"])

assert len(test) == 361
assert set(test["content_group"]).isdisjoint(development_groups)

# Apply the TRAINING vocabulary to test headlines.
vectorizer = tfidf_saved["vectorizer"]
X_test_tfidf = (
    vectorizer.transform(test["headline_b"])
    - vectorizer.transform(test["headline_a"])
)

# Encode test headlines with the same frozen MiniLM model.
torch.set_num_threads(2)
encoder = SentenceTransformer(
    encoder_config["model_id"],
    revision=encoder_config["revision"],
    device="cpu",
    cache_folder=str(project_dir / "model_cache"),
)

n_test = len(test)
texts = test["headline_a"].tolist() + test["headline_b"].tolist()
vectors = encoder.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=encoder_config["normalize_embeddings"],
)
X_test_emb = vectors[n_test:] - vectors[:n_test]

# Generate predictions without fitting or tuning anything.
predictions = {
    "zero_lift": np.zeros(n_test),
    "training_mean": np.full(n_test, train["lift_pp"].mean()),
    "tfidf_ridge": tfidf_saved["model"].predict(X_test_tfidf),
    "minilm_ridge": minilm_saved["model"].predict(X_test_emb),
}

y_test = test["lift_pp"].to_numpy(dtype=float)
test_results = pd.DataFrame([
    {
        "model": name,
        "rmse_pp": root_mean_squared_error(y_test, prediction),
        "mae_pp": mean_absolute_error(y_test, prediction),
    }
    for name, prediction in predictions.items()
])

# Save metrics and individual predictions for uncertainty estimates.
test_results.to_csv(results_dir / "test_metrics.csv", index=False)

test_predictions = test.copy()
for name, prediction in predictions.items():
    test_predictions[name] = prediction
test_predictions.to_csv(results_dir / "test_predictions.csv", index=False)

print("Model selected using validation:", selected_model)
print("Test experiments:", n_test)
print(test_results.round(4).to_string(index=False))

# %%
from pathlib import Path
import json
import numpy as np
import pandas as pd

results_dir = project_dir / "results"

df = pd.read_csv(results_dir / "test_predictions.csv").reset_index(drop=True)
plan = json.loads((results_dir / "evaluation_plan.json").read_text())

group_column = plan["bootstrap_unit"]
n_boot = plan["bootstrap_repetitions"]
rng = np.random.default_rng(plan["bootstrap_seed"])

assert df[group_column].notna().all()

# Each array contains the row positions belonging to one content group.
group_rows = list(df.groupby(group_column, sort=True).indices.values())
n_groups = len(group_rows)

# Columns are ordered: TF-IDF, then MiniLM.
actual = df["lift_pp"].to_numpy(dtype=float)
predicted = df[["tfidf_ridge", "minilm_ridge"]].to_numpy(dtype=float)
squared_errors = (predicted - actual[:, None]) ** 2

# Calculate the observed improvement using all test experiments.
observed_rmse = np.sqrt(squared_errors.mean(axis=0))
gain_pp = observed_rmse[0] - observed_rmse[1]
gain_pct = 100 * gain_pp / observed_rmse[0]

# Store absolute and relative improvements for each bootstrap sample.
draws = np.empty((n_boot, 2))

for b in range(n_boot):
    selected_groups = rng.choice(
        n_groups, size=n_groups, replace=True
    )
    rows = np.concatenate([group_rows[g] for g in selected_groups])

    # Both models use precisely the same resampled experiments.
    sampled_rmse = np.sqrt(squared_errors[rows].mean(axis=0))
    difference = sampled_rmse[0] - sampled_rmse[1]

    draws[b, 0] = difference
    draws[b, 1] = 100 * difference / sampled_rmse[0]

# Approximate 95% percentile confidence intervals.
bounds = np.quantile(draws, [0.025, 0.975], axis=0)

bootstrap_summary = pd.DataFrame({
    "measure": ["RMSE reduction (pp)", "RMSE reduction (%)"],
    "estimate": [gain_pp, gain_pct],
    "ci_low": bounds[0],
    "ci_high": bounds[1],
})

bootstrap_summary.to_csv(
    results_dir / "test_bootstrap_summary.csv", index=False
)
pd.DataFrame(draws, columns=["gain_pp", "gain_pct"]).to_csv(
    results_dir / "test_bootstrap_draws.csv", index=False
)

print("Test experiments:", len(df))
print("Content groups:", n_groups)
print("Bootstrap repetitions:", n_boot)
print(bootstrap_summary.round(5).to_string(index=False))

# %%
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

results_dir = project_dir / "results"
figure_dir = project_dir / "figures"
figure_dir.mkdir(exist_ok=True)

metrics = pd.read_csv(results_dir / "test_metrics.csv").set_index("model")
summary = pd.read_csv(results_dir / "test_bootstrap_summary.csv")
groups = pd.read_csv(
    results_dir / "test_predictions.csv", usecols=["content_group"]
)
plan = json.loads((results_dir / "evaluation_plan.json").read_text())

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "pdf.fonttype": 42,
})

def save_and_show(fig, filename):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"{filename}.{extension}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.show()
    plt.close(fig)

# Figure 1: Compare test RMSE, with the axis starting at zero.
labels = {
    "zero_lift": "Zero lift",
    "training_mean": "Training mean",
    "tfidf_ridge": "TF-IDF + ridge",
    "minilm_ridge": "MiniLM + ridge",
}
values = metrics.loc[list(labels), "rmse_pp"]

fig, ax = plt.subplots(figsize=(7.4, 4.2))
bars = ax.barh(
    list(labels.values()),
    values,
    color=["#BCC5CE", "#BCC5CE", "#5876A3", "#087F8C"],
    height=0.6,
)
ax.bar_label(bars, fmt="%.4f", padding=6)
ax.invert_yaxis()
ax.set_xlim(0, values.max() * 1.20)
ax.set_xlabel("Test RMSE (percentage points; lower is better)")
ax.set_title(
    f"Predicting headline CTR differences\n{len(groups)} held-out experiments",
    loc="left",
    pad=12,
)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.15)
ax.set_axisbelow(True)
save_and_show(fig, "test_rmse")

# Figure 2: Relative RMSE improvement and its bootstrap interval.
interval = summary.loc[
    summary["measure"] == "RMSE reduction (%)"
].iloc[0]
estimate = interval["estimate"]
low, high = interval["ci_low"], interval["ci_high"]

fig, ax = plt.subplots(figsize=(7.4, 3.5))
ax.errorbar(
    estimate, 0,
    xerr=[[estimate - low], [high - estimate]],
    fmt="o", color="#087F8C",
    markersize=9, capsize=7, elinewidth=2.5,
)
ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
ax.text(
    estimate, 0.22,
    f"{estimate:.2f}% reduction\n95% CI: {low:.2f}% to {high:.2f}%",
    ha="center", va="bottom",
)
ax.set_xlim(min(-0.5, low - 0.5), high + 0.8)
ax.set_ylim(-0.6, 0.8)
ax.set_yticks([])
ax.set_xlabel("Reduction in test RMSE (%); positive favors MiniLM")
ax.set_title(
    "MiniLM compared with TF-IDF\n"
    f"{plan['bootstrap_repetitions']:,} bootstrap resamples; "
    f"{groups['content_group'].nunique()} content groups",
    loc="left",
    pad=12,
)
ax.spines[["top", "right", "left"]].set_visible(False)
save_and_show(fig, "minilm_gain_ci")

print("Figures saved to:", figure_dir)
