# Can embeddings predict headline click lift?

**Dibya Deepta Mishra · November 6, 2026 · Draft for review**

I started this project while preparing to discuss Florian Zettelmeyer's work on advertising measurement. The paper I had in mind was [PIE, with Brett Gordon and Robert Moakler](https://doi.org/10.3386/w35044). It asks how we can use evidence from past advertising experiments to predict incrementality for other campaigns.

I wanted to try something small with real data. Upworthy's headline experiments gave me a way to do that. My question was: if I have two titles for the same article, can their wording help me predict the difference in click-through rates?

PIE uses campaign information, including information that can be available after a campaign runs. I used headline text. The outcome is also different: headline click lift, rather than advertising incrementality. So this is a PIE-inspired exercise, not a replication.

## Building the sample

The [exploratory file](https://osf.io/download/3vqmp/) contains 22,666 variants across 4,873 experiments. I excluded experiments with any variant dated June 25, 2013 through January 10, 2014, following the [archive authors' warning](https://upworthy.natematias.com/2024-06-upworthy-archive-update.html) about possible randomization problems.

Within each experiment, I looked for distinct headlines with the same image ID and normalized excerpt. Matching the excerpt was an extra restriction; the archive does not identify it as displayed test content. I then used a deterministic hash to choose one pair per experiment. Clicks, impressions and the recorded winner played no role in choosing the pair.

That left 1,930 comparisons. A and B refer to headline order, not to which headline won. The label is `100 × (CTR(B) − CTR(A))`. If A gets a 3% click-through rate and B gets 3.4%, the observed lift is 0.4 percentage points. Random assignment supports interpreting the difference as an experimental effect; the prediction model does not identify the causal effect of individual words.

Splitting the data needed some care. Different experiments can contain the same headline or article. I linked experiments sharing normalized headlines or article slugs, checking all their cleaned variants, and kept each connected group together. The resulting split had 1,172 training experiments, 397 for validation and 361 for testing. This reduces content leakage, though it will miss related articles that those links cannot identify. All three splits come from the public exploratory file.

## What I fitted

I compared TF-IDF with frozen MiniLM embeddings. TF-IDF represents individual words and two-word combinations; its vocabulary and weights were fitted on training headlines only. MiniLM converts each headline into a normalized vector of 384 numbers. I used [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) without fine-tuning it.

For both models, the input was B's features minus A's features. I fitted ridge regression without an intercept, so reversing the headlines reverses the predicted lift. Validation selected a regularization strength of 10 for both models, and MiniLM had the lower validation error. I kept the original training fits for the test evaluation.

| Model | Test RMSE, percentage points |
|---|---:|
| Zero lift | 0.5701 |
| Training mean | 0.5701 |
| TF-IDF plus ridge | 0.5635 |
| MiniLM plus ridge | **0.5447** |

MiniLM reduced RMSE by 0.01878 percentage points relative to TF-IDF, or **3.33%**. Across 2,000 paired bootstrap resamples of the test set's 331 content groups, the approximate 95% interval for the relative reduction was **1.35% to 5.41%**.

I would call this a modest improvement. It is a reduction in prediction error, not a 3.33% increase in clicks. The labels are noisy experimental estimates. The bootstrap keeps the fitted models fixed, so it also leaves out uncertainty from retraining. And historical Upworthy headlines are a fairly particular setting.

I used Spark on one EC2 machine for preparation and a SageMaker notebook on CPU for modeling, moving files by upload and download. This dataset could have been handled on a laptop. Using AWS was part of learning the workflow.

Next I want to test the frozen models on more Upworthy experiments, after removing related-content overlap. Article context and uncertainty in the experimental labels are two additions worth trying later. For now, the result is narrower: in this held-out sample, embeddings helped predict observed headline click lift better than the simple text baseline.

Data: [The Upworthy Research Archive](https://doi.org/10.1038/s41597-021-00934-7), Matias, Munger, Aubin Le Quéré and Ebersole (2021), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This project filters the data and constructs paired comparisons and predictions.
