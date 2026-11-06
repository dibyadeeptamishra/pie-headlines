# Data provenance and attribution

The processed data in `data/processed/*.csv` and the source-derived fields in `results/test_predictions.csv` come from the **Upworthy Research Archive exploratory dataset**.

- Authors: J. Nathan Matias, Kevin Munger, Marianne Aubin Le Quéré, and Charles Ebersole.
- Paper: *The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media* (2021), Scientific Data. https://doi.org/10.1038/s41597-021-00934-7
- Archive and documentation: https://upworthy.natematias.com/about-the-archive
- OSF project: https://osf.io/jd64p/
- Exact source download: https://osf.io/download/3vqmp/
- Source size: 14,260,949 bytes.
- SHA-256: `8368313b060f4015a0c6fb34e6d788163cee29554144aba9390b14922eb9d8ce`.
- Dataset license: **Creative Commons Attribution 4.0 International** — https://creativecommons.org/licenses/by/4.0/

The archive documentation credits an agreement between Good/Upworthy and Cornell University. Retain attribution and the license link when redistributing these data or derivatives.

Modifications in this project: excluded a flagged date window; normalized whitespace; restricted comparisons to same-image/excerpt groups; selected a single pair per experiment; added CTR differences, content groups, split assignments, and model predictions. Changes are described in README.md. The original creators do not endorse this project.

MiniLM weights are not bundled. The encoder's model card identifies its license as Apache 2.0: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 . The included joblib files contain the fitted ridge regressions and TF-IDF vectorizer, not pretrained encoder weights.

No open-source license has yet been selected for this project's original code or prose. The dataset license and pretrained model license are separate from that choice.
