# Understanding Molecular Predictions

An interactive Streamlit dashboard for exploring water solubility and comparing saved AttentiveFP predictions with the supplied ESOL estimates.

## Run locally

With Python 3.12 installed, run these commands from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit.

## Explore model performance

- **Data split** opens on the 114 held-out test records. Switch to the validation set (112), training set (902), or all records for exploration. The split is experimental.
- **Prediction source** switches the active model between AttentiveFP and the supplied ESOL estimates. MAE and RMSE are calculated on the same filtered records for both sources.
- **Molecular properties** plots measured solubility or absolute error against any of the six molecular descriptors. **Prediction agreement** switches between predicted versus measured solubility and signed error versus measured solubility.
- **Compound details** shows the measurement, predictions, errors, and structure. Click a scatter point, search by name, or use **Smallest error** and **Largest error** to highlight a record across all three charts.

## Data and attribution

[`delaney-processed.csv`](delaney-processed.csv) matches the [Delaney ESOL dataset distributed by DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv). All 1,128 rows are retained. Solubility is expressed as log₁₀ of concentration in mol/L. The estimate column contains the ESOL values supplied with the CSV.

In the course workspace, the app reads `predictions.csv` and `test_predictions.csv`
from the saved AttentiveFP run (128 hidden channels, seed 43). A standalone copy
can load the same predictions from `gnn-predictions.csv` if provided. Predictions
are matched to the original ESOL CSV by its one-based record ID and checked
against the compound name and measured value. With only the ESOL CSV available,
the dashboard opens in ESOL exploration mode.

Original reference: John S. Delaney (2004), *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure*. [Journal article](https://pubs.acs.org/doi/10.1021/ci034243x)
