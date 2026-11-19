# Understanding Molecular Predictions

An interactive Streamlit dashboard for exploring measured water solubility, molecular properties, and supplied ESOL estimates across 1,128 records.

## Run locally

With Python 3.12 installed, run these commands from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit.

## Explore the data

- **Molecular-weight range** updates the summary metrics, all three charts, and the compound list. Hover over a histogram bar for its solubility range, count, and share of the selected records.
- **Property in scatter plot** lets you compare measured solubility with any of the six molecular descriptors.
- **ESOL estimate comparison** switches between measured versus estimated values and signed error versus measured solubility. Signed error is the supplied estimate minus the measurement. A positive value indicates an overestimate.
- **Compound details** shows measurements, estimate error, descriptors, SMILES, and a 2D structure drawn locally by RDKit. Click a point in either scatter plot or search by compound name to highlight that record across all three charts.

## Data and attribution

[`delaney-processed.csv`](delaney-processed.csv) matches the [Delaney ESOL dataset distributed by DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv). All 1,128 rows are retained. Solubility is expressed as log₁₀ of concentration in mol/L. The estimate column contains the ESOL values supplied with the CSV.

Original reference: John S. Delaney (2004), *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure*. [Journal article](https://pubs.acs.org/doi/10.1021/ci034243x)
