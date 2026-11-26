"""Run with: python -m streamlit run app.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

TITLE = "Understanding Molecular Predictions"
FOLDER = Path(__file__).resolve().parent
DATA = FOLDER / "delaney-processed.csv"
EXPORT = FOLDER / "gnn-predictions.csv"
MODEL_RUN = FOLDER.parent / "molecular-model-training/outputs/comparison/attentivefp-wider-seed43"
MEASURED = "Measured log S"
ESOL = "Supplied ESOL estimate"
GNN = "AttentiveFP prediction"
ERROR = "Signed error (log S)"
ABS_ERROR = "Absolute error (log S)"
COUNT_PROPERTIES = ["Ring count", "Hydrogen-bond donors", "Rotatable bonds", "Minimum atom degree"]
WEIGHT = "Molecular weight (g/mol)"
PROPERTIES = [WEIGHT, "Polar surface area (Å²)", *COUNT_PROPERTIES]
SPLITS = {"test": "Test", "validation": "Validation", "train": "Training", "all": "All records"}
SPLIT_CONTEXT = {
    "test": "Test molecules were held out during training and model selection.",
    "validation": "Validation scores were used to select the model.",
    "train": "Training scores describe molecules the model learned from.",
    "all": "The combined scores include training, validation, and test records.",
}
LOG_S_RANGE = [-12, 2]
LOG_S_TICKS = dict(tick0=0, dtick=1, tickformat=".0f", ticks="outside", ticklen=3, tickfont_size=11)
TEAL = "#087F8C"
HIGHLIGHT = "#B54A0D"
REFERENCE = dict(dash="dot", color="#263D53")
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": False}


def read_predictions():
    if EXPORT.is_file():
        return pd.read_csv(EXPORT)
    if MODEL_RUN.is_dir():
        columns = ["record_id", "compound_id", "measured_log_s", "predicted_log_s"]
        train_val = pd.read_csv(MODEL_RUN / "predictions.csv", usecols=[*columns, "split"])
        test = pd.read_csv(MODEL_RUN / "test_predictions.csv", usecols=columns).assign(split="test")
        return pd.concat([train_val, test], ignore_index=True)
    return None


def load_data():
    data = pd.read_csv(DATA).rename(columns=dict(zip(
        ["Compound ID", "measured log solubility in mols per litre",
         "ESOL predicted log solubility in mols per litre", "Molecular Weight", "Polar Surface Area",
         "Number of Rings", "Number of H-Bond Donors", "Number of Rotatable Bonds", "Minimum Degree"],
        ["Compound", MEASURED, ESOL, *PROPERTIES],
    )))
    data.index = pd.RangeIndex(1, len(data) + 1, name="record_id")
    data["plot_id"] = data.index.astype(str)
    data["smiles"] = data["smiles"].str.strip()

    saved = read_predictions()
    if saved is None:
        return data
    saved = saved.set_index("record_id", verify_integrity=True)
    if len(saved) != len(data) or not data.index.isin(saved.index).all():
        raise ValueError("The prediction export must contain exactly one row per ESOL record.")
    saved = saved.loc[data.index]
    if (not saved["compound_id"].eq(data["Compound"]).all()
            or not np.allclose(saved["measured_log_s"], data[MEASURED], atol=1e-7, rtol=0)
            or not saved["split"].isin(("train", "validation", "test")).all()
            or not np.isfinite(saved["predicted_log_s"]).all()):
        raise ValueError("The saved predictions do not match the ESOL measurements or split labels.")
    return data.join(saved[["split", "predicted_log_s"]].rename(columns={"predicted_log_s": GNN}))


def show_chart(fig, **kwargs):
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=12, b=10))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, **kwargs)


def select_compound(chart_key):
    points = st.session_state[chart_key]["selection"]["points"]
    if points and (values := points[-1].get("customdata")):
        st.session_state["compound"] = int(values[1])


def show_scatter(data, x, y, prediction, key, compound, error_extent):
    fig = px.scatter(data, x=x, y=y, custom_data=["Compound", "plot_id", ERROR, prediction],
                     color_discrete_sequence=[TEAL], render_mode="svg")
    fig.update_layout(clickmode="event+select")
    fig.update_yaxes(gridcolor="rgba(45, 65, 85, 0.10)")
    count = len(data)
    size = 4 if count > 500 else 6 if count > 100 else 7
    opacity = .38 if count > 500 else .62 if count > 100 else .75
    x_format = ".0f" if x in COUNT_PROPERTIES else ".2f"
    y_format = "+.2f" if y == ERROR else ".2f"
    hover = f"<b>%{{customdata[0]}}</b><br>{x}: %{{x:{x_format}}}<br>{y}: %{{y:{y_format}}}"
    if y == prediction:
        hover += "<br>Signed error: %{customdata[2]:+.2f} log S"
    elif y in (ERROR, ABS_ERROR):
        hover += f"<br>{prediction}: %{{customdata[3]:.2f}} log S"
    hover += "<extra></extra>"
    fig.update_traces(marker=dict(size=size, opacity=opacity, line_width=0), hovertemplate=hover)

    if x in COUNT_PROPERTIES:
        span = data[x].max() - data[x].min()
        fig.update_xaxes(tick0=0, dtick=max(1, int(np.ceil(span / 10))))
    if x == MEASURED:
        fig.update_xaxes(range=LOG_S_RANGE, constrain="domain", **LOG_S_TICKS)
    if y == prediction:
        low, high = LOG_S_RANGE
        fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=REFERENCE, layer="below")
        fig.update_yaxes(range=LOG_S_RANGE, scaleanchor="x", scaleratio=1, constrain="domain", **LOG_S_TICKS)
    elif y == ERROR:
        fig.add_hline(y=0, line=REFERENCE, layer="below")
        fig.update_yaxes(range=[-error_extent, error_extent], zeroline=False)
    elif y == MEASURED and data[MEASURED].max() - data[MEASURED].min() >= 3:
        fig.update_yaxes(**LOG_S_TICKS)
    elif y == ABS_ERROR:
        fig.update_yaxes(range=[0, error_extent])

    if compound is not None:
        fig.add_scatter(x=[compound[x]], y=[compound[y]], mode="markers",
                        marker=dict(color=HIGHLIGHT, size=12, line=dict(color="white", width=2)),
                        customdata=[compound[["Compound", "plot_id", ERROR, prediction]].tolist()],
                        showlegend=False, hovertemplate=hover)
    show_chart(fig, key=key, selection_mode="points", on_select=lambda: select_compound(key))


st.set_page_config(page_title=TITLE, layout="wide")
st.html("""
<style>
.stMainBlockContainer {max-width: 1370px; padding-top: 2.4rem;}
.st-key-solubility-charts h3, .st-key-estimate-and-detail h3 {font-size: 1.35rem;}
@media (max-width: 720px) {
  :is(.st-key-model-controls, .st-key-solubility-charts, .st-key-estimate-and-detail) .stHorizontalBlock {
    flex-wrap: wrap;
  }
  :is(.st-key-model-controls, .st-key-solubility-charts, .st-key-estimate-and-detail) .stHorizontalBlock > .stColumn {
    min-width: 100%;
  }
}
</style>
""")
st.title(TITLE)
st.caption("Aqueous solubility in log₁₀(mol/L). An increase of 1 log S unit means ten times the solubility.")

try:
    data = load_data()
except (ValueError, OSError, KeyError) as error:
    st.error(f"Could not load the dashboard data. {error}")
    st.stop()

has_model = GNN in data.columns
prediction = ESOL
split = "all"
with st.container(key="model-controls"):
    if has_model:
        split_control, source_control, weight_control = st.columns([1, 1.15, 1.65], vertical_alignment="center")
        split = split_control.selectbox("Data split", list(SPLITS), format_func=SPLITS.get)
        prediction = source_control.selectbox("Prediction source", [GNN, ESOL])
    else:
        weight_control = st
    lower, upper = int(data[WEIGHT].min()), int(np.ceil(data[WEIGHT].max()))
    limits = weight_control.slider("Molecular-weight range (g/mol)", lower, upper, (lower, upper),
                                   key="weight_limits")
if has_model:
    st.caption(f"AttentiveFP · seed 43 · experimental split. {SPLIT_CONTEXT[split]}")

cohort = data if split == "all" else data.loc[data["split"].eq(split)]
selected = cohort.loc[cohort[WEIGHT].between(*limits)].copy()
if st.session_state.get("compound") not in selected.index:
    st.session_state.pop("compound", None)
if selected.empty:
    st.info("No records in this selection. Widen the molecular-weight range or choose another split.")
    st.stop()
selected[ERROR] = selected[prediction] - selected[MEASURED]
selected[ABS_ERROR] = selected[ERROR].abs()
compound_id = st.session_state.get("compound")
row = selected.loc[compound_id] if compound_id is not None else None

count, mae, rmse = st.columns(3)
count.metric("Selected records", f"{len(selected):,} / {len(cohort):,}")
other = (ESOL if prediction == GNN else GNN) if has_model else None
other_error = selected[other] - selected[MEASURED] if other else None
largest_error = selected[ABS_ERROR].max()
if other_error is not None:
    largest_error = max(largest_error, other_error.abs().max())
error_extent = max(.5, 1.08 * largest_error)
model_name = "AttentiveFP" if prediction == GNN else "ESOL"
other_name = "ESOL" if prediction == GNN else "AttentiveFP"
for metric, label, value, compare in (
    (mae, "MAE", selected[ABS_ERROR].mean(), None if other_error is None else other_error.abs().mean()),
    (rmse, "RMSE", np.sqrt(selected[ERROR].pow(2).mean()),
     None if other_error is None else np.sqrt(other_error.pow(2).mean())),
):
    metric.metric(f"{model_name} {label}", f"{value:.3f} log S",
                  delta=f"{other_name} {compare:.3f}" if compare is not None else None, delta_color="off",
                  delta_arrow="off",
                  help=("Mean absolute error is the average distance from the measurement." if label == "MAE"
                        else "Root mean squared error gives larger misses more weight."))

with st.container(key="solubility-charts"):
    distribution, relationship = st.columns([1, 1.2])
    with distribution:
        st.subheader("Measured solubility")
        edges = np.linspace(*LOG_S_RANGE, 29)
        counts, _ = np.histogram(selected[MEASURED], bins=edges)
        histogram = go.Figure(go.Bar(
            x=(edges[:-1] + edges[1:]) / 2, y=counts, width=.48, marker_color=TEAL,
            customdata=np.column_stack((edges[:-1], edges[1:], counts / len(selected))),
            hovertemplate="%{customdata[0]:.1f} to %{customdata[1]:.1f} log S<br>"
                          "%{y:,d} records (%{customdata[2]:.1%})<extra></extra>",
        ))
        histogram.update_xaxes(title=MEASURED, range=LOG_S_RANGE, **LOG_S_TICKS)
        histogram.update_yaxes(title="Records", tickformat="d")
        if counts.max() <= 5:
            histogram.update_yaxes(dtick=1)
        median = selected[MEASURED].median()
        histogram.add_vline(x=median, line=REFERENCE)
        if row is not None:
            histogram.add_vline(x=row[MEASURED], line_color=HIGHLIGHT, line_width=3,
                                annotation_text=f"Selected: {row[MEASURED]:.2f}",
                                annotation_position="top left" if row[MEASURED] >= -5 else "top right",
                                annotation_font_color=HIGHLIGHT)
        show_chart(histogram)
        q1, q3 = selected[MEASURED].quantile([.25, .75])
        st.caption(f"The middle half spans {q1:.2f} to {q3:.2f} log S. The dotted line marks the median.")

    with relationship:
        st.subheader("Molecular properties")
        descriptor_control, view_control = st.columns([1, 1])
        descriptor = descriptor_control.selectbox("Property", PROPERTIES)
        property_views = [ABS_ERROR, MEASURED] if has_model else [MEASURED, ABS_ERROR]
        property_view = view_control.selectbox("Show by property", property_views)
        show_scatter(selected, descriptor, property_view, prediction, "property_points", row, error_extent)
        if descriptor == "Minimum atom degree":
            st.caption("Minimum degree is the fewest bonded neighbors of any atom in the molecule.")
        elif property_view == ABS_ERROR:
            st.caption(f"Absolute error exceeds 1 log S for {selected[ABS_ERROR].gt(1).sum():,} of "
                       f"{len(selected):,} selected records.")

with st.container(key="estimate-and-detail"):
    comparison, lookup = st.columns([1, 1.2])
    with comparison:
        st.subheader("Prediction agreement")
        view = st.radio("Comparison view", ["Agreement", "Signed error"], horizontal=True,
                        key="estimate_view", label_visibility="collapsed")
        show_scatter(selected, MEASURED, ERROR if view == "Signed error" else prediction,
                     prediction, "estimate_points", row, error_extent)
        if view == "Signed error":
            st.caption("Signed error is predicted minus measured solubility. Positive errors are overestimates.")
        else:
            st.caption("The diagonal marks exact agreement. Select any point to inspect its compound.")

    with lookup:
        st.subheader("Compound details")
        examples = st.columns(2)
        for column, label, record_id in (
            (examples[0], "Smallest error", selected[ABS_ERROR].idxmin()),
            (examples[1], "Largest error", selected[ABS_ERROR].idxmax()),
        ):
            column.button(label, on_click=st.session_state.update, kwargs={"compound": int(record_id)},
                          help="Select a compound by absolute error for the active predictor.")
        compounds = selected["Compound"].to_dict()
        st.selectbox("Compound", sorted(compounds, key=lambda key: compounds[key].casefold()),
                     format_func=compounds.get, index=None, placeholder="Search by compound name", key="compound")
        if row is not None:
            split_label = f" · {SPLITS[row['split']]}" if has_model else ""
            st.write(f"**Measured log S:** {row[MEASURED]:.3f}{split_label}")
            for column in (GNN, ESOL) if has_model else (ESOL,):
                st.write(f"**{column}:** {row[column]:.3f} · error {row[column] - row[MEASURED]:+.3f} log S")
            molecule = Chem.MolFromSmiles(row["smiles"])
            if molecule is not None:
                st.image(Draw.MolToImage(molecule, size=(380, 200)), width=320)
            with st.expander("Molecular properties and SMILES"):
                for label in PROPERTIES:
                    st.write(f"{label}: {row[label]:.4g}")
                st.code(row["smiles"], language=None)

st.caption("Source: Delaney ESOL measurements and supplied estimates."
           + (" AttentiveFP predictions are loaded from the saved model run." if has_model else ""))
