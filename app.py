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
DATA = Path(__file__).resolve().with_name("delaney-processed.csv")
MINIMUM_DEGREE = "Minimum atom degree"
PROPERTIES = ["Molecular weight (g/mol)", "Polar surface area (Å²)", "Ring count",
              "Hydrogen-bond donors", "Rotatable bonds", MINIMUM_DEGREE]
MEASURED = "Measured log S"
ESTIMATE = "Supplied ESOL estimate"
ERROR = "Estimate error"
SIGNED_ERROR = "Signed error (log S)"
LOG_S_RANGE = [-12, 2]
LOG_S_TICKS = dict(tick0=0, dtick=1, tickformat=".0f", ticks="outside", ticklen=3, tickfont_size=11)
TEAL = "#087F8C"
HIGHLIGHT = "#B54A0D"
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": False}

st.set_page_config(page_title=TITLE, layout="wide")
st.html("""
<style>
.stMainBlockContainer {max-width: 1370px; padding-top: 2.4rem;}
.st-key-solubility-charts h3, .st-key-estimate-and-detail h3 {font-size: 1.35rem;}
@media (max-width: 720px) {
  .st-key-solubility-charts .stHorizontalBlock, .st-key-estimate-and-detail .stHorizontalBlock {
    flex-wrap: wrap;
  }
  .st-key-solubility-charts .stHorizontalBlock > .stColumn,
  .st-key-estimate-and-detail .stHorizontalBlock > .stColumn {min-width: 100%;}
}
</style>
""")


@st.cache_data
def load_data():
    data = pd.read_csv(DATA).rename(columns={
        "Compound ID": "Compound",
        "measured log solubility in mols per litre": MEASURED,
        "ESOL predicted log solubility in mols per litre": ESTIMATE,
        "Molecular Weight": PROPERTIES[0], "Polar Surface Area": PROPERTIES[1],
        "Number of Rings": PROPERTIES[2], "Number of H-Bond Donors": PROPERTIES[3],
        "Number of Rotatable Bonds": PROPERTIES[4],
        "Minimum Degree": MINIMUM_DEGREE,
    })
    data["smiles"] = data["smiles"].str.strip()
    data[ERROR] = data[ESTIMATE] - data[MEASURED]
    return data


def show_chart(fig, **kwargs):
    fig.update_layout(height=304, margin=dict(l=10, r=10, t=12, b=10))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, **kwargs)


def select_compound(chart_key):
    points = st.session_state[chart_key]["selection"]["points"]
    if points and (compound := points[-1].get("customdata")):
        st.session_state["compound"] = compound[0]


def show_scatter(fig, key, x, y, compound):
    fig.update_layout(clickmode="event+select")
    fig.update_yaxes(gridcolor="rgba(45, 65, 85, 0.10)")
    count = len(fig.data[0].x)
    if count > 500:
        size, opacity = 4, .36
    elif count > 100:
        size, opacity = 5, .55
    else:
        size, opacity = 7, .72

    x_format = ".0f" if x in PROPERTIES[2:] else ".2f"
    y_label, y_format = (SIGNED_ERROR, "+.2f") if y == ERROR else (y, ".2f")
    hover = f"<b>%{{customdata[0]}}</b><br>{x}: %{{x:{x_format}}}<br>{y_label}: %{{y:{y_format}}}"
    if y == ESTIMATE:
        hover += "<br>Estimate − measured: %{customdata[1]:+.2f} log S"
    if y == ERROR:
        hover += "<br>Supplied estimate: %{customdata[2]:.2f} log S"
    hover += "<extra></extra>"
    fig.update_traces(marker=dict(size=size, opacity=opacity, line_width=0), hovertemplate=hover)
    if compound is not None:
        fig.add_scatter(
            x=[compound[x]], y=[compound[y]], mode="markers", name=compound["Compound"],
            marker=dict(color=HIGHLIGHT, size=12, line=dict(color="white", width=2)),
            customdata=[[compound["Compound"], compound[ERROR], compound[ESTIMATE]]], showlegend=False,
            hovertemplate=hover,
        )
    show_chart(fig, key=key, selection_mode="points", on_select=lambda: select_compound(key))


data = load_data()
st.title(TITLE)
st.caption("ESOL dataset · Water solubility in log₁₀(mol/L). A rise of 1 log S unit means ten times the solubility.")

# One shared filter for the summaries, charts and compound list.
weight = PROPERTIES[0]
lower, upper = int(data[weight].min()), int(data[weight].max()) + 1
weight_filter, property_control = st.columns([1.5, 1], vertical_alignment="center")
limits = weight_filter.slider("Molecular-weight range (g/mol)", lower, upper, (lower, upper))
property_name = property_control.selectbox("Property in scatter plot", PROPERTIES)
selected = data.loc[data[weight].between(*limits)]
compounds = sorted(selected["Compound"], key=str.casefold)
if st.session_state.get("compound") not in compounds:
    st.session_state.pop("compound", None)
if selected.empty:
    st.info("No records in this weight range. Widen the range to see compounds.")
    st.stop()

compound = st.session_state.get("compound")
row = selected.loc[selected["Compound"].eq(compound)].iloc[0] if compound is not None else None

count, solubility, error = st.columns(3)
count.metric("Selected records", f"{len(selected):,} / {len(data):,}")
median = selected[MEASURED].median()
solubility.metric("Median measured log S", f"{median:.2f}")
error.metric("Mean absolute estimate error", f"{selected[ERROR].abs().mean():.2f} log S",
             help="Average distance between the supplied ESOL estimate and the measured value.")
st.caption("Click a point in either scatter plot to highlight its compound and open its details below.")

# Dataset overview and descriptor comparison.
with st.container(key="solubility-charts"):
    distribution, relationship = st.columns([1, 1.2])
    with distribution:
        st.subheader("Measured solubility")
        edges = np.linspace(*LOG_S_RANGE, 29)
        bin_counts, _ = np.histogram(selected[MEASURED], bins=edges)
        histogram = go.Figure(go.Bar(
            x=(edges[:-1] + edges[1:]) / 2, y=bin_counts, width=.48, marker_color=TEAL,
            customdata=np.column_stack((edges[:-1], edges[1:], bin_counts / len(selected))),
            hovertemplate="%{customdata[0]:.1f} to %{customdata[1]:.1f} log S<br>"
                          "%{y:,d} records (%{customdata[2]:.1%})<extra></extra>",
        ))
        histogram.update_xaxes(title=MEASURED, range=LOG_S_RANGE, **LOG_S_TICKS)
        histogram.update_yaxes(title="Records", tickformat="d")
        if bin_counts.max() <= 5:
            histogram.update_yaxes(dtick=1)
        histogram.add_vline(x=median, line_dash="dot", line_color="#263D53")
        if row is not None:
            histogram.add_vline(x=row[MEASURED], line_color=HIGHLIGHT, line_width=3,
                                annotation_text=f"Selected: {row[MEASURED]:.2f}",
                                annotation_position="top left" if row[MEASURED] >= -5 else "top right",
                                annotation_font_color=HIGHLIGHT)
        show_chart(histogram)
        q1, q3 = selected[MEASURED].quantile([.25, .75])
        st.caption(f"The middle half spans {q1:.2f} to {q3:.2f} log S. The dotted line marks the median.")

    with relationship:
        st.subheader("Solubility by property")
        property_plot = px.scatter(
            selected, x=property_name, y=MEASURED, hover_name="Compound",
            custom_data=["Compound"],
            color_discrete_sequence=[TEAL], render_mode="svg",
        )
        if selected[MEASURED].max() - selected[MEASURED].min() >= 3:
            property_plot.update_yaxes(**LOG_S_TICKS)
        if property_name in PROPERTIES[2:]:
            span = selected[property_name].max() - selected[property_name].min()
            property_plot.update_xaxes(tick0=0, dtick=max(1, int(np.ceil(span / 10))))
        else:
            property_plot.update_xaxes(nticks=9)
        show_scatter(property_plot, "property_points", property_name, MEASURED, row)
        if property_name == weight and len(selected) >= 20:
            light_cutoff, heavy_cutoff = selected[weight].quantile([.25, .75])
            if light_cutoff < heavy_cutoff:
                light_median = selected.loc[selected[weight].le(light_cutoff), MEASURED].median()
                heavy_median = selected.loc[selected[weight].gt(heavy_cutoff), MEASURED].median()
                st.caption(f"Median log S is {light_median:.2f} in the lightest quarter and "
                           f"{heavy_median:.2f} in the heaviest quarter of the selection.")
        elif property_name == MINIMUM_DEGREE:
            degree_one = selected[MINIMUM_DEGREE].eq(1).sum()
            st.caption("Minimum degree is the fewest bonded neighbors of any atom in the structure. "
                       f"{degree_one:,} of {len(selected):,} selected records have a minimum degree of 1.")

# Estimate agreement and individual compound details.
with st.container(key="estimate-and-detail"):
    comparison, lookup = st.columns([1, 1.2])
    with comparison:
        st.subheader("ESOL estimate comparison")
        view = st.radio("Comparison view", ["Agreement", "Signed error"], horizontal=True,
                        key="estimate_view", label_visibility="collapsed")
        y_column = ERROR if view == "Signed error" else ESTIMATE
        estimate_plot = px.scatter(
            selected, x=MEASURED, y=y_column, custom_data=["Compound", ERROR, ESTIMATE],
            labels={ERROR: SIGNED_ERROR},
            color_discrete_sequence=[TEAL], render_mode="svg",
        )
        estimate_plot.update_xaxes(range=LOG_S_RANGE, constrain="domain", **LOG_S_TICKS)
        reference = dict(dash="dot", color="#263D53")
        if view == "Signed error":
            limit = int(np.ceil(data[ERROR].abs().max()))
            estimate_plot.update_yaxes(range=[-limit, limit], zeroline=False, **LOG_S_TICKS)
            estimate_plot.add_hline(y=0, line=reference, layer="below")
            interpretation = "Signed error is the supplied estimate minus the measurement. Positive values indicate overestimates."
        else:
            low, high = LOG_S_RANGE
            estimate_plot.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=reference, layer="below")
            estimate_plot.update_yaxes(range=LOG_S_RANGE, scaleanchor="x", scaleratio=1,
                                       constrain="domain", **LOG_S_TICKS)
            interpretation = "The diagonal marks exact agreement."
        show_scatter(estimate_plot, "estimate_points", MEASURED, y_column, row)
        st.caption(interpretation)
        if len(selected) == len(data):
            quartiles = pd.qcut(data[MEASURED], 4, labels=False)
            large_errors = data[ERROR].abs().gt(1)
            low = large_errors[quartiles.eq(0)].mean()
            middle = large_errors[quartiles.isin([1, 2])].mean()
            high = large_errors[quartiles.eq(3)].mean()
            st.caption(f"Estimates differ by more than 1 log S in {low:.0%} of the least soluble quarter, "
                       f"{middle:.0%} of the middle half, and {high:.0%} of the most soluble quarter.")

    with lookup:
        st.subheader("Compound details")
        st.selectbox("Compound", compounds, index=None, placeholder="Search by compound name", key="compound")
        if row is not None:
            st.write(f"**Measured:** {row[MEASURED]:.2f} log S · **Supplied estimate:** {row[ESTIMATE]:.2f} log S")
            st.write(f"**Estimate error:** {row[ERROR]:+.2f} log S")
            molecule = Chem.MolFromSmiles(row["smiles"])
            if molecule is not None:
                st.image(Draw.MolToImage(molecule, size=(380, 200)), width=320)
            with st.expander("Molecular properties and SMILES"):
                for label in PROPERTIES:
                    st.write(f"{label}: {row[label]:.4g}")
                st.code(row["smiles"], language=None)

st.caption("Source: supplied Delaney ESOL data · All measurement records are included, including shared SMILES strings.")
