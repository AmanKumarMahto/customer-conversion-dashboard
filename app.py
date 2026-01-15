import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Customer Conversion Dashboard", page_icon="📊", layout="wide")

st.title("📊 Customer Conversion & Segmentation Dashboard")
st.caption("Made by Aman Kumar Mahto | Business & Data Analytics")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/marketing_data.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

# ----------------------------
# Helpers
# ----------------------------
def exists(col):
    return col in df.columns

def conversion_rate(frame):
    if "Output" not in frame.columns or len(frame) == 0:
        return 0.0
    return (frame["Output"].astype(str).str.lower() == "yes").mean() * 100

def safe_unique(col):
    return sorted(df[col].dropna().astype(str).unique().tolist()) if exists(col) else []

def apply_filter(frame, col, chosen):
    if col in frame.columns and chosen:
        return frame[frame[col].astype(str).isin(chosen)]
    return frame

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔎 Filters")

if st.sidebar.button("🔄 Reset Filters"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

filter_df = df.copy()

filters = [
    ("Gender", "Gender"),
    ("Occupation", "Occupation"),
    ("Monthly Income", "Monthly Income"),
    ("Educational Qualifications", "Education"),
    ("Marital Status", "Marital Status"),
    ("Family size", "Family Size"),
]

selected_filters = {}

for col_name, label in filters:
    if col_name in df.columns:
        values = sorted(df[col_name].dropna().astype(str).unique().tolist())
        chosen = st.sidebar.multiselect(label, values, default=values, key=f"filter_{col_name}")
        selected_filters[col_name] = chosen
        if chosen:
            filter_df = filter_df[filter_df[col_name].astype(str).isin(chosen)]

st.sidebar.markdown("---")
st.sidebar.write(f"✅ Records after filter: **{len(filter_df):,}**")


# ----------------------------
# Data Quality Panel
# ----------------------------
with st.sidebar.expander("📌 Data Quality"):
    missing_pct = (filter_df.isna().mean() * 100).sort_values(ascending=False).head(8)
    st.write("Top missing columns (%)")
    st.dataframe(missing_pct.reset_index().rename(columns={"index": "Column", 0: "Missing %"}), use_container_width=True)

# ----------------------------
# KPIs
# ----------------------------
st.markdown("### ✅ Key Metrics")

total = len(filter_df)
yes_count = (filter_df["Output"].astype(str).str.lower() == "yes").sum() if exists("Output") and total > 0 else 0
no_count = (filter_df["Output"].astype(str).str.lower() == "no").sum() if exists("Output") and total > 0 else 0
conv_rate = conversion_rate(filter_df) if exists("Output") else 0.0
overall_rate = conversion_rate(df) if exists("Output") else 0.0
delta = conv_rate - overall_rate

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📌 Total Responses", f"{total:,}")
c2.metric("✅ Conversions (Yes)", f"{yes_count:,}")
c3.metric("❌ Non-Conversions (No)", f"{no_count:,}")
c4.metric("📈 Conversion Rate", f"{conv_rate:.2f}%")
c5.metric("📊 vs Overall Avg", f"{delta:+.2f}%")

st.divider()

# ----------------------------
# Dynamic Segment Comparison
# ----------------------------
st.markdown("### 🔥 Segment Comparison (Company View)")

segment_options = []
for col, _ in filters:
    if exists(col):
        segment_options.append(col)

default_segment = "Occupation" if "Occupation" in segment_options else (segment_options[0] if segment_options else None)

compare_col = st.selectbox("Compare Conversion Rate by:", segment_options, index=segment_options.index(default_segment) if default_segment in segment_options else 0)

# Conversion rate by segment
seg_summary = None
if compare_col and exists("Output") and total > 0:
    seg_summary = (
        filter_df.groupby(compare_col)["Output"]
        .agg(
            Total=("count"),
            Yes=lambda x: (x.astype(str).str.lower() == "yes").sum()
        )
        .reset_index()
    )
    seg_summary["Conversion Rate (%)"] = np.where(seg_summary["Total"] > 0, (seg_summary["Yes"] / seg_summary["Total"]) * 100, 0)
    seg_summary["Share (%)"] = np.where(total > 0, (seg_summary["Total"] / total) * 100, 0)
    seg_summary = seg_summary.sort_values("Conversion Rate (%)", ascending=False)

    top_seg = seg_summary.head(12)

    fig_seg = px.bar(
        top_seg,
        x=compare_col,
        y="Conversion Rate (%)",
        title=f"Top Segments by Conversion Rate ({compare_col})",
        text_auto=".2f"
    )
    st.plotly_chart(fig_seg, use_container_width=True)

st.divider()

# ----------------------------
# Two Column Insights
# ----------------------------
left, right = st.columns(2)

# Volume vs Conversion Scatter (actionable)
with left:
    st.markdown("### 🎯 Volume vs Conversion")
    if seg_summary is not None and len(seg_summary) > 0:
        scatter_df = seg_summary.copy()
        fig_scatter = px.scatter(
            scatter_df.head(30),
            x="Total",
            y="Conversion Rate (%)",
            size="Share (%)",
            hover_name=compare_col,
            title="High Volume + High Conversion = Best Target Segments",
            opacity=0.7
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Not enough data after filters to build scatter insights.")

# Geo Map (if lat/long exists)
with right:
    st.markdown("### 🗺 Geo Distribution")
    lat_col = "latitude" if "latitude" in df.columns else None
    lon_col = "longitude" if "longitude" in df.columns else None

    if lat_col and lon_col:
        geo_df = filter_df.dropna(subset=[lat_col, lon_col]).copy()
        if len(geo_df) > 0:
            # small sample for performance
            geo_df = geo_df.sample(min(2000, len(geo_df)), random_state=42)

            # color by Output if available
            if exists("Output"):
                fig_map = px.scatter_mapbox(
                    geo_df,
                    lat=lat_col,
                    lon=lon_col,
                    color="Output",
                    zoom=3,
                    height=420,
                    title="Customer Points (Filtered)"
                )
            else:
                fig_map = px.scatter_mapbox(
                    geo_df,
                    lat=lat_col,
                    lon=lon_col,
                    zoom=3,
                    height=420,
                    title="Customer Points (Filtered)"
                )

            fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geo rows available after filtering.")
    else:
        st.info("Latitude/Longitude columns not found.")

st.divider()

# ----------------------------
# Conversion Drivers Table
# ----------------------------
st.markdown("### 📌 Conversion Drivers (Top Segments Table)")

if seg_summary is not None and len(seg_summary) > 0:
    driver_table = seg_summary[[compare_col, "Total", "Yes", "Conversion Rate (%)", "Share (%)"]].head(15)
    st.dataframe(driver_table, use_container_width=True)

    # Download segment summary
    seg_csv = driver_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Segment Summary (CSV)",
        data=seg_csv,
        file_name="segment_summary.csv",
        mime="text/csv"
    )
else:
    st.info("Segment table not available (filters may be too strict).")

st.divider()

# ----------------------------
# Recommendations (Auto)
# ----------------------------
st.markdown("### ✅ Recommendations (Auto-Generated)")

rec1 = "Focus targeting on **high conversion + high volume segments** to maximize ROI."
rec2 = "Run experiments to improve conversion in **high volume but low conversion segments**."
rec3 = "Use geo clustering to identify **high-intent locations** and optimize localized campaigns."

if seg_summary is not None and len(seg_summary) > 0:
    best_seg = seg_summary.iloc[0][compare_col]
    rec1 = f"Focus targeting on top segment: **{best_seg}** (highest conversion in current filter)."

st.success(f"1) {rec1}\n\n2) {rec2}\n\n3) {rec3}")

# ----------------------------
# Download Filtered Data
# ----------------------------
st.markdown("### ⬇️ Download Filtered Dataset")

csv = filter_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Data (CSV)",
    data=csv,
    file_name="filtered_data.csv",
    mime="text/csv"
)

# ----------------------------
# Preview
# ----------------------------
with st.expander("📄 View Filtered Dataset (Preview)"):
    st.dataframe(filter_df.head(50), use_container_width=True)
