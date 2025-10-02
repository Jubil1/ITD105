# student_performance_dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Student Exam Performance", layout="wide")
st.title("📊 Exploratory Data Analysis (EDA) — Student Exam Performance")

# --- Helper functions ---
@st.cache_data
def load_local(path: str = "student-mat.csv") -> pd.DataFrame:
    return pd.read_csv(path, sep=";")  # dataset uses semicolon separator

def load_uploaded(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, sep=";")

# --- Data source (sidebar) ---
st.sidebar.header("📂 Data source")
source = st.sidebar.radio("Load from:", ("Local file (student-mat.csv)", "Upload CSV file"))

df_orig = None
if source == "Upload CSV file":
    uploaded = st.sidebar.file_uploader("Upload student-mat.csv", type=["csv"])
    if uploaded is not None:
        try:
            df_orig = load_uploaded(uploaded)
            st.success(f"Uploaded file loaded — {df_orig.shape[0]} rows × {df_orig.shape[1]} columns")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
else:
    try:
        df_orig = load_local()
        st.success(f"Local file 'student-mat.csv' loaded — {df_orig.shape[0]} rows × {df_orig.shape[1]} columns")
    except FileNotFoundError:
        st.error("Local file 'student-mat.csv' not found. Put it in the same folder or use the uploader.")
    except Exception as e:
        st.error(f"Error loading file: {e}")

if df_orig is None:
    st.info("No dataset loaded yet. Use the sidebar to load student-mat.csv")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("🔎 Dataset Filter")
all_cols = df_orig.columns.tolist()
filter_col = st.sidebar.selectbox("Select a column to filter (or 'None'):", ["None"] + all_cols)

df = df_orig.copy()
if filter_col != "None":
    unique_count = df_orig[filter_col].nunique()
    if df_orig[filter_col].dtype == "object" or unique_count <= 20:
        options = sorted(df_orig[filter_col].dropna().unique().tolist())
        selected_vals = st.sidebar.multiselect(f"Select values for '{filter_col}':", options=options, default=options)
        if selected_vals:
            df = df[df[filter_col].isin(selected_vals)]
    else:
        lo, hi = st.sidebar.slider(f"Range for '{filter_col}':", float(df_orig[filter_col].min()), float(df_orig[filter_col].max()), (float(df_orig[filter_col].min()), float(df_orig[filter_col].max())))
        df = df[df[filter_col].between(lo, hi)]

st.sidebar.markdown(f"**Filtered shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# --- Main Tabs ---
tab_preview, tab_stats, tab_corr, tab_vis = st.tabs(["📑 Preview", "📊 Statistics", "🔥 Correlations", "📦 Visualizations"])

# --- Preview ---
with tab_preview:
    st.subheader("Dataset Preview")
    
    # Slider for row count
    n_rows_slider = st.slider(
        "Select number of rows to preview:",
        min_value=1,
        max_value=min(100, len(df)),
        value=5
    )
    
    # Number input for row count
    n_rows_input = st.number_input(
        "Or enter number of rows:",
        min_value=1,
        max_value=len(df),
        value=n_rows_slider,
        step=1
    )
    
    # Use whichever is last (number input overrides slider)
    n_rows = n_rows_input if n_rows_input != n_rows_slider else n_rows_slider

    st.dataframe(df.head(int(n_rows)))

    st.markdown("---")
    st.markdown("### Dataset Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().astype(int),
        "Non-Null Count": df.notnull().sum().astype(int)
    })
    st.dataframe(info_df)
    
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        st.success("No missing values found ✅")
    else:
        st.warning(f"Total missing values in dataset: {total_missing}")

# --- Statistics ---
with tab_stats:
    st.subheader("Summary Statistics")
    st.markdown("**Numeric Columns**")
    st.dataframe(df.describe().T)
    st.markdown("**Categorical Columns**")
    st.dataframe(df.describe(include=["object"]).T)

# --- Correlations ---
with tab_corr:
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, ax=ax)
    st.pyplot(fig)

# --- Visualizations ---
with tab_vis:
    st.subheader("Interactive Visualizations")
    subtab1, subtab2, subtab3, subtab4 = st.tabs(["📈 Scatter Plot", "📊 Bar Chart", "🔗 Pair Plot", "📦 Boxplot"])

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Scatter Plot
    with subtab1:
        st.markdown("#### Scatter Plot")
        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_axis = st.selectbox("X-axis:", numeric_cols, index=numeric_cols.index("G1") if "G1" in numeric_cols else 0)
            with col2:
                y_axis = st.selectbox("Y-axis:", numeric_cols, index=numeric_cols.index("G3") if "G3" in numeric_cols else 1)
            with col3:
                color_col = st.selectbox("Color by:", ["None"] + categorical_cols)

            fig = px.scatter(df, x=x_axis, y=y_axis,
                             color=None if color_col == "None" else color_col,
                             opacity=0.7, title=f"{y_axis} vs {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Scatter plot needs at least 2 numeric columns.")

    # Bar Chart
    with subtab2:
        st.markdown("#### Average by Category")
        if categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Categorical (X-axis):", categorical_cols)
            with col2:
                y_axis = st.selectbox("Numeric (Y-axis):", numeric_cols)

            bar_data = df.groupby(x_axis, as_index=False)[y_axis].mean()
            bar_fig = px.bar(bar_data, x=x_axis, y=y_axis, color=x_axis, title=f"Average {y_axis} by {x_axis}")
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("Need at least one categorical and one numeric column.")

    # Pair Plot
    with subtab3:
        st.markdown("#### Pair Plot")
        if len(numeric_cols) >= 2:
            selected_num_cols = st.multiselect("Numeric columns:", numeric_cols, default=["G1","G2","G3"] if all(c in numeric_cols for c in ["G1","G2","G3"]) else numeric_cols[:2])
            hue_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)

            if len(selected_num_cols) >= 2:
                if hue_col != "None":
                    pairplot_fig = sns.pairplot(df[selected_num_cols + [hue_col]], hue=hue_col, diag_kind="kde")
                else:
                    pairplot_fig = sns.pairplot(df[selected_num_cols], diag_kind="kde")
                st.pyplot(pairplot_fig)
            else:
                st.warning("Select at least 2 numeric columns.")
        else:
            st.info("Not enough numeric columns for pair plot.")

    # Boxplot
    with subtab4:
        st.markdown("#### Boxplot of Numeric Features")
        if numeric_cols:
            feature = st.selectbox("Select numeric feature:", numeric_cols)
            fig = px.box(df, y=feature, points="all", title=f"Boxplot of {feature}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for boxplot.")
