# student_performance.py
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="DAPAT PISO AMONG GRADO SIR", layout="wide")
st.title("Exploratory Data Analysis (EDA) — Student Exam Performance")

# --- Helper functions ---
@st.cache_data
def load_local(path: str = "student-mat.csv") -> pd.DataFrame:
    # NOTE: this dataset uses semicolon separators
    return pd.read_csv(path, sep=";")

def load_uploaded(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, sep=";")

# --- UI: choose source ---
st.sidebar.header("Data source")
source = st.sidebar.radio("Load from:", ("Local file (student-mat.csv)", "Upload CSV file"))

df = None
if source == "Upload CSV file":
    uploaded = st.sidebar.file_uploader("Upload student-mat.csv", type=["csv"])
    if uploaded is not None:
        try:
            df = load_uploaded(uploaded)
            st.success(f"Uploaded file loaded — {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
else:
    try:
        df = load_local()
        st.success(f"Local file 'student-mat.csv' loaded — {df.shape[0]} rows × {df.shape[1]} columns")
    except FileNotFoundError:
        st.error("Local file 'student-mat.csv' not found. Put the CSV in the same folder as this script or use the uploader.")
    except pd.errors.ParserError as pe:
        st.error(f"Parsing error: {pe}. Make sure the file uses ';' as the separator.")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# --- Display first few rows ---
if df is not None:
    st.subheader("First Few Rows of the Dataset")

    # Row Slider
    n_rows_slider = st.slider("Select number of rows to preview:",
                       min_value=1,
                       max_value=min(100, len(df)),
                       value=5
                       )
    
    # Row input
    n_rows_input = st.number_input (
        "Or enter number of rows:",
        min_value=1,
        max_value=len(df),
        value=n_rows_slider,
        step=1
    )

    # Use whichever is last
    n_rows = n_rows_input if n_rows_input != n_rows_slider else n_rows_slider

    st.dataframe(df.head(int(n_rows)))  # Interactive preview (first 5 rows only)

    # Extra info
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Shape:**", df.shape)
else:
    st.info("No dataset loaded yet.")

# --- Dataset Filters ---
if df is not None:
    st.sidebar.header("🔎 Dataset Filter")

    # Let user pick which column to filter
    filter_col = st.sidebar.selectbox("Selects a column to filter by:", df.columns)

    # Check if selected column is categorical or numeric
    if df[filter_col].dtype == "object" or df[filter_col].nunique() < 20:
        # For categorical or low-cardinality columns → multiselect
        filter_vals = st.sidebar.multiselect(
            f"Select values for {filter_col}:",
            options=df[filter_col].unique(),
            default=df[filter_col].unique()
        )
        df = df[df[filter_col].isin(filter_vals)]
    else:
        # For numeric columns → slider
        min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
        filter_range = st.sidebar.slider(
            f"Select range for {filter_col}:",
            min_value=min_val, max_value=max_val,
            value=(min_val, max_val)
        )
        df = df[df[filter_col].between(filter_range[0], filter_range[1])]

    # Show filtered shape
    st.sidebar.markdown(f"**Filtered dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")


# --- Dataset Information ---
if df is not None:
    st.subheader("Dataset Information")

    # Data types and non-null counts (similar to df.info())
    buffer = []
    df.info(buf=buffer) if False else None  # placeholder, Streamlit can't render .info() directly

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Non-Null Count": df.notnull().sum()
    })
    st.dataframe(info_df)

    # Quick missing values summary
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        st.success("No missing values found ✅")
    else:
        st.warning(f"Total missing values in dataset: {total_missing}")

# --- Summary Statistics ---
if df is not None:
    st.subheader("Summary Statistics")

    # Numeric columns summary
    st.markdown("**📊 Numeric Columns**")
    st.dataframe(df.describe().T)  # transpose for easier reading

    # Categorical columns summary
    st.markdown("**🔤 Categorical Columns**")
    st.dataframe(df.describe(include=["object"]).T)

# Heatmap
if df is not None:
    st.subheader("Correlation Heatmap (Numeric Features Only)")

    # Compute correlation matrix
    corr = df.corr(numeric_only=True)

    # Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr,
                annot=True,       # show correlation values
                fmt=".2f",        # decimal format
                cmap="coolwarm",  # color palette
                cbar=True,
                square=True,
                ax=ax)

    st.pyplot(fig)


# --- Boxplot Visualization ---
if df is not None:
    st.subheader("Boxplot — Numeric Features")

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_cols) > 0:
        # Sidebar selection
        feature = st.selectbox("Select a numeric feature for boxplot:", numeric_cols)

        # Plotly boxplot
        fig = px.box(df, y=feature, points="all", title=f"Boxplot of {feature}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns available for boxplot.")

# Interactive Scatter Plot
if df is not None:
    st.subheader("Interactive Scatter Plot — Student Performance")

    # Numeric columns for x/y selection
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Select X-axis feature:", numeric_cols, index=numeric_cols.index("G1") if "G1" in numeric_cols else 0)
        with col2:
            y_axis = st.selectbox("Select Y-axis feature:", numeric_cols, index=numeric_cols.index("G3") if "G3" in numeric_cols else 1)

        color_feature = st.selectbox("Color by (optional):", ["None"] + categorical_cols)

        # Build scatter plot
        if color_feature != "None":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_feature,
                             size_max=8, opacity=0.7,
                             title=f"Scatter Plot: {y_axis} vs {x_axis} (colored by {color_feature})")
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis,
                             size_max=8, opacity=0.7,
                             title=f"Scatter Plot: {y_axis} vs {x_axis}")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for scatter plot.")

# --- Interactive Bar Chart ---
if df is not None:
    st.subheader("Interactive Bar Chart")

    # Separate categorical and numeric columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if categorical_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Select X-axis (categorical):", categorical_cols, index=0)
        with col2:
            y_axis = st.selectbox("Select Y-axis (numeric):", numeric_cols, index=numeric_cols.index("G3") if "G3" in numeric_cols else 0)

        # Group and plot
        bar_data = df.groupby(x_axis, as_index=False)[y_axis].mean()
        bar_fig = px.bar(
            bar_data,
            x=x_axis,
            y=y_axis,
            title=f"Average {y_axis} by {x_axis}",
            color=x_axis
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.warning("Not enough categorical or numeric columns available for bar chart.")

# --- Interactive Pair Plot ---
if df is not None:
    st.subheader("Interactive Pair Plot")

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if len(numeric_cols) >= 2:
        selected_num_cols = st.multiselect(
            "Select numeric columns for pair plot:",
            numeric_cols,
            default=["G1", "G2", "G3"] if all(c in numeric_cols for c in ["G1", "G2", "G3"]) else numeric_cols[:2]
        )

        hue_col = st.selectbox(
            "Select categorical column for coloring (hue):",
            ["None"] + categorical_cols
        )

        if len(selected_num_cols) >= 2:
            # Build the pair plot
            if hue_col != "None":
                pair_fig = sns.pairplot(df[selected_num_cols + [hue_col]], hue=hue_col)
            else:
                pair_fig = sns.pairplot(df[selected_num_cols])

            st.pyplot(pair_fig)
        else:
            st.warning("Please select at least 2 numeric columns for pair plot.")
    else:
        st.warning("Not enough numeric columns for pair plot.")
