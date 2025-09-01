import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AI & DS Enrollment Forecast", page_icon="📈", layout="wide")
st.title("📈 AI & DS Enrollment Forecast Dashboard")
st.caption("Clean KPIs • Clear Forecast • Research context for Lebanon")

def read_csv_safe(p):
    try: return pd.read_csv(p)
    except: return None

def read_json_safe(p):
    try:
        with open(p) as f: return json.load(f)
    except:
        return None

# ---- Sidebar: allow uploads to override repo files
st.sidebar.header("Data Inputs")
hist = st.sidebar.file_uploader("Enrollment history (semester totals)", type=["csv"])
fc  = st.sidebar.file_uploader("Forecast numbers (Admit_Semester, Predicted_Enrollments)", type=["csv"])
met = st.sidebar.file_uploader("Forecast backtest metrics (JSON)", type=["json"])
cls = st.sidebar.file_uploader("Classification metrics (JSON)", type=["json"])
interest = st.sidebar.file_uploader("Lebanon AI/DS interest (Year,Learners)", type=["csv"])

hist = pd.read_csv(hist) if hist else read_csv_safe("enrollments_history_clean.csv")
forecast_df = pd.read_csv(fc) if fc else read_csv_safe("forecast_semester_numbers.csv")
backtest_metrics = json.load(met) if met else read_json_safe("forecast_backtest_metrics.json")
cls_metrics = json.load(cls) if cls else read_json_safe("classification_metrics.json")
interest = pd.read_csv(interest) if interest else read_csv_safe("ai_ds_interest_lebanon.csv")

# ---- KPIs (will show partial if forecast missing)
col1, col2, col3, col4 = st.columns(4)
if hist is not None and len(hist):
    last_actual = int(hist['Enrollments'].iloc[-1])
    col1.metric("Last Actual (students)", f"{last_actual:,}")
else:
    col1.metric("Last Actual (students)", "—")

if hist is not None and forecast_df is not None and len(forecast_df):
    next_sem = forecast_df['Admit_Semester'].iloc[0]
    next_enr = int(forecast_df['Predicted_Enrollments'].iloc[0])
    last_actual = int(hist['Enrollments'].iloc[-1]) if hist is not None and len(hist) else 0
    growth = ((next_enr - last_actual)/max(1,last_actual))*100 if last_actual else 0
    col2.metric(f"Next Semester ({next_sem})", f"{next_enr:,}", delta=f"{growth:+.1f}%")
else:
    col2.metric("Next Semester (forecast)", "—")

horizon = len(forecast_df) if forecast_df is not None else 0
col3.metric("Forecast Horizon", f"{horizon} semesters" if horizon else "—")

if backtest_metrics:
    col4.metric("Backtest MAPE", f"{backtest_metrics.get('MAPE_%',0):.1f}%")
else:
    col4.metric("Backtest MAPE", "—")

st.divider()

# ---- Trend chart
if hist is not None and len(hist):
    base = hist[['Admit_Semester','Enrollments']].copy()
    base['Type'] = 'Actual'
    if forecast_df is not None and len(forecast_df):
        f = forecast_df.rename(columns={'Predicted_Enrollments':'Enrollments'}).copy()
        f['Type'] = 'Forecast'
        data = pd.concat([base, f], ignore_index=True)
    else:
        data = base
    fig = px.line(data, x="Admit_Semester", y="Enrollments", color="Type", markers=True,
                  title="Enrollment Trend (Historical & Forecast)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No enrollment history yet. Upload 'enrollments_history_clean.csv' or add a semester mapping in Colab.")

# ---- Clean forecast numbers
if forecast_df is not None and len(forecast_df):
    st.subheader("Upcoming Enrollment (clean numbers)")
    show = forecast_df.rename(columns={"Admit_Semester":"Admit Semester",
                                       "Predicted_Enrollments":"Predicted Enrollments"})
    st.dataframe(show, use_container_width=True, hide_index=True)

# ---- Model metrics
st.subheader("Model Quality")
c1, c2 = st.columns(2)
if backtest_metrics:
    c1.markdown("**Forecast Backtest (hold-out)**")
    c1.write(pd.DataFrame([{
        "MAE": round(backtest_metrics["MAE"], 2),
        "RMSE": round(backtest_metrics["RMSE"], 2),
        "MAPE (%)": round(backtest_metrics["MAPE_%"], 2)
    }]))
else:
    c1.info("No forecast metrics yet.")

if cls_metrics:
    c2.markdown("**Classification (Final Status)**")
    c2.write(pd.DataFrame([{
        "Accuracy": round(cls_metrics["Accuracy"], 3),
        "Precision (macro)": round(cls_metrics["Precision_macro"], 3),
        "Recall (macro)": round(cls_metrics["Recall_macro"], 3),
        "F1 (macro)": round(cls_metrics["F1_macro"], 3)
    }]))
else:
    c2.info("No classification metrics file.")

# ---- Lebanon AI/DS context
with st.expander("📚 Lebanon AI & DS interest (context)", expanded=True):
    st.markdown("- Upload `ai_ds_interest_lebanon.csv` (Year, Learners) to visualize demand.")
    if interest is not None and {'Year','Learners'}.issubset(interest.columns):
        fig2 = px.bar(interest, x="Year", y="Learners", title="AI/DS Learners in Lebanon")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No interest file uploaded yet.")
