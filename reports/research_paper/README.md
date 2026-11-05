# Manufacturing Process Health and Operational Efficiency Analysis

An exploratory data analytics and machine-learning project completed for the Unified Mentor internship assignment.

The project examines machine-level operating conditions, production performance, quality indicators, temporal patterns, and unsupervised anomaly screening using the supplied manufacturing dataset.

## 1. Project scope

The analysis covers:

- Dataset validation and data-quality investigation
- Machine-level sensor and operational analysis
- Production, defect-rate, and error-rate analysis
- Operation-mode comparisons
- KPI calculations and visualizations
- Production forecasting experiments and baseline comparisons
- Efficiency-label reconstruction audit
- Unsupervised anomaly screening using Isolation Forest
- Interactive Streamlit dashboard
- Research paper and executive summary

**Important evidence limitation:** The assignment brief references Thales Group and 6G-enabled smart factories. The supplied CSV does not independently establish that its records were collected at real Thales facilities or that a 6G network was experimentally evaluated. Findings should therefore be interpreted as an exploratory analysis of the supplied dataset, not verified factory or 6G deployment results.

## 2. Dataset

Expected location:

`data/raw/Thales_Group_Manufacturing.csv`

The supplied dataset contains 100,000 records and 14 columns.

| Category | Fields |
|---|---|
| Identification and time | Date, Timestamp, Machine_ID, Operation_Mode |
| Machine sensors | Temperature_C, Vibration_Hz, Power_Consumption_kW |
| Network indicators | Network_Latency_ms, Packet_Loss_% |
| Quality and production | Quality_Control_Defect_Rate_%, Production_Speed_units_per_hr, Error_Rate_% |
| Existing indicators | Predictive_Maintenance_Score, Efficiency_Status |

The original CSV is preserved in `data/raw/`. Derived datasets and analysis outputs are stored separately.

## 3. Key analytical conclusions

The analysis found that:

- The dataset contains 50 machines and three operation modes.
- Missing values and fully duplicated rows were not detected in the initial data-quality audit.
- Duplicate machine-and-timestamp keys were detected and investigated rather than automatically deleted.
- Overall production, error-rate, and defect-rate relationships were negligible in the exploratory correlation analysis.
- Machine-level and operation-mode comparisons did not establish meaningful differences in the tested production and quality metrics.
- The production forecasting experiments did not outperform simple mean-based baselines on the chronological test sets.
- The supplied `Efficiency_Status` label can be reconstructed using candidate production-speed and error-rate rules, but its original formula and provenance are unverified.
- Isolation Forest flagged 2.597% of records overall. The flagged records should be treated as candidates for investigation, not confirmed machine failures.
- A change in observation frequency around March 1 was identified and documented.

These results are descriptive and exploratory. They do not establish causation, equipment failure, safety limits, or verified operational performance.

## 4. Technology stack

- Python
- Pandas and NumPy
- SciPy
- Scikit-learn
- Plotly
- Streamlit
- Matplotlib and Seaborn

Dependencies are listed in `requirements.txt`.

## 5. Setup

Run the following commands in Windows PowerShell from the project root.

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Run the dashboard

From the project root:

```powershell
streamlit run dashboard/app.py
```

The dashboard includes six sections:

1. Overview
2. Machine Analysis
3. Quality Analysis
4. Sensor & Network Analysis
5. Anomaly Detection
6. Data Quality

It supports machine, operation-mode, and date filtering. The anomaly section also provides a downloadable results dataset.

## 7. Project structure

```text
Thales_Manufacturing_ML_Starter/
│
├── data/
│   ├── raw/
│   │   └── Thales_Group_Manufacturing.csv
│   └── processed/
│       └── Derived analysis datasets
│
├── dashboard/
│   └── app.py
│
├── src/
│   └── Data validation, analysis, KPI,
│       forecasting, and anomaly scripts
│
├── reports/
│   ├── figures/
│   ├── final/
│   │   ├── figures/
│   │   ├── tables/
│   │   ├── research_paper.md
│   │   └── executive_summary.md
│   └── Supporting analysis outputs
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Important methodological limitations

- The CSV's collection process and real-world provenance are not independently verified.
- Engineering thresholds were not supplied by an authoritative machine specification.
- The existing maintenance score's generating model is undocumented.
- The existing efficiency label is not treated as an independently validated prediction target.
- Forecasting models did not demonstrate improvement over simple baselines.
- Isolation Forest results are screening outputs, not confirmed faults.
- Statistical tests and correlations are exploratory and should not be interpreted as causal evidence.
- Duplicate machine-and-timestamp records and the March 1 observation-frequency change are documented data-quality considerations.

## 9. Final deliverables

The `reports/final/` directory contains:

- `research_paper.md`
- `executive_summary.md`
- `figures/` — selected static figures
- `tables/` — supporting analytical CSV tables

The root `reports/` directory retains additional analysis outputs and interactive visualizations.

## 10. Reproducibility

Use the supplied original CSV and the scripts in `src/` to inspect and reproduce the analysis. The derived datasets and report tables are retained as evidence of the analytical workflow.

Do not interpret a successful dashboard launch as independent validation of the dataset, the model assumptions, or the conclusions. Re-run the relevant validation and analysis scripts when reproducing the work in a new environment.
