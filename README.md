# Manufacturing Process Health and Operational Efficiency Analysis

**Unified Mentor internship project** — exploratory manufacturing analytics, statistical investigation, anomaly screening, forecasting evaluation and an interactive Streamlit dashboard.

## Project scope

This project analyses the supplied manufacturing CSV at `data/raw/Thales_Group_Manufacturing.csv`. It covers:

- Dataset validation, timestamp construction and duplicate machine/timestamp investigation.
- Machine-level temperature, vibration and power-consumption profiles and variability.
- Production speed, defect rate and error rate diagnostics by machine, time and operation mode.
- Efficiency-status distribution and operation-mode comparisons.
- Exploratory sensor deviation scoring and unsupervised anomaly screening.
- Production-speed forecasting experiments at 30-, 60- and 120-minute horizons, evaluated against simple baselines.
- A six-tab Streamlit dashboard with machine, mode, date and time filters, metric comparison, and anomaly CSV download.
- A research paper, executive summary, selected figures and supporting evidence tables.

## Important interpretation and provenance limits

The assignment brief frames the work in a Thales/6G smart-factory context. The supplied CSV and this analysis do **not** independently verify that the observations were collected at a Thales facility or that a 6G network was experimentally evaluated. Network latency and packet-loss columns are analysed as supplied measurements; they are not proof of a specific network generation or deployment.

`Efficiency_Status` is supplied in the CSV and its original construction rule is undocumented. `Predictive_Maintenance_Score` is also supplied without its generating model or validation evidence. Neither should be treated as independently verified ground truth. No authoritative machine-specific engineering limits or confirmed maintenance-failure labels were supplied.

The dashboard's `Exploratory_Sensor_Deviation_Index` is a relative percentile of robust sensor deviations. It is **not** a validated Machine Health Index, failure probability, safety score or diagnosis. Anomaly flags are investigation candidates, not confirmed faults. Forecasting models did not demonstrate reliable improvement over simple baselines in the evaluated chronological tests. No trained model artifact is included in `models/`.

## Dataset

Expected file: `data/raw/Thales_Group_Manufacturing.csv`.

The packaged CSV contains **100,000 records, 14 columns, 50 machine identifiers and three operation modes** (`Active`, `Idle`, `Maintenance`). The parsed observation period is **1 January 2025 through 10 March 2025 (69 calendar days)**. The raw file is preserved; derived files are stored under `data/processed/` and `reports/`.

The dataset has no explicit shift field. Therefore, shift-based comparisons cannot be claimed as completed without a documented shift schedule or a defensible shift definition.

## Requirements and setup (Windows PowerShell)

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the dashboard

From the project root:

```powershell
streamlit run dashboard/app.py
```

Dashboard sections:

1. **Overview** — supplied efficiency distribution, operation-mode summary and optional metric comparison.
2. **Machine Analysis** — machine-level production and sensor summaries.
3. **Quality Analysis** — defect/error comparisons and production-versus-defect scatter analysis.
4. **Sensor & Network Analysis** — sensor distributions, exploratory sensor deviation and network summaries.
5. **Anomaly Detection** — anomaly counts, rates, comparisons and downloadable filtered records.
6. **Data Quality** — missingness, duplicate timestamp and observation-interval limitations.

Sidebar capabilities include machine selection, operation-mode selection, calendar-date range, time-of-day range and an optional two-metric comparison explorer. The comparison is descriptive and does not establish causality.

## Reproduce the analysis

Run individual analysis scripts from the project root after installing requirements. The principal entry points include:

```powershell
python src/data_validation.py
python src/eda_analysis.py
python src/machine_health_analysis.py
python src/machine_sensor_stability.py
python src/production_quality_analysis.py
python src/production_quality_relationships.py
python src/production_quality_trends.py
python src/operation_mode_statistical_tests.py
python src/machine_anomaly_detection.py
python src/build_forecasting_dataset.py
python src/evaluate_forecast_baseline.py
```

Additional audit scripts in `src/` investigate data-generating patterns, efficiency-label provenance, timestamp intervals, forecasting feasibility, feature leakage and anomaly quality associations. They are separate diagnostic stages; consult each script's comments and output paths before rerunning it. Some scripts regenerate files in `data/processed/` or `reports/`.

## Key KPI definitions

- **Average Production Speed:** arithmetic mean of `Production_Speed_units_per_hr` over the selected records.
- **Defect Density Score:** production-weighted mean of the supplied defect-rate percentage: `sum(speed × defect_rate) / sum(speed)`. This weights each record by its reported production speed; it is not a count of defective units because the dataset does not provide unit-level defect counts or exposure duration for each record.
- **Error Frequency Index:** production-weighted mean of the supplied error-rate percentage, using the same weighting formula. It is a descriptive rate, not an event count.
- **Efficiency Distribution:** record counts and percentages for the supplied `Efficiency_Status` values; the source label is not independently validated.
- **Exploratory Sensor Deviation Index:** percentile rank of the average robust deviation across temperature, vibration and power, relative to the selected reference dataset. Higher values mean more unusual sensor combinations, not poorer validated machine health.

No validated composite Machine Health Index can be calculated from the supplied fields alone because authoritative operating limits, sensor calibration and failure outcomes are absent. The exploratory deviation index is reported instead, with its limitation explicit.

## Project structure

```text
├── dashboard/                 Streamlit application
├── data/
│   ├── raw/                    Original supplied CSV
│   └── processed/              Derived datasets and model inputs
├── models/                     Reserved for model artifacts (none included)
├── reports/
│   ├── final/
│   │   ├── figures/             Selected static PNG figures
│   │   ├── tables/              Selected supporting CSV evidence
│   │   ├── research_paper.md
│   │   └── executive_summary.md
│   └──                              Full analysis outputs and interactive charts
├── src/                        Validation, EDA, KPI, forecasting and anomaly scripts
├── tests/                      Automated project and visualization checks
├── requirements.txt
└── README.md
```

## Validation and limitations

The project includes automated tests under `tests/`. Run them with:

```powershell
python -m pytest -q
```

The test suite is intentionally small and does not constitute full end-to-end validation of every dashboard interaction or every research conclusion. The separate chart-generation smoke check can be run with `python tests/test_kpi_visualizations.py`. The dashboard was manually exercised during project development; repeat those checks in the target environment after dependency installation.

Results are exploratory and descriptive. Correlation does not establish causation; repeated machine observations may not be independent; irregular observation intervals affect time-based interpretation; and the supplied target labels require provenance checks. See `reports/final/research_paper.md` for methods, results, limitations and supporting evidence.
