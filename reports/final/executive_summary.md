# Executive Summary

## Manufacturing Process Health and Operational Efficiency Analysis

### Project Overview

This project examines a supplied manufacturing dataset attributed in the internship brief to Thales Group. The objective was to explore machine operating conditions, production performance, quality indicators, network measurements, and potential operational anomalies in a smart-factory context.

The analysis used 100,000 records covering 50 machines, three operation modes, and 69 calendar days, from 1 January 2025 to 10 March 2025.

The work included data-quality assessment, exploratory data analysis, machine-level comparisons, production and quality analysis, production-weighted rate KPIs, forecasting experiments, anomaly screening, and an interactive Streamlit dashboard.

### Key Findings

**1. Data quality and reliability**

- The dataset contained 100,000 records and 14 original columns.
- No missing cells or fully duplicated rows were identified.
- Twenty-eight duplicate machine-timestamp keys were detected, involving 56 records.
- The duplicate records were documented rather than silently removed from the original dataset.
- The origin and calculation rules of some supplied fields, including `Efficiency_Status` and `Predictive_Maintenance_Score`, could not be independently verified.

**2. Machine health and sensor behaviour**

- Machine-level sensor analysis did not identify observations beyond the selected machine-specific IQR thresholds.
- This does not establish that all machines were healthy; it only means the chosen statistical screening method did not flag those observations.
- Machine-level sensor variation and daily trends were examined. The observed trends were descriptive and were not sufficient to establish machine deterioration.

**3. Production and quality performance**

Average production speed was approximately 275.92 units per hour. The unweighted mean defect rate was 5.01%, and the unweighted mean error rate was 7.50%. When weighted by reported production speed, the defect-rate KPI was 5.00% and the error-frequency KPI was 7.52%. These are weighted percentages, not observed defective-unit or error-event counts.

The three operation modes had similar average production and quality measurements:

| Operation mode | Records | Mean production speed | Mean error rate |
|---|---:|---:|---:|
| Active | 70,054 | 276.20 | 7.49% |
| Idle | 20,057 | 274.13 | 7.52% |
| Maintenance | 9,889 | 277.54 | 7.54% |

The machine-blocked Friedman tests did not find statistically significant differences between operation modes for production speed, error rate, or defect rate. The measured effect sizes were also very small.

**4. Predictive modelling**

Production-speed forecasting was tested at 30-minute, 60-minute, and 120-minute horizons.

Random Forest and Ridge models were compared with simple global-mean, machine-mean, and persistence baselines. The trained models did not outperform the strongest simple baselines on the chronological test data.

The results do not support presenting production-speed forecasting as a successful predictive model for this dataset. The weak relationships between current and future measurements suggest that the available features contain limited forecasting signal.

**5. Anomaly screening**

An Isolation Forest model was trained using the reference period and evaluated on later observations.

- Reference period: 69,929 records, with 1,456 flagged anomalies (2.08%).
- Evaluation period: 30,071 records, with 1,141 flagged anomalies (3.79%).
- Overall: 2,597 flagged records (2.60%).

The evaluation-period anomaly rate was approximately 1.71 percentage points higher than the reference-period rate. This is a screening result, not proof of machine deterioration or an increase in actual equipment failures.

The flagged records did not show higher average defect or error rates than the unflagged records. The model's anomaly labels therefore require engineering review before operational use.

### Dashboard and Deliverables

An interactive Streamlit dashboard was developed to present:

- Overall manufacturing KPIs
- Machine-level performance comparisons
- Production and quality analysis
- Sensor and network measurements
- Anomaly screening results
- Data-quality observations and limitations
- Machine, operation-mode, calendar-date and time-of-day filters
- An optional two-metric comparison explorer

The project also includes analysis scripts, processed datasets, statistical outputs, visualizations, and forecasting evaluation results.

### Recommendations

1. Obtain authoritative documentation for the dataset, measurement units, target labels, and sensor definitions.
2. Investigate the duplicate machine-timestamp records and the change in sampling frequency observed on 1 March 2025.
3. Validate anomaly flags with maintenance logs, machine inspection records, and engineering feedback.
4. Collect longer and more representative time-series data before attempting production forecasting again.
5. Establish engineering-approved operating thresholds before interpreting sensor deviations as machine-health risks. The current sensor-deviation percentile is exploratory and is not a validated Machine Health Index.
6. Avoid using the supplied efficiency labels as independent ground truth until their generation rules are documented and validated.

### Overall Conclusion

The project provides an exploratory assessment of manufacturing process measurements, operational efficiency, data quality, and statistical anomaly screening. It demonstrates a reproducible analytical workflow and an interactive dashboard for investigating the supplied dataset.

However, the analysis did not establish reliable production forecasting, verified machine-failure prediction, or causal relationships between operating conditions and product quality. The results should be treated as exploratory findings rather than validated industrial performance claims.

**Important scope note:** The dataset was supplied for the internship and attributed in the brief to Thales Group. Its provenance and connection to an actual 6G-enabled production environment were not independently verified. The project therefore does not claim to demonstrate performance from a verified live 6G factory. The dataset does not include a shift field or documented shift schedule, so shift-based findings are not claimed.