# Forecasting Model Evaluation and Results

## 1. Objective

Evaluate whether current manufacturing-process measurements can predict future production speed at 30-minute, 60-minute and 120-minute horizons.

The target variable is `Future_Production_Speed`.

## 2. Forecast dataset construction

Forecast datasets were constructed by matching each source observation to a future observation from the same machine.

| Horizon | Valid pairs | Machines | Reused target groups | Repeated source groups |
|---|---:|---:|---:|---:|
| 30 minutes | 37,887 | 50 | 0 | 0 |
| 60 minutes | 37,211 | 50 | 0 | 0 |
| 120 minutes | 37,030 | 50 | 0 | 0 |

No invalid records were removed from the final datasets. The matching process used a tolerance around the requested forecast horizon, so actual lead times may differ slightly from the nominal horizon.

## 3. Evaluation methodology

A chronological train/validation/test evaluation was used rather than a random row split.

The following baselines were evaluated:

- **Global mean:** predicts the training-set average production speed.
- **Machine mean:** predicts the training-set average for the corresponding machine.
- **Persistence:** predicts future production speed using the current production speed.

A Random Forest regression pipeline was also evaluated using machine identity, operation mode and available current process/sensor measurements.

Performance was measured using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R²)

## 4. Baseline test results

| Horizon | Baseline | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|
| 30 min | Global mean | 112.893 | 130.097 | -0.00001 |
| 30 min | Machine mean | 112.849 | 130.035 | 0.00094 |
| 30 min | Persistence | 151.866 | 186.375 | -1.05234 |
| 60 min | Global mean | 113.133 | 130.397 | approximately 0 |
| 60 min | Machine mean | 113.171 | 130.476 | -0.00122 |
| 60 min | Persistence | 151.690 | 185.763 | -1.02948 |
| 120 min | Global mean | 113.314 | 130.311 | -0.00007 |
| 120 min | Machine mean | 113.425 | 130.431 | -0.00192 |
| 120 min | Persistence | 150.498 | 185.113 | -1.01809 |

The mean baselines achieve R² values close to zero, indicating that they explain almost none of the variation in the test targets. Persistence performs substantially worse.

## 5. Random Forest results

| Horizon | Split | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| 30 min | Train | 70.363 | 82.119 | 0.59925 |
| 30 min | Validation | 112.583 | 130.198 | -0.01465 |
| 30 min | Test | 113.280 | 130.863 | -0.01182 |
| 60 min | Train | 70.481 | 82.249 | 0.59938 |
| 60 min | Validation | 112.923 | 130.662 | -0.01336 |
| 60 min | Test | 113.768 | 131.383 | -0.01517 |
| 120 min | Train | 70.414 | 82.233 | 0.59887 |
| 120 min | Validation | 112.059 | 129.524 | -0.00910 |
| 120 min | Test | 113.793 | 131.088 | -0.01203 |

The Random Forest fits the training data substantially better than it performs on validation and test data. This indicates poor generalization.

It does not outperform the best mean baseline on any of the three test horizons.

## 6. Forecast-pair relationship analysis

The relationship between current production speed and its matched future production speed was evaluated using Pearson and Spearman correlation.

| Horizon | Pearson | Spearman | Mean machine Pearson |
|---|---:|---:|---:|
| 30 min | -0.00879 | -0.00879 | -0.00925 |
| 60 min | 0.00454 | 0.00449 | 0.00398 |
| 120 min | 0.00096 | 0.00099 | 0.00023 |

The correlations are effectively zero at all three horizons. Machine-level correlations also show no consistent positive relationship.

This indicates that current production speed alone provides almost no linear or monotonic information about its matched future value in these datasets.

## 7. Interpretation

The experiments did not establish a reliable relationship between the available current measurements and future production speed.

The Random Forest's training performance is not maintained on later observations. Its negative validation and test R² values, combined with test MAE slightly worse than the mean baselines, indicate that the fitted model is not suitable for reliable forecasting.

The poor persistence baseline also suggests that simply carrying the current production-speed value forward is not useful for these records.

The results do not prove that future production speed is inherently unpredictable. Other data, features, sampling strategies or modeling approaches may produce different results. However, the tested methods do not justify a forecasting deployment.

## 8. Limitations

1. The dataset's real-world provenance and factory-specific measurement context have not been independently verified.
2. The data is attributed to Thales Group in the supplied project brief; this analysis does not independently establish that the records are actual Thales factory measurements.
3. No verified machine-specific operating limits or failure thresholds were available.
4. Correlation analysis cannot rule out all nonlinear relationships.
5. The forecast evaluation covers the available observation period and cannot establish long-term performance.
6. The supplied dataset does not establish that measurements were collected from a live 6G-enabled manufacturing deployment.

## 9. Conclusion

The evaluated global mean, machine mean, persistence and Random Forest approaches did not demonstrate useful predictive performance for 30-minute, 60-minute or 120-minute production-speed forecasting.

The Random Forest showed substantial training-to-test performance degradation and failed to outperform the best simple baseline on every test horizon.

Therefore, the forecasting model is not recommended for operational use. The negative result is retained as an experimental finding rather than presenting an unsuccessful model as a successful predictive system.

Future work should prioritize obtaining verified process context, additional explanatory variables, reliable temporal sampling information and independently validated targets before attempting further forecasting experiments.