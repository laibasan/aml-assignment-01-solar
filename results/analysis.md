# Task 5 analysis

All numbers below were produced by `python src/train_eval.py` on the Plant 1 hourly table merged with Open-Meteo.

## Location / time-zone check (Task 3.4)

Verification days 16–18 May 2020, peak hour of sensor irradiation vs Open-Meteo shortwave (converted to kW/m²):

- 2020-05-16: sensor 12, Open-Meteo 12 (shift 0)
- 2020-05-17: sensor 11, Open-Meteo 12 (shift +1)
- 2020-05-18: sensor 13, Open-Meteo 12 (shift −1)

Sunrise and sunset on figure 05 line up; Open-Meteo timezone is GMT+5:30. Across all 34 days the most common peak shift is 0 (15 days). The ±1 hour argmax differences are from different midday shapes (clouds), not a systematic clock error, so the starting coordinates 14.82, 78.28 were kept.

## 5.1 Set A normal-equation weights

```
theta0_intercept = 6856.387167
theta1_irradiation = 8268.070799
theta2_module_temp = -31.159144
theta3_ambient_temp = -15.158485
theta4_sin_hour = -30.795558
theta5_cos_hour = -390.635565
```

Largest-magnitude feature weight (excluding the intercept): **irradiation** with value **8268.070799**.

Sign check against the Task 2 physics (dataset observations first, then interpretation):

- Observation: AC power rises with irradiation in `01_ac_vs_irradiation.png` (correlation 0.997).
- Interpretation: more sunlight on the panels should increase generated power.
- Conclusion: the irradiation coefficient is **positive**, which **agrees with** that observation.

- Observation: module temperature is higher when irradiation is higher (`02_module_vs_ambient.png`).
- Interpretation: after irradiation is already in the model, a hotter module often means lower electrical efficiency, so the module-temperature weight can be negative even though hot panels and high power occur together in raw scatter plots.
- Conclusion: the module-temperature coefficient is **negative**.

- Observation: ambient temperature also rises during the day with sunlight.
- Interpretation: it is correlated with irradiation, so its fitted sign after scaling is a partial effect, not a simple bivariate slope.
- Conclusion: the ambient-temperature coefficient is **negative**.

## 5.2 Set B vs Set A (daytime RMSE)

- Set A daytime RMSE: **727.0394 kW**
- Set B daytime RMSE: **3426.6740 kW**
- Difference (B minus A): **2699.6346 kW**
- As a percentage of peak hourly AC power in training (27325.90 kW): **9.88%**
- Set A daytime RMSE as a fraction of peak: **2.66%**

Public weather is clearly worse than on-site sensors, but the extra daytime error is still a modest fraction of peak hourly power. For a rooftop installer who only needs a rough generation estimate, Open-Meteo can be usable. It is not a replacement for on-site sensors when you need tight operational accuracy.

## Learning rates (Task 4.7, Set A training rows only)

Batch GD, 500 iterations:

- alpha=1e-5: cost fell (about 3.72e10 -> 5.93e8) but was still far above the normal-equation cost after 500 steps. **Too small** for this budget.
- alpha=1e-4: cost fell to about 1.30e8 without diverging. **Approximately appropriate**; this value was used afterwards.
- alpha=1e-3: cost exploded to a non-physical value. **Too large**.

SGD, 50 epochs:

- alpha=1e-4: still decreasing slowly at epoch 50 (end J about 5.95e8). **Too small**.
- alpha=1e-3: dropped quickly then flattened (end J about 1.31e8). **Approximately appropriate**.
- alpha=1e-2: fastest drop, lowest J after 50 epochs (about 1.06e8) and did not diverge. **Approximately appropriate**; chosen for the remaining SGD fits.

Set A irradiation, module temperature and ambient temperature are highly correlated, so batch GD at alpha=1e-4 still needs many more than 500 iterations to match the normal-equation theta (50,000 iterations were used for the final batch fits).

## 5.3 Solver comparison

- Normal equation: closed form, one matrix solve, no learning rate.
- Batch GD: 50000 iterations, alpha=0.0001, max |theta_GD - theta_normal| = 0.000115
- SGD: 50 epochs, alpha=0.01, max |theta_SGD - theta_normal| = 858.986819

Did they reach the same theta? Batch GD **yes, to at least two decimal places**. SGD **did not match the normal-equation vector as closely as batch GD**.

SGD's Set B test RMSE is slightly **lower** than the normal equation even though its theta is not the least-squares solution. That is a test-week accident after clipping, not evidence that SGD minimised J better on the training set.

For this dataset (hundreds of hourly rows, five features plus intercept) the normal equation is the practical choice: it is exact for this least-squares problem and cheap at this size.

For a dataset with 10 million rows, forming and inverting X^T X is still only (d+1) x (d+1), but each iteration that uses every row becomes expensive. SGD (or mini-batch GD) would be the scalable option because it updates from one row (or a small batch) at a time. That matches the usual optimization decision for least squares: normal equation for small/medium n and small d; gradient methods when n is huge or when you cannot form the full design matrix comfortably.

## 5.4 Batch vs stochastic cost curves

- Batch J(theta) is **smoother** because every update uses all training rows, so the direction is the full gradient of J.
- SGD J(theta) is **noisier** because each update uses a single row, so the path jitters around the true gradient.

## 5.5 Residuals vs hour of day (Set A normal equation)

Worst hour by mean absolute residual: **10:00** (mean |residual| = 1171.79 kW).

The largest mean absolute residuals fall in late morning / midday, when AC power is highest. Figure 09 also shows a pattern: residuals are mostly positive from about 08:00 to 16:00 (the model under-predicts) and negative near sunrise/sunset (it over-predicts). A linear model with instantaneous irradiation cannot capture sun-angle, thermal lag, or brief cloud transients that matter most while the plant is producing.
