# How much solar-forecast accuracy do you lose without on-site sensors?

*AI4003 Applied Machine Learning — Assignment 01. Least-squares regression implemented by hand (normal equation, batch GD, SGD). No scikit-learn.*

A grid-connected solar plant already knows how much power it is making. A rooftop installer usually does not. They have a public weather feed, not a pyranometer on the array. This project asks one practical question:

**How much prediction accuracy is lost when on-site irradiation and temperature sensors are replaced by free Open-Meteo weather?**

We built a linear model of Plant 1 hourly AC power from the [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) Kaggle set (15 May–17 June 2020) and answered that question with numbers from a chronological test week, not from a shuffled split.

## The data, and why timestamps matter

Plant 1 reports generation **per inverter** every 15 minutes and weather from **one** sensor logger. The two files do not even use the same date string: generation looks like `15-05-2020 00:00` (day-first) and the sensor file looks like `2020-05-15 00:00:00`. If you parse the first file as month-first, 15 May becomes 5 March and every later merge is wrong.

After summing AC and DC across inverters and merging on timestamp, **26** timestamps existed in only one file (1 generation-only, 25 sensor-only). Hourly means gave the expected **816** slots (34 days × 24 hours). **20** of those hours had missing values. Short gaps were filled with time interpolation (at most two hours); remaining incomplete rows were dropped. **804** hourly rows were saved.

One dataset quirk is worth stating plainly. Physics says AC should be a little below DC (inverter losses). In this Plant 1 file the daytime AC/DC ratio is about **0.098**. That is not a 10% inverter. DC is almost certainly on a different scale (~10×). We still predict **AC power**; DC is not a feature.

Average AC by hour of day looks like a solar plant: near zero at night, rising after 06:00, peaking at **11:00** (mean about 21,028 kW), then falling through 18:00. AC vs irradiation is almost a straight line (correlation **0.997**). Module temperature sits close to ambient in the dark and runs 20–30 °C hotter under strong sun.

## Public weather, and a location check

Open-Meteo was requested in Python with `requests` (no browser paste) for latitude **14.82**, longitude **78.28**, timezone **Asia/Kolkata**, 15 May–17 June 2020: shortwave radiation, 2 m temperature, and cloud cover. **816** hourly rows came back. Sensor irradiation is in **kW/m²**; Open-Meteo shortwave is in **W/m²**. Plots divide by 1000 to overlay them; the Set B model keeps W/m² and lets training-set scaling handle the magnitude.

The assignment says: if the two radiation curves peak an hour or more apart, the coordinates or timezone are wrong. For 16–18 May 2020 the peak hours were 12 vs 12, 11 vs 12, and 13 vs 12. Sunrise and sunset still line up, and the API timezone is GMT+5:30. Across all 34 days the most common peak shift is **0**. The ±1 hour argmax on two of the three check days is a cloud-shaped midday, not a clock error. We kept the starting coordinates.

## Two feature sets, three solvers

Training is **15 May–10 June** (636 hours). Testing is **11–17 June** (168 hours). No shuffling.

- **Set A** (on-site): irradiation, module temperature, ambient temperature, plus `sin(2πh/24)` and `cos(2πh/24)`.
- **Set B** (public weather only): shortwave radiation, 2 m temperature, cloud cover, plus the same hour features.

Every feature is scaled with **training** mean and standard deviation only. Those same numbers are applied to the test set and later to the Flask app. An intercept column of ones is added **after** scaling.

The hypothesis and cost are the lecture-1 least-squares pair:

`h(X) = X θ`, `J(θ) = ½ Σ (h − y)²`

θ is found three ways, all in NumPy: the normal equation `θ = (XᵀX)⁻¹ Xᵀy`, batch gradient descent, and stochastic gradient descent. Negative predictions are clipped to zero. Test error is RMSE in kW, reported on **all** test hours and on **daytime** hours (`irradiation > 0`). Night hours are easy zeros; they make every model look better than it is.

Learning rates were chosen on **Set A training cost only**. For batch GD (500 iterations): `10⁻⁵` was too small, `10⁻⁴` was usable, `10⁻³` exploded. For SGD (50 epochs): `10⁻⁴` was too small; `10⁻³` and `10⁻²` both worked; `10⁻²` reached the lowest training J and was kept.

Set A features are highly correlated (irradiation vs module temperature ≈ 0.97). Batch GD at `α = 10⁻⁴` therefore needs far more than 500 steps to match the normal equation. After **50,000** iterations the largest absolute θ difference was **0.00012**. SGD after 50 epochs did **not** reach the same vector (largest gap ≈ **859**).

## Test-week RMSE (kW)

| Solver | Features | All hours | Daytime only |
| --- | --- | ---: | ---: |
| Normal equation | Set A | 556.4 | **727.0** |
| Batch GD (α=10⁻⁴, 50,000 iters) | Set A | 556.4 | 727.0 |
| SGD (α=10⁻², 50 epochs) | Set A | 563.2 | 737.2 |
| Normal equation | Set B | 2633.0 | **3426.7** |
| Batch GD (α=10⁻⁴, 50,000 iters) | Set B | 2633.0 | 3426.7 |
| SGD (α=10⁻², 50 epochs) | Set B | 2475.2 | 3135.5 |

Training peak hourly AC was **27,326 kW**. Set A daytime RMSE is **2.66%** of that peak, inside the assignment’s “small fraction / roughly 10% or less” range for a correct on-site model. Set B is clearly worse: **2,700 kW** extra daytime error, **9.88%** of peak.

SGD’s Set B test RMSE is slightly *lower* than the normal equation even though its θ is not the least-squares solution. That is a test-week accident after clipping, not proof that SGD minimised J better on the training set.

## What the weights say

Set A normal-equation weights (scaled features): intercept 6856, irradiation **+8268** (largest), module temperature **−31**, ambient temperature **−15**, sin hour −31, cos hour −391.

Irradiation’s positive sign matches the scatter plot. The negative temperature weights are **partial** effects after irradiation is already in the model: a hotter module, holding sunlight fixed, is consistent with lower electrical efficiency. They are not a claim that “hot days produce less power” in the raw bivariate sense.

Residuals of the Set A normal-equation model are worst around **10:00** (mean |residual| ≈ 1,172 kW). From about 08:00 to 16:00 the model tends to under-predict; near sunrise and sunset it over-predicts. Instantaneous irradiation cannot capture sun angle, thermal lag, or brief clouds.

On the test-week time series, Set A tracks the daily peaks. Set B follows the daylight envelope but misses cloud dips and several peak hours — exactly what you expect when the weather is a grid cell, not a sensor on the glass.

## Answer for a rooftop installer

Public weather is **not** as good as on-site sensors. Replacing them costs about **2,700 kW** of extra daytime RMSE, or about **10% of this plant’s peak hourly AC**.

For a rooftop installer who only needs a rough generation shape from a free API, Open-Meteo is usable. It is **not** a replacement when you need operational accuracy close to the sensor model.

For this dataset size, use the **normal equation**. Use SGD when you have millions of rows. The Flask app loads saved Set B weights and does not retrain.

Code, tables, and figures: see the GitHub repository linked from the project README. Replace this sentence with your public repo URL after you push.
