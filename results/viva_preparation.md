# Viva preparation (this implementation)

Short answers tied to the code in `src/`. Numbers are from the last `python src/train_eval.py` run.

## Why do timestamps need alignment?

Generation is per inverter; sensors are one row per plant. `prepare.py` parses `DATE_TIME` separately because generation uses `15-05-2020 00:00` and sensors use `2020-05-15 00:00:00`. Power is summed by timestamp, then merged with sensors on `datetime`. If you parse generation as month-first, 15 May becomes 5 March and the merge is wrong. 26 timestamps existed in only one file (1 generation-only, 25 sensor-only).

## Why hourly resampling?

Raw data is 15-minute. Open-Meteo is hourly. `resample("h").mean()` makes one row per hour so Set A and Set B share the same time grid. After resampling there were 816 hourly slots (34×24); 20 had missing values.

## Why is the train/test split chronological?

Train is 15 May–10 June; test is 11–17 June. There is no shuffle. Weather is temporally correlated; a random split would leak nearby hours into both sides and make test RMSE look too good.

## Why scale features?

Irradiation is ~0–1 kW/m², temperatures ~20–60 °C, Open-Meteo radiation is W/m² (hundreds), hour sine/cosine are ~[-1,1]. Batch/SGD steps use one alpha for every coordinate, so unscaled features drag theta at very different speeds. Scaling is `(x - train_mean) / train_std` in `scale_with_train`.

## Why reuse training mean and std on the test set?

Test statistics are unknown at deployment. Using test mean/std would leak test information into the features. The Flask app applies the saved training mean/std from `set_b_normal.npz`.

## Normal equation

`fit_normal` computes `theta = inv(X.T @ X) @ (X.T @ y)` with intercept column already in `X`. This is the exact minimiser of `J(theta) = 1/2 * sum((X @ theta - y)^2)` when `X.T @ X` is invertible. Set A `theta1` (scaled irradiation) is about 8268, the largest feature weight.

## Batch gradient descent

`fit_batch_gd` starts at theta = 0 and, each iteration, does `theta := theta + alpha * X.T @ (y - X @ theta)`, which is the assignment update using all n rows. Cost is stored after every iteration. Chosen alpha = 1e-4. Final Set A fit used 50,000 iterations because irradiation and temperatures are correlated; 500 steps were not enough to match the normal equation.

## Stochastic gradient descent

`fit_sgd` loops over training rows one at a time: `theta := theta + alpha * (y_i - x_i @ theta) * x_i`. Cost is recorded after each epoch. Chosen alpha = 0.01, 50 epochs. Set A max |theta_SGD - theta_normal| was about 859, so SGD did not reach the same vector.

## Learning rate

Too small: cost drops slowly (batch 1e-5; SGD 1e-4). Too large: cost explodes (batch 1e-3). About right: batch 1e-4; SGD 1e-3 and 1e-2. Alpha was chosen on Set A **training** cost only, never on test RMSE.

## Cost function

`cost` is `0.5 * sum((h - y)^2)`, not mean squared error. It is what GD minimises. RMSE is used only on the test set so the error is in kW.

## RMSE

`rmse = sqrt(mean((y - y_hat)^2)) = sqrt(2J / m)`. Reported twice: all 168 test hours, and daytime hours with `irradiation > 0`. Night hours are near zero and would hide daytime error. Set A normal-equation daytime RMSE was 727 kW vs training peak 27,326 kW (about 2.7%). Set B was 3427 kW.

## Set A vs Set B

Set A: on-site irradiation, module temp, ambient temp, plus sin/cos hour. Set B: Open-Meteo shortwave radiation, 2 m temperature, cloud cover, plus the same time features. Same target `ac_power`. Set B daytime RMSE was 2700 kW worse, about 9.9% of peak training AC.

## Irradiation vs shortwave radiation

Sensor `IRRADIATION` is kW/m² on the plant. Open-Meteo `shortwave_radiation` is W/m² at the chosen lat/lon. `fetch_weather.py` divides by 1000 only for the overlay plot. Set B keeps W/m² and lets scaling handle the magnitude. The Flask form also expects W/m².

## Why clip negative predictions?

`clip_nonnegative` is `max(pred, 0)`. A plant cannot inject negative AC. Night linear predictions can go slightly negative.

## Why daytime RMSE?

Most hours of a day are night. A model that predicts 0 at night looks accurate on 24-hour RMSE even if midday is wrong. The assignment therefore also scores `irradiation > 0`.

## Why should GD approach the normal equation?

Both minimise the same convex quadratic J. With a small enough alpha and enough iterations, batch GD should reach the same theta. After 50,000 steps, Set A max |theta_batch - theta_normal| was 0.00012 (agrees to two decimal places). SGD is noisier and did not.

## Why can public weather be worse than on-site sensors?

Open-Meteo is a gridded reanalysis at 14.82, 78.28, not a pyranometer on the array. Figure 08 shows Set B missing peaks and cloud dips that Set A tracks. Peak hours on 17–18 May differ by ±1 hour because of midday shape, not because the timezone was wrong (GMT+5:30, sunrise aligned).

## Extra questions you may get

**AC/DC ratio ~0.098?** Physics wants ~0.95–0.99. The Plant 1 file’s DC column is ~10× AC. We still predict AC.

**Why interpolate missing hours?** Outer merge + hourly mean left 20 incomplete hours; time interpolation (limit 2) then drop leftover NaNs. 804 rows were saved.

**Does the web app retrain?** No. `app/app.py` only loads `set_b_normal.npz`.
