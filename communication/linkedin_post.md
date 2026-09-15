# LinkedIn post (paste after the Medium article and GitHub repo are live)

Replace the two URLs before posting.

---

Can you predict a solar plant’s hourly AC power if you only have free public weather — no on-site irradiation sensor?

For our AI4003 Applied Machine Learning assignment we implemented least-squares linear regression **by hand** (normal equation, batch GD, and SGD; NumPy only) on Plant 1 of the Solar Power Generation Data set, then swapped on-site sensors for Open-Meteo.

On a held-out test week (11–17 June 2020):

- On-site sensors (Set A): daytime RMSE **727 kW** (2.7% of peak hourly AC)
- Public weather only (Set B): daytime RMSE **3,427 kW**
- Accuracy lost: **~2,700 kW**, or **~10% of peak**

That is good enough for a rough rooftop estimate, not for tight operational control.

Write-up: [MEDIUM_URL]
Code: https://github.com/laibasan and https://github.com/zohaib-2548 (add the repo URL after you push)

#MachineLearning #SolarEnergy #OpenMeteo #LinearRegression #AppliedML
