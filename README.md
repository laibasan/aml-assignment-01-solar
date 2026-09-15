# Predicting Plant 1 hourly AC power from weather

Course assignment for AI4003 Applied Machine Learning. Least-squares linear regression is implemented by hand (normal equation, batch GD, stochastic GD). No scikit-learn or other ML libraries.

**Contributors:** [laibasan](https://github.com/laibasan) and [zohaib-2548](https://github.com/zohaib-2548)

**Repository:** https://github.com/laibasan/aml-assignment-01-solar

The practical question is how much prediction accuracy is lost when on-site irradiation and temperature sensors (Set A) are replaced by free Open-Meteo weather (Set B).

## Dataset

[Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) (Plant 1, 15 May–17 June 2020). Put the four Kaggle CSVs in `data/`.

## Installation

```text
pip install -r requirements.txt
```

Allowed packages: numpy, pandas, matplotlib, requests, Flask.

## How to run

From the project root:

```text
python src/load_data.py
python src/prepare.py
python src/eda.py
python src/fetch_weather.py
python src/train_eval.py
```

`fetch_weather.py` needs internet (Open-Meteo archive API). It writes `data/plant1_openmeteo.csv`. Do not paste API data by hand.

`train_eval.py` fits six models on training dates 15 May–10 June 2020 and reports test RMSE on 11–17 June 2020. Weights are saved under `results/saved_weights/`; it does not use the test set to choose alpha.

## Front end (local)

After training:

```text
python app/app.py
```

Open http://127.0.0.1:5000. Enter hour of day and Set B weather (shortwave radiation in **W/m²**, 2 m temperature, cloud cover). The app loads `results/saved_weights/set_b_normal.npz` and does not retrain.

Production start (Render / Railway):

```text
gunicorn wsgi:app --bind 0.0.0.0:$PORT
```

Live URL: add it here after deploy.

**GitHub:** https://github.com/laibasan/aml-assignment-01-solar

Two-person GitHub + deploy click-path: `communication/github_two_accounts.md`.
Blog draft: `communication/medium_blog.md`. LinkedIn draft: `communication/linkedin_post.md`.
Screenshot for the report: `results/figures/10_frontend.png` (captured from the working UI).

## Repository layout

```text
data/          raw CSVs, plant1_hourly.csv, plant1_openmeteo.csv
src/           load_data, prepare, eda, fetch_weather, regression, train_eval
results/       tables, figures, analysis.md, saved_weights
app/           Flask predictor
```

## Limitations

- Plant 1 `DC_POWER` is about 10× `AC_POWER`; AC is the target, DC is not a feature.
- Set A features are highly correlated, so batch GD needs many iterations to match the normal equation.
- Public weather is not measured at the array, so Set B error is larger.
- Linear regression cannot represent inverter clipping or fast cloud transients.

Do not treat this README as a results table; numbers live in `results/tables/` and `results/analysis.md` from the last `train_eval.py` run.
