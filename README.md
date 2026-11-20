# Forecasting German Day-Ahead Electricity Prices

Final project, Practical Data Science (WS 2026/27), Julius-Maximilians-Universität Würzburg.
Group: Rashad Zarbaliyev, [Member 2], [Member 3].

We forecast the hourly German day-ahead electricity price (EUR/MWh) for the next day, using SMARD's day-ahead
forecasts, past prices and calendar information, i.e. no actual values of the forecast day. As an extension, we classify
whether an hour will have a negative price.

## Data

All data comes from [SMARD](https://www.smard.de/en/downloadcenter/download-market-data/), the electricity market
platform of the Bundesnetzagentur (licence: CC BY 4.0). Download settings: country Germany, resolution hour,
01.10.2018 - 17.11.2026, CSV. The four files are already in `data/`; to update them, download again and keep the file names:

| File | Content | Use |
| --- | --- | --- |
| `Day-ahead_prices_..._Hour.csv` | Day-ahead price DE/LU | Target |
| `Forecasted_consumption_..._Hour.csv` | Grid load and residual load forecast | Features |
| `Forecasted_generation_Day-Ahead_..._Hour.csv` | Wind onshore, wind offshore, solar forecast | Features |
| `Actual_generation_..._Hour.csv` | Generation by source | EDA only |

Public holidays come from the `holidays` Python package (official nationwide German holidays).

## Project structure

```
data/                     SMARD CSV files (included; see above for how to re-download)
src/data_pipeline.py      loading, cleaning, feature engineering, time split, monthly retraining loop
notebooks/
  01_Data_Preparation_and_EDA.ipynb
  02_Price_Forecasting_Models.ipynb
  03_Negative_Price_Classification.ipynb
src/report_figures.py     creates all figures of the report (run after notebook 02)
outputs/                  test predictions, result tables and feature importances (created by notebook 02)
app/dashboard.py          Gradio dashboard: compare the test forecasts with the actual prices
report/                   LaTeX source of the report in the chair's template (main.tex, literatur.bib, figures/)
requirements.txt
Report_Forecasting_German_Day-Ahead_Electricity_Prices.pdf   final report
docs/                     overview of the project for the team (Word)
```

## For the team: open the notebooks without installing anything

The notebooks on this page already show all results and charts; just click on them in the `notebooks/` folder.
To run or change them, use Google Colab:

| Notebook | |
| --- | --- |
| 01 Data preparation and EDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greenmario2099/pds-electricity-price-forecasting/blob/main/notebooks/01_Data_Preparation_and_EDA.ipynb) |
| 02 Price forecasting models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greenmario2099/pds-electricity-price-forecasting/blob/main/notebooks/02_Price_Forecasting_Models.ipynb) |
| 03 Negative-price classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greenmario2099/pds-electricity-price-forecasting/blob/main/notebooks/03_Negative_Price_Classification.ipynb) |

1. On this page, click **Code > Download ZIP** and save the file.
2. Open a notebook with its Colab button and run the first code cell ("Colab setup").
3. Because the repository is private, the cell asks you to upload a ZIP: choose the file from step 1.
   The notebook then finds the code and data by itself.

The project overview for the team is in `docs/`; it also links the slides.

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook          # run 01, 02, 03 in this order
cd ..
python src/report_figures.py   # recreates the figures in report/figures/
python app/dashboard.py        # then open the printed local address, e.g. http://127.0.0.1:7860
```

Notebook 02 takes about 20 minutes on a laptop (models are retrained every month).
All random seeds are fixed (`random_state=1`), so the results are reproducible.
Tested with Python 3.13, pandas 3.0, scikit-learn 1.9 and gradio 6, and with the library versions of Google Colab
(pandas 2.2, scikit-learn 1.6): notebooks 01 and 03 give identical results; in notebook 02 the random forest can differ
slightly in the decimals.

## Dashboard

`app/dashboard.py` is a small Gradio app (`gr.Interface`, as in lecture 09): pick a day of the test period (Jan-Nov 2026)
and compare the out-of-sample forecasts of the models with the actual prices. Three example days are one click away:
a normal workday (3 March), a Sunday with negative prices (10 May) and the price spike of 9 November.

## Report

The report is written in the chair's Overleaf template ([Uni Wue] Template Chair for Enterprise AI).
To compile it: open the template in Overleaf ("Open as Template"), replace `main.tex` and `declaration.tex`
with the files in `report/`, and upload `literatur.bib` and the `figures/` folder. The university seal (`seal.pdf`)
comes with the template. Locally: `cd report && latexmk -pdf main.tex` (needs biber). All figures are created by
`src/report_figures.py`, except `fig_dashboard.png`, which is a screenshot of the dashboard.

## Method

* **Leakage rule:** every feature must be known at noon on day D-1: SMARD day-ahead forecasts, prices of day D-1 and earlier,
  calendar information. Neighbour-country prices (same auction) and actual generation (published afterwards) are excluded.
  Note: the wind and solar day-ahead forecasts only have to be published by 18:00 on day D-1 (EU Regulation 543/2013, Art. 14), so they are not guaranteed to be available at noon. We use them as a proxy for the weather-based forecasts that traders prepare themselves; the load forecast is published before the auction.
* **Features (27):** load, residual load, wind and solar forecasts and derived features (renewable share, daily min/max,
  change vs. previous day), calendar (hour, weekday, month, holidays), lagged prices and rolling price statistics.
* **Split:** training 2023-2024, validation 2025, test 2026 (Jan-Nov). No shuffling.
* **Monthly retraining:** each month is predicted by models retrained on the previous two years (window length chosen on 2025).
* **Models (all scikit-learn, as in the course):** seasonal naive and linear regression (baselines), random forest, neural network (MLP),
  and the simple average of linear regression, random forest and MLP. The final model was chosen on the validation year before looking at the test set.
* **Evaluation:** MAE (main metric), RMSE, MAE on price spikes and negative hours, errors per month, and the share of test days
  on which the final model beats each other model, with a sign test.
* **Robustness check:** the test evaluation repeated without every feature that uses the official wind and solar forecasts.
* **Explainability:** random-forest feature importance and a partial dependence plot of the residual load.

## Results

**Price forecast, test set 2026 (models retrained monthly)**

| Model | MAE (EUR/MWh) | RMSE | Improvement vs. naive |
| --- | --- | --- | --- |
| Average of linear regression, random forest and MLP (final model) | 18.18 | 27.76 | 45.3% |
| Linear regression | 20.39 | 31.59 | 38.7% |
| Random forest | 21.00 | 31.76 | 36.9% |
| Neural network (MLP) | 23.37 | 33.20 | 29.7% |
| Naive (same hour yesterday) | 33.25 | 51.57 | - |

The final model is more accurate than the naive forecast on 79% of the 321 test days and more accurate than each single model
on 61-73% of the days (sign test: p < 0.001 against every model). The best single model changes from month to month; the average is the best
model in 8 of the 11 test months.

**Robustness check:** the official wind and solar forecasts only have to be published by 18:00 on the day before delivery,
so they may not be available at the noon auction. Without every feature based on them, the test MAE of the final model rises
from 18.18 to 29.38 EUR/MWh, still 11.6% better than the naive forecast. These forecasts carry most of the skill; a trader with
own weather-based forecasts lies between the two results.

**Negative-price classification, test set 2026 (6.0% negative hours)**

| Model | PR-AUC | Recall | Precision | F1 |
| --- | --- | --- | --- | --- |
| Logistic regression (final) | 0.843 | 0.929 | 0.619 | 0.743 |
| Random forest | 0.817 | 0.963 | 0.493 | 0.652 |
| Rule: renewable share > 0.9 | 0.691 | 0.874 | 0.462 | 0.605 |

## Main limitations

* Price spikes and negative prices are smoothed (bias towards the average).
* Gas and CO2 prices drive the price level but are not part of the SMARD data; in autumn 2026 the price level rose above
  anything in the training data, which increased all errors.
* The official wind and solar forecasts may be published after the auction closes; see the robustness check above.
* No systematic hyperparameter tuning; test period of 10.5 months in one market.
