"""
Creates all figures used in the final report and saves them to report/figures/.

Run from the project folder AFTER notebook 02 (it needs the files in outputs/):
    python src/report_figures.py

Every figure is computed from the SMARD data or from the saved outputs of the
notebooks, so the report figures are fully reproducible.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data_pipeline import load_data, build_features, load_actual_generation, FEATURES  # noqa: E402

DATA_DIR = str(ROOT / 'data') + '/'
OUT = ROOT / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name, dpi=200):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


data = load_data(DATA_DIR)
features = build_features(data)
price = data['price']

# --- Figure: daily average price with market events ---
fig, ax = plt.subplots(figsize=(10, 3.6))
daily = price.resample('D').mean()
ax.plot(daily.index, daily.values, lw=0.7, color='tab:blue')
for date, label in [('2022-02-24', 'War in Ukraine'), ('2023-04-15', 'Nuclear phase-out'),
                    ('2025-10-01', '15-min products')]:
    t = pd.Timestamp(date, tz='Europe/Berlin')
    ax.axvline(t, color='grey', ls='--', lw=0.8)
    ax.text(t, daily.max() * 0.93, ' ' + label, fontsize=8)
ax.set_ylabel('EUR/MWh')
save(fig, 'fig_daily_price.pdf')

# --- Figure: daily profile by year ---
fig, ax = plt.subplots(figsize=(7, 3.6))
for year in [2019, 2021, 2023, 2025, 2026]:
    s = price[price.index.year == year]
    s.groupby(s.index.hour).mean().plot(ax=ax, label=str(year))
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Average price (EUR/MWh)')
ax.legend(ncol=5, fontsize=8, frameon=False)
save(fig, 'fig_daily_profile.pdf')

# --- Figure: merit order, price vs residual load 2019 vs 2025 ---
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, year in zip(axes, [2019, 2025]):
    s = data.loc[str(year)].sample(4000, random_state=1)
    ax.scatter(s['residual_load_fc'] / 1000, s['price'], s=4, alpha=0.2)
    ax.set_title(str(year))
    ax.set_xlabel('Forecasted residual load (GWh)')
axes[0].set_ylabel('Price (EUR/MWh)')
axes[0].set_ylim(-150, 500)
save(fig, 'fig_merit_order.png')

# --- Figure: generation mix ---
gen = load_actual_generation(DATA_DIR).loc['2019':'2025']
mix = gen.groupby(gen.index.year).sum() / 1e6
fig, ax = plt.subplots(figsize=(9, 4))
mix.plot(kind='bar', stacked=True, ax=ax, width=0.75, colormap='tab20')
ax.set_ylabel('TWh')
ax.set_xlabel('')
ax.legend(bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
plt.xticks(rotation=0)
save(fig, 'fig_generation_mix.pdf')

# --- Figures from the test predictions (notebook 02) ---
preds = pd.read_csv(ROOT / 'outputs' / 'test_predictions_2026.csv', index_col='time', parse_dates=True)
comparison = pd.read_csv(ROOT / 'outputs' / 'model_comparison_test_2026.csv', index_col=0)
final = 'Average of the three models'

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
for ax, (start, end) in zip(axes, [('2026-05-04', '2026-05-10'), ('2026-11-09', '2026-11-15')]):
    week = preds.loc[start:end]
    ax.plot(week.index, week['actual'], color='black', lw=1.6, label='Actual')
    ax.plot(week.index, week[final], color='tab:red', lw=1.2, label='Final model (average)')
    ax.plot(week.index, week['Naive (same hour yesterday)'], color='tab:grey', lw=1, ls='--', label='Naive')
    ax.set_ylabel('EUR/MWh')
    ax.set_title(f'{start} to {end}', fontsize=10)
axes[0].legend(ncol=3, fontsize=8, frameon=False)
save(fig, 'fig_test_weeks.pdf')

err = (preds['actual'] - preds[final]).abs()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
err.groupby(preds.index.hour).mean().plot(kind='bar', ax=axes[0], color='tab:blue')
axes[0].set_xlabel('Hour of the day')
axes[0].set_ylabel('MAE (EUR/MWh)')
bins = pd.cut(preds['actual'], bins=[-600, 0, 50, 100, 150, 200, 300, 1000],
              labels=['≤0', '0-50', '50-100', '100-150', '150-200', '200-300', '>300'])
err.groupby(bins).mean().plot(kind='bar', ax=axes[1], color='tab:blue')
axes[1].set_xlabel('Actual price range (EUR/MWh)')
plt.setp(axes[0].get_xticklabels(), rotation=0, fontsize=7)
plt.setp(axes[1].get_xticklabels(), rotation=0)
save(fig, 'fig_error_analysis.pdf')

# --- Feature importance and partial dependence of the random forest (same model as in notebook 02) ---
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.inspection import PartialDependenceDisplay  # noqa: E402

importance = pd.read_csv(ROOT / 'outputs' / 'feature_importance_rf.csv', index_col=0).iloc[:, 0]
fig, ax = plt.subplots(figsize=(7, 4.8))
importance.head(15).sort_values().plot(kind='barh', ax=ax, color='tab:blue')
ax.set_xlabel('Feature importance (share of total)')
save(fig, 'fig_feature_importance.pdf')

latest_train = features.loc['2024-11-18':'2026-11-17']
latest_rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=1)
latest_rf.fit(latest_train[FEATURES], latest_train['price'])
sample = features.loc['2026', FEATURES].sample(2000, random_state=1)
fig, ax = plt.subplots(figsize=(7, 4))
PartialDependenceDisplay.from_estimator(latest_rf, sample, ['residual_load_fc'], ax=ax)
ax = fig.axes[-1]
ax.set_xlabel('Forecasted residual load (MWh)')
ax.set_ylabel('Average price forecast (EUR/MWh)')
save(fig, 'fig_partial_dependence.pdf')

# --- Precision-recall curves for the negative-price classifiers (same settings as notebook 03) ---
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.metrics import precision_recall_curve, average_precision_score  # noqa: E402

test_train = features.loc['2024-01':'2025-12']
test = features.loc['2026-01':'2026-11']
models = {
    'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    'Random forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=1),
}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
scores = {'Rule: renewable share': test['renewable_share']}
for name, m in models.items():
    m.fit(test_train[FEATURES], test_train['negative'])
    scores[name] = m.predict_proba(test[FEATURES])[:, 1]
for name, s in scores.items():
    p, r, _ = precision_recall_curve(test['negative'], s)
    ax.plot(r, p, label=f'{name} (PR-AUC {average_precision_score(test["negative"], s):.2f})')
ax.axhline(test['negative'].mean(), color='grey', ls='--', lw=1, label='Random guess')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(fontsize=8, frameon=False)
save(fig, 'fig_pr_curves.pdf')
