"""
Gradio dashboard for the project "Forecasting German Day-Ahead Electricity Prices".

Start it from the project folder with:
    python app/dashboard.py
and open the local address that is printed (e.g. http://127.0.0.1:7860).

Pick a day of the test period (Jan-Nov 2026) and compare the out-of-sample forecasts
of the models (saved by notebook 02) with the actual day-ahead prices.
"""
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

OUTPUTS = Path(__file__).resolve().parent.parent / 'outputs'

# ---------- load the saved results of notebook 02 ----------
preds = pd.read_csv(OUTPUTS / 'test_predictions_2026.csv', index_col='time', parse_dates=True)
comparison = pd.read_csv(OUTPUTS / 'model_comparison_test_2026.csv', index_col=0)

MODELS = [c for c in preds.columns if c != 'actual']
FINAL_MODEL = 'Average of the three models'
DAYS = sorted({d.strftime('%Y-%m-%d') for d in preds.index})


def show_day(day, models):
    day_data = preds.loc[day]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(day_data.index.hour, day_data['actual'], color='black', lw=2.5, label='actual')
    for m in models:
        ax.plot(day_data.index.hour, day_data[m], lw=1.5, label=m)
    ax.set_xlabel('Hour')
    ax.set_ylabel('EUR/MWh')
    ax.set_title(f'Day-ahead price on {day}')
    ax.legend(fontsize=8)
    plt.tight_layout()

    table = day_data[['actual'] + list(models)].round(1)
    table.insert(0, 'hour', day_data.index.hour)

    lines = [f"Average price: {day_data['actual'].mean():.1f} EUR/MWh "
             f"(min {day_data['actual'].min():.1f}, max {day_data['actual'].max():.1f})"]
    for m in models:
        lines.append(f'MAE of {m}: {(day_data[m] - day_data["actual"]).abs().mean():.1f} EUR/MWh')
    return fig, '\n'.join(lines), table


EXAMPLES = [['2026-03-03', ['Naive (same hour yesterday)', FINAL_MODEL]],   # normal workday
            ['2026-05-10', ['Naive (same hour yesterday)', FINAL_MODEL]],   # Sunday with negative prices at midday
            ['2026-11-09', MODELS]]                                        # extreme evening price spike

interface_args = dict(
    fn=show_day,
    inputs=[gr.Dropdown(DAYS, value=DAYS[-1], label='Day (test period 2026)'),
            gr.CheckboxGroup(MODELS, value=['Naive (same hour yesterday)', FINAL_MODEL], label='Models')],
    outputs=[gr.Plot(label='Forecast vs. actual'), gr.Textbox(label='Summary'), gr.Dataframe(label='Hourly values')],
    examples=EXAMPLES,
    cache_examples=False,
    title='German Day-Ahead Electricity Price Forecast',
    description=f'Out-of-sample forecasts for 2026, models retrained every month. Final model: {FINAL_MODEL} '
                f'(linear regression, random forest, neural network). Test MAE Jan-Nov 2026: '
                f'{comparison.loc[FINAL_MODEL, "MAE"]:.2f} EUR/MWh (naive: '
                f'{comparison.loc["Naive (same hour yesterday)", "MAE"]:.2f}). '
                'Examples below: a normal workday, a Sunday with negative prices and an extreme price spike.',
)
try:
    demo = gr.Interface(**interface_args, flagging_mode='never')    # Gradio 5 and newer
except TypeError:
    demo = gr.Interface(**interface_args, allow_flagging='never')   # Gradio 4

if __name__ == '__main__':
    demo.launch()
