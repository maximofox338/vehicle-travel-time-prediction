# Vehicle Travel Time Prediction

Predicting the rush-hour travel time of 132 urban road links from ~26 million historical
2-minute records. For each link, the model forecasts the average travel time of every 2-minute
slice in **[08:00, 09:00) for June 2017** from the morning signal [06:00, 08:00), the link
attributes and the road topology.

**Result on the held-out test month: 13.6% MAPE, against 17.0% for the historical-average
baseline.**

![Predicted vs actual](figures/fig6_pred_vs_actual.png)

## Results

Leave-one-month-out cross-validation over the six labelled months, then a single evaluation on
June 2017:

| Model | CV MAPE (%) | Test MAPE (%) | Test R² |
|---|---|---|---|
| Baseline (historical mean) | 21.53 | 16.99 | 0.879 |
| Ridge | 18.55 | – | – |
| **Lasso (selected)** | **18.54** | **13.62** | **0.924** |
| MLP | 23.30 | – | – |
| Gradient boosting *(appendix, outside the course)* | 14.21 | 11.36 | 0.927 |

The course allowed only linear models and neural networks, so gradient boosting is reported
for reference and was never eligible for selection.

![Model comparison](figures/fig5_model_comparison.png)

## Approach

- **Streaming ingestion.** The ~2 GB of zipped records are read in chunks. Only the
  [06:00, 09:00) rows are kept (3.6M rows), and they are cached after the first pass.
- **Matching the metric's granularity.** The official MAPE is computed over a monthly-averaged
  profile, so there is one example per *(month, link, slice)*, not per day.
- **Multiplicative-correction target.** The models learn `r = travel_time / historical_mean`
  and predict `r̂ × historical_mean`. Because the historical mean cancels, the MAPE on `r`
  equals the MAPE on the travel time, which means hyper-parameters are tuned directly on the
  competition metric.
- **Features.** Static link attributes, calendar context, the month's morning signal (mean,
  trend, std, value one hour earlier), a leave-one-month-out historical mean and a
  topology-based neighbour mean.
- **Leakage-safe CV.** The historical features are rebuilt in every fold from the training
  months only. A unit test asserts that changing the validation month's labels leaves every
  training feature unchanged.

### Findings worth reading

- **The test score is better than the CV score, and the reason is known.** The baseline shows
  the same gap (21.5% → 17.0%). Traffic in 2017 is slower than in 2016. A held-out 2017 month
  gets a historical mean dominated by 2016, while the June 2017 test month gets one with three
  months from each year. See §7 of the report.
- **Ridge and Lasso are tied** (0.01 points apart), and the regularised linear model beats the
  MLP.
- **On 2016 months the linear model loses to the baseline.** Its gain comes from learning the
  2016 → 2017 level shift. Gradient boosting beats the baseline on every month.

## Run it

```bash
git clone https://github.com/maximofox338/vehicle-travel-time-prediction.git
cd vehicle-travel-time-prediction
pip install -r requirements.txt
python make_results.py --demo     # synthetic data, under a minute, writes to demo_output/
python test_pipeline.py           # unit checks, no data needed
```

`--demo` generates a small synthetic dataset with the same file format
(`make_sample_data.py`) and runs the full pipeline: EDA, CV, test evaluation, figures and the
submission file. The numbers it produces are meaningless; it shows that the code runs.

### With the real data

The original dataset (`gy_link_info.txt`, `gy_link_top.txt`, `gy_link_travel_time_part{1,2,3}.zip`,
~2 GB) is not included. Point the pipeline at it:

```bash
GY_DATA_DIR=/path/to/data python make_results.py
```

The first run scans the archives once and caches the morning rows, which takes a few minutes.
Later runs take about 3 minutes. The notebook `vehicle_travel_time_prediction.ipynb`
reproduces the same steps interactively and is committed with its outputs.

## Structure

```
src/
  config.py       paths and constants (GY_DATA_DIR / GY_CACHE_DIR / GY_OUT_DIR)
  data_prep.py    streaming scan of the archives + cache
  features.py     monthly-profile examples, leakage-safe historical features
  models.py       baseline / Ridge / Lasso / MLP / gradient boosting, MAPE
make_results.py   end-to-end run: figures, results.json, predictions/prediction.csv
make_sample_data.py   synthetic dataset for --demo
test_pipeline.py  unit checks
report.md         full write-up (rendered to the PDF by build_report.py)
```

## Authors

Angel Miwa · Miroslava Lange · Maximo Fox. Final practical assignment for *Machine Learning
Fundamentals*, June 2026.
