================================================================================
 Vehicle Travel Time Prediction
 Machine Learning Fundamentals - Final Practical Assignment
 Authors: Angel Miwa / Miroslava Lange / Maximo Fox
 Date: 2026/06/25
================================================================================

WHAT THIS IS
------------
A complete, reproducible pipeline that predicts the average travel time of every
2-minute slice in [08:00, 09:00) for June 2017, for each of the 132 road links,
using the historical travel-time data, the link attributes and the road topology.

The selected model (Lasso regression on a multiplicative-correction target)
reaches a MAPE of 13.62% on the held-out June-2017 test month, beating both the
historical-average baseline (16.99%) and a neural network.


CONTENTS
--------
Vehicle_Travel_Time_Prediction_Report.pdf  Written report (read this first).
report.md                                  Markdown source of the report.
vehicle_travel_time_prediction.ipynb       Notebook reproducing the whole pipeline
                                           (already executed, with outputs).
src/
    config.py        Paths and constants.
    data_prep.py     Streams the ~2 GB archives, caches the morning rows.
    features.py      Monthly-profile feature engineering.
    models.py        Baseline / Ridge / Lasso / MLP and the metrics (MAPE, ...).
make_results.py      Script version: runs everything and writes figures + results.json.
build_report.py      Renders report.md to the PDF.
build_notebook.py    Generates the notebook.
predictions/prediction.csv   The submission: 3,960 rows (132 links x 30 slices).
figures/                     Figures used in the report.
results.json                 All numbers reported (metrics, chosen hyper-parameters).
requirements.txt             Python dependencies.


DATA (not bundled - ~2 GB)
--------------------------
Place the original dataset files next to this folder (i.e. in the parent
directory), so the layout is:

    <parent>/
        gy_link_info.txt
        gy_link_top.txt
        gy_link_travel_time_part1.zip
        gy_link_travel_time_part2.zip
        gy_link_travel_time_part3.zip
        vehicle-travel-time-prediction/  <-- this folder

Alternatively set the GY_DATA_DIR environment variable to the data directory.


HOW TO REPRODUCE
----------------
1) pip install -r requirements.txt
2) From inside this folder, run either:
       python make_results.py          # builds the cache, trains, writes figures,
                                        # results.json and predictions/prediction.csv
   or open and run vehicle_travel_time_prediction.ipynb top to bottom.
3) (optional) python build_report.py   # rebuilds the PDF from report.md

The first run performs a one-off streaming scan of the ~2 GB archives and writes a
small cache to <parent>/data_cache/ (a few minutes). Subsequent runs reuse the
cache and finish in well under two minutes.


METHOD IN ONE PARAGRAPH
-----------------------
Because the metric (MAPE) is computed over a monthly-averaged profile, we build the
supervised problem at that granularity (one example per month, link and target
slice). Features: static attributes, calendar context, the month's morning
[06:00, 08:00) signal, a leave-one-month-out historical mean and a topology-based
neighbour mean. The models learn the correction factor r = travel_time /
historical_mean and we recover the prediction as r * historical_mean. Models are
compared with leave-one-month-out cross-validation; the final Lasso model is then
evaluated once on the June-2017 test month.
