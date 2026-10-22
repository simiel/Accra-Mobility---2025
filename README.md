# Accra Mobility 2025

**Author:** [Samuel Mensah](https://github.com/simiel)

This repository is a hackathon workspace for predicting how long a trip in Accra takes. Each row is one trip, with origin and destination coordinates, straight-line distance, actual distance traveled, and the local start time. Hourly Accra weather (precipitation type, previous-hour rainfall, and temperature) is joined on day and hour. The label is `Target`, travel time in minutes. The notebooks explore that table, engineer features, fit classical regressors and neural networks, and write Zindi submission files with columns `trip_id` and `Target`.

The included files cover **57,596** training trips and **24,684** test trips. Almost every trip is dated May 2024 (a few rows fall in early June). `data/Accra_weather.csv` is one row per hour for all **744** hours of May 2024. Travel time is right-skewed: the training target runs from **1.02** to **585.93** minutes, with a mean of about **10.08** minutes. **191** training trips (**0.33%**) last longer than 60 minutes. Precipitation in the weather file averages about **0.152 mm** in the previous hour (maximum about **4.34 mm**). Temperature stays between **25.92°C** and **30.49°C**, with a mean of about **28.30°C**.

## Key features

- Exploratory analysis of the travel-time distribution, outliers, duplicates, missing values, and correlation with distance, calendar fields, and weather.
- A day-hour key that left-joins hourly weather onto each trip.
- Trip features: hour, day, weekday, weekend, rush hour (07:00–09:00 and 17:00–19:00), sine/cosine hour-of-day, distance ratio (actual over straight-line, clipped and guarded against division by zero), distance difference, distance buckets, and time-of-day buckets.
- Weather features: one-hot precipitation type, 3-hour rolling means, one-hour lags, and hour-to-hour differences for rainfall and temperature. The precipitation dummies are built in `Noon.ipynb` and left out of the final feature list there.
- A fixed seed, `SEED = 2025`, on the splits and models that take one.
- Model families carried through to saved predictions or notebook metrics: k-nearest neighbors, random forests, gradient boosting, XGBoost, ridge regression, SVR, scikit-learn `MLPRegressor`, and TensorFlow/Keras MLPs. Commit history also records passes with linear regression, Bayesian ridge, histogram gradient boosting, AdaBoost, and linear SVR. The import cells list a still wider scikit-learn catalog (regularized linear models, extra trees, stacking, Gaussian processes, and others) that those cells do not all fit.
- Neural-net training with `StandardScaler`, batch normalization, dropout, early stopping, and `ReduceLROnPlateau`. `Noon.ipynb` compares six depth / dropout / learning-rate / optimizer settings with 5-fold cross-validation (Adam, Nadam, RMSprop, SGD, Adagrad, Adadelta).
- A small voting ensemble in `StarterNotebook.ipynb`: equal-weight average of a random forest, gradient boosting, XGBoost, k-NN, ridge, and RBF SVR.
- Submission files in the sample format (`trip_id`, `Target`), including clipping of neural-net predictions at 1 minute in `Noon.ipynb`.

## Repository layout

```
data/
  Train.csv                  # 57,596 trips plus Target
  Test.csv                   # 24,684 trips, no label
  Accra_weather.csv          # hourly weather, May 2024
  VariableDefinitions.csv    # attribute dictionary for trip and weather fields
submissions/                 # prediction files and the sample submission
BaselineSubmission.csv       # random-forest baseline written by winner.ipynb
StarterNotebook.ipynb        # starter EDA, k-NN, and the voting ensemble
Noon.ipynb                   # outlier filter, richer features, Keras search
Neural network.ipynb         # starter EDA, k-NN, and a deeper Keras MLP
winner.ipynb                 # random-forest baseline and a Keras MLP submission
winner.html                  # rendered export of an earlier starter notebook
```

`data/VariableDefinitions.csv` also describes fields that are not columns in these CSVs (for example order-finish time and a separate actual-duration field). In the files here, duration is the `Target` column.

| Notebook | What the saved run contains |
| --- | --- |
| `StarterNotebook.ipynb` | Starter EDA, then a 5-feature k-NN and a 6-model voting ensemble with printed RMSE. The later TensorFlow and “enhanced features” cells have no saved output. |
| `Noon.ipynb` | Outlier filtering, trip and weather features, the Keras cross-validation search, and submission `FF2_20250705_161320.csv`. The voting-ensemble cell stops on a Keras error (`compile()` before use). The neural-network ensemble cell stops on an attribute error. The mega-ensemble cell (a 0.4 / 0.6 mix of those two ensembles) has no output. |
| `Neural network.ipynb` | Starter EDA, a default k-NN, and a 512–256–128 network with a printed validation RMSE. The later enhanced k-NN cell has no saved output. |
| `winner.ipynb` | Random forest written to `BaselineSubmission.csv`, plus a 128–64–32 network written to `submissions/FF4_*.csv`. |
| `winner.html` | HTML export titled `StarterNotebook`, with its own printed metrics (see below). |

## Approach

The classical cells start from five columns: straight-line distance, actual distance, temperature, day of month, and previous-hour precipitation, with missing values filled as 0.

`Noon.ipynb` is the fullest pipeline. It drops training trips outside the 1st–99th percentile of `Target`, builds the trip and weather features above, scales with `StandardScaler`, and runs 5-fold cross-validation over the six Keras configurations. The best saved setting is a 256–128–64 network with batch normalization, dropout 0.2, and RMSprop at learning rate 0.001. That notebook then predicts the test set and stores `submissions/FF2_20250705_161320.csv`.

`winner.ipynb` keeps the starter random forest on the five-column set (30% holdout) and adds a separate network on distance, the distance ratio, origin and destination coordinates, cyclic hour, and a rush-hour flag. Hidden widths are 128, 64, and 32, with ReLU, dropout, Adam at 0.001, batch size 256, and up to 200 epochs with early stopping.

`StarterNotebook.ipynb` fits the voting ensemble on the same five columns and a 5% holdout. XGBoost in that ensemble uses 1,000 trees, learning rate 0.1, and `max_depth` 3.

## Results

Every figure below is an RMSE in minutes printed in a notebook or in `winner.html`. This repository does not contain a Zindi leaderboard score. Each row is a separate experiment: the splits differ, and only `Noon.ipynb` removes extreme travel times before scoring.

| Source | Model | Evaluation saved with the score | RMSE (minutes) |
| --- | --- | --- | --- |
| `Noon.ipynb` | Keras MLP, 256–128–64, dropout 0.2, RMSprop, lr 0.001 | 5-fold CV on trips kept inside the 1st–99th percentile of travel time | 3.0918 |
| `StarterNotebook.ipynb` | Equal-weight vote of random forest, gradient boosting, XGBoost, k-NN (k=5), ridge, and RBF SVR | 5% holdout, five features, full target range | 3.9893 |
| `winner.html` | `MLPRegressor` | 5% holdout, five features | 4.0521 |
| `winner.html` | TensorFlow network in that export | 20% validation | 4.0415 |
| `winner.ipynb` | Keras MLP, 128–64–32, Adam, dropout | 20% validation on the spatial and cyclic-hour features | 4.0745 |
| `StarterNotebook.ipynb` | `KNeighborsRegressor`, k=5 | 5% holdout, five features | 4.3082 |
| `winner.ipynb` | `RandomForestRegressor` | 30% holdout, five features | 4.4766 |
| `Neural network.ipynb` | Keras MLP, 512–256–128 | 20% validation | 4.4974 |
| `Neural network.ipynb` | `KNeighborsRegressor` (library defaults) | 5% holdout, five features | 4.9620 |

Other prediction files in `submissions/`, all dated 5 July 2025, record further XGBoost and neural-net exports (`XGBRegressor*.csv`, `XGB1_*.csv`, `NN__*.csv`, `NNN__*.csv`, `NNN2__*.csv`) and `VotingEnsemble_20250705_161821.csv`. `SampleSubmission.csv` and `SampleSubmission2.csv` are templates with the same 24,684 ids.

## Tech stack

Python **3.12** (kernel metadata 3.12.7; `Noon.ipynb` ran under Anaconda). Jupyter notebooks. Libraries imported by the notebooks: pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras, XGBoost, and joblib.

There is no `requirements.txt`, environment file, or dependency lockfile. The notebooks contain commented `pip install` lines for `xgboost` and `tensorflow`.

## Getting started

Prerequisites: Python 3.12, Jupyter, and the libraries named above.

The repository defines no environment variables and no secret configuration. Data and sample submissions are already in the tree.

From the repository root (the notebooks read `data/` and `submissions/` with relative paths):

```bash
jupyter notebook
```

Open `Noon.ipynb` for the feature pipeline and the cross-validated network, `StarterNotebook.ipynb` for the voting ensemble, or `winner.ipynb` for the random-forest baseline and the `FF4` network. Cells that call `to_csv` write a new timestamped file under `submissions/` or overwrite `BaselineSubmission.csv`.

There is no build step and no test command in this repository.

## Project status

Hackathon entry from July 2025. The notebooks call the task a hackathon and tell the reader to submit the prediction file to Zindi for scoring. Submission filenames and most modeling commits are from 5 July 2025. `winner.ipynb` and `submissions/FF4_20250705_162845.csv` were committed on 6 July 2025. The 25 August 2025 commits only update notebook kernel metadata. The deliverable is these notebooks and CSV submissions: a finished exploration of the trip-time task.

## What this demonstrates

- Taking a mobility ETA problem from raw trip and weather tables through to submission files in the organizer’s schema.
- Reading a skewed target before modeling: the long tail above an hour is small (0.33% of training trips), and `Noon.ipynb` handles it with a percentile filter before cross-validation.
- Feature work that matches the domain: cyclic time, rush hour and weekend, how much farther the real route is than the straight line, and short-horizon weather (lags, differences, and 3-hour rolls), joined carefully on local day and hour.
- A broad, seeded model search across linear models, neighbors, SVMs, forests, boosting (including XGBoost), and multilayer perceptrons, with the experiments left in the notebooks, the HTML export, and the submission folder.
- Training practice on the networks: scaling, batch normalization, dropout, early stopping, learning-rate schedules, and a 5-fold comparison of width, dropout, learning rate, and optimizer. The best saved cross-validation RMSE is **3.09 minutes** on the filtered target.
- An equal-weight vote of six model families, with a saved holdout RMSE of **3.99 minutes** on the unfiltered target, alongside later ensemble designs (a neural-net ensemble and a 0.4/0.6 mega-ensemble) whose saved runs in `Noon.ipynb` did not finish.
