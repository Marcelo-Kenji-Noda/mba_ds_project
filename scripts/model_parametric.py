# %%
## DATAPREP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datapath = "data/gbsg.csv"

class SurvivalDataset:
    def __init__(self,df:pd.DataFrame, time_col: str, event_col: str, features: list[str] = []):
        self.df = df
        self.time_col = time_col
        self.event_col = event_col
        if not features:
            self.features = [c for c in df.columns if c not in [time_col, event_col]]
        else:
            self.features = features
            
        self.time_df = df.loc[:, time_col]
        self.event_df = df.loc[:, event_col]
        self.features_df = df.loc[:, features]
    
df = pd.read_csv(datapath).drop(columns=['pid'])
dataset = SurvivalDataset(df=df, time_col='rfstime', event_col='status')

# %%
from lifelines import (
    WeibullAFTFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    KaplanMeierFitter,
    NelsonAalenFitter
)

# %%
km_model = KaplanMeierFitter().fit(dataset.time_df.values, dataset.event_df.values, label="Kaplan-Meier")
na_model = NelsonAalenFitter().fit(dataset.time_df.values, dataset.event_df.values, label="Nelson-Aalen")

# %%
models = {
    'WeibullAFT': WeibullAFTFitter(),
    'LogLogisticAFT': LogLogisticAFTFitter(),
    'LogNormalAFT': LogNormalAFTFitter()
}
models_results = {m:{} for m in models}

for model in models:
    print(f"Fitting model: {model} ...")
    try:
        models[model] = models[model].fit(dataset.df, duration_col=dataset.time_col, event_col=dataset.event_col)
    except Exception as e:
        print(f"Error fitting model {model}: {e}")

# %%
for m in models:
    try:
        models_results[m]['AIC'] = models[m].AIC_
        models_results[m]['BIC'] = models[m].BIC_
        models_results[m]['log_likelihood'] = models[m].log_likelihood_
    except Exception as e:
        print(f"\033[91m{m} does not have AIC/BIC/log_likelihood attributes.\033[0m")

# %%
models_results_df = pd.DataFrame(models_results).T
models_results_df.dropna().sort_values(by='AIC')

# %%
models['WeibullAFT'].plot()

# %%
models['LogLogisticAFT'].plot()

# %%
models['LogNormalAFT'].plot()

# %%
for model in models:
    print(f"\nSummary for model: {model}")
    try:
        print(models[model].summary)
    except Exception as e:
        print(f"Error retrieving summary for model {model}: {e}")

# %%
from sklearn.model_selection import KFold

# -----------------------
# CV CONFIGURATION
# -----------------------
K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=42)

cv_aic_bic_results = {m: {"AIC": [], "BIC": [], "log_likelihood": []} for m in models}

# -----------------------
# MANUAL K-FOLD LOOP
# -----------------------
for model_name, model_class in models.items():
    print(f"Running {K}-fold CV (AIC/BIC) for model: {model_name}")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset.df), start=1):
        train_df = dataset.df.iloc[train_idx]
        test_df  = dataset.df.iloc[test_idx]

        try:
            # Re-instantiate model to avoid leakage
            model = model_class.__class__()
            
            model.fit(
                train_df,
                duration_col=dataset.time_col,
                event_col=dataset.event_col
            )
            
            cv_aic_bic_results[model_name]["AIC"].append(model.AIC_)
            cv_aic_bic_results[model_name]["BIC"].append(model.BIC_)
            cv_aic_bic_results[model_name]["log_likelihood"].append(model.log_likelihood_)
            
        except Exception as e:
            print(f"  Fold {fold} failed for model {model_name}: {e}")
    
    print(
        f"  Mean AIC: {np.mean(cv_aic_bic_results[model_name]['AIC']):.2f} | "
        f"Mean BIC: {np.mean(cv_aic_bic_results[model_name]['BIC']):.2f}"
    )

# -----------------------
# SUMMARY TABLE
# -----------------------
cv_aic_bic_summary = pd.DataFrame({
    model: {
        "mean_AIC": np.mean(vals["AIC"]),
        "std_AIC":  np.std(vals["AIC"]),
        "mean_BIC": np.mean(vals["BIC"]),
        "std_BIC":  np.std(vals["BIC"]),
        "mean_log_likelihood": np.mean(vals["log_likelihood"])
    }
    for model, vals in cv_aic_bic_results.items()
}).T

cv_aic_bic_summary



