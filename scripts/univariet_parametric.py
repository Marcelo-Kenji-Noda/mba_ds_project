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
    WeibullFitter,
    ExponentialFitter,
    LogNormalFitter,
    LogLogisticFitter,
    NelsonAalenFitter,
    PiecewiseExponentialFitter,
    GeneralizedGammaFitter,
    SplineFitter,
    KaplanMeierFitter
)
from lifelines.plotting import qq_plot

# %%
knots = np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [0, 50, 100])

models = { 
    'Weibull': WeibullFitter(),
    'Exponential': ExponentialFitter(),
    'LogNormal': LogNormalFitter(),
    'LogLogistic': LogLogisticFitter(),
    'NelsonAalen': NelsonAalenFitter(),
    'PiecewiseExponential': PiecewiseExponentialFitter([40, 60]),
    'GeneralizedGamma': GeneralizedGammaFitter(),
    'Spline': SplineFitter(knots),
    'KaplanMeier': KaplanMeierFitter()
}

models_results = {m:{} for m in models}

for model in models:
    models[model] = models[model].fit(dataset.time_df.values, dataset.event_df.values, label=model)

# %%
models_results

# %%
for m in models:
    try:
        models_results[m]['AIC'] = models[m].AIC_
        models_results[m]['BIC'] = models[m].BIC_
        models_results[m]['log_likelihood'] = models[m].log_likelihood_
    except AttributeError:
        print(f"\033[91m{m} does not have AIC/BIC/log_likelihood attributes.\033[0m")

# %%
models_results_df = pd.DataFrame(models_results).T
models_results_df.dropna().sort_values(by='AIC')

# %%
print(models_results_df.dropna().sort_values(by='AIC').to_markdown())

# %%
fig, axes = plt.subplots(3, 3, figsize=(10, 7.5))

models['Weibull'].plot_survival_function(ax=axes[0][0])
models['Exponential'].plot_survival_function(ax=axes[0][1])
models['LogNormal'].plot_survival_function(ax=axes[0][2])
models['KaplanMeier'].plot_survival_function(ax=axes[1][0])
models['LogLogistic'].plot_survival_function(ax=axes[1][1])
models['PiecewiseExponential'].plot_survival_function(ax=axes[1][2])
models['GeneralizedGamma'].plot_survival_function(ax=axes[2][0])
models['Spline'].plot_survival_function(ax=axes[2][1])

# Add a centered suptitle and adjust layout so it doesn't overlap the subplots
fig.suptitle('Função de sobrevivência — parametric and nonparametric models', fontsize=14)
# Leave space for the suptitle (top margin) before tightening layout
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
fig, ax_grid = plt.subplots(3, 3, figsize=(10, 7.5))
ax_flat = ax_grid.flatten()

plot_order = [
    'Weibull', 'Exponential', 'LogNormal', 'KaplanMeier',
    'LogLogistic', 'PiecewiseExponential', 'GeneralizedGamma', 'Spline'
]

for i, name in enumerate(plot_order):
    # plot the model
    models[name].plot_survival_function(ax=ax_flat[i])
    # overlay Kaplan-Meier on every subplot except when the subplot is the KM itself
    if name != 'KaplanMeier':
        models['KaplanMeier'].plot_survival_function(
            ax=ax_flat[i],
            ci_show=False,
            label='KaplanMeier',
            linestyle='--'
        )

# hide the unused (9th) subplot
ax_flat[-1].axis('off')

fig.suptitle('Função de sobrevivência — parametric and nonparametric models', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
fig, axes = plt.subplots(3, 3, figsize=(10, 7.5))


models['Weibull'].plot_cumulative_hazard(ax=axes[0][0])
models['Exponential'].plot_cumulative_hazard(ax=axes[0][1])
models['LogNormal'].plot_cumulative_hazard(ax=axes[0][2])
models['NelsonAalen'].plot_cumulative_hazard(ax=axes[1][0])
models['LogLogistic'].plot_cumulative_hazard(ax=axes[1][1])
models['PiecewiseExponential'].plot_cumulative_hazard(ax=axes[1][2])
models['GeneralizedGamma'].plot_cumulative_hazard(ax=axes[2][0])
models['Spline'].plot_cumulative_hazard(ax=axes[2][1])

# Add a centered suptitle and adjust layout so it doesn't overlap the subplots
fig.suptitle('Cumulative hazard functions — parametric and nonparametric models', fontsize=14)
# Leave space for the suptitle (top margin) before tightening layout
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
fig_ch, ax_grid_ch = plt.subplots(3, 3, figsize=(10, 7.5))
ax_flat_ch = ax_grid_ch.flatten()

plot_order = [
    'Weibull', 'Exponential', 'LogNormal', 'NelsonAalen',
    'LogLogistic', 'PiecewiseExponential', 'GeneralizedGamma', 'Spline'
]

for i, name in enumerate(plot_order):
    # main model cumulative hazard
    models[name].plot_cumulative_hazard(ax=ax_flat_ch[i])
    # overlay NelsonAalen on every subplot (skip overlaying on itself to avoid duplicate)
    if name != 'NelsonAalen':
        models['NelsonAalen'].plot_cumulative_hazard(
            ax=ax_flat_ch[i],
            ci_show=False,
            label='NelsonAalen',
            linestyle='--',
            color='k'
        )
    ax_flat_ch[i].legend()

# hide the unused (9th) subplot
ax_flat_ch[-1].axis('off')

fig_ch.suptitle('Cumulative hazard functions — parametric and NelsonAalen for comparison', fontsize=14)
fig_ch.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# ## Model Selection using QQPLot
# 

# %%
def check_qq_plot_support(models):
    allowed_models = []
    for model_name in models:
        try:
            qq_plot(models[model_name])
            plt.close()  # Close the plot without displaying
            print(f"\033[92mQQ-plot supported for {model_name}\033[0m")  # Green text
            allowed_models.append(model_name)
        except Exception:
            print(f"\033[91mQQ-plot not implemented for {model_name}\033[0m")  # Red text
    return allowed_models

allowed_models = check_qq_plot_support(models)
plt.close()

# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.reshape(4,)

for i, model in enumerate(allowed_models):
    qq_plot(models[model], ax=axes[i])
    
plt.tight_layout()

# %% [markdown]
# ## Diving deeper on the splines model

# %%
## Diving deeper on the splines model
spline_models = {
    'Spline_2_knot_0_50': SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [0, 50])),
    'Spline_3_knot_0_50_100': SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [0, 50, 100])),
    'Spline_4_knots_0_25_50_75': SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [0, 25, 50, 75])),
    'Spline_5_knots_0_20_40_60_80': SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [0, 20, 40, 60, 80])),
    'Spline_6_knots_0_20_40_60_80_100': SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [0, 20, 40, 60, 80, 100])),
}


def get_models_results_df(models_dict, dataset):
    models_results = {m:{} for m in models_dict}

    for model in models_dict:
        models_dict[model] = models_dict[model].fit(dataset.time_df.values, dataset.event_df.values, label=model)
        
    for m in models_dict:
        try:
            models_results[m]['AIC'] = models_dict[m].AIC_
            models_results[m]['BIC'] = models_dict[m].BIC_
            models_results[m]['log_likelihood'] = models_dict[m].log_likelihood_
        except AttributeError:
            print(f"\033[91m{m} does not have AIC/BIC/log_likelihood attributes.\033[0m")
            
    return pd.DataFrame(models_results).T.dropna().sort_values(by='AIC')

# %%
get_models_results_df(spline_models, dataset)

# %% [markdown]
# # Trying out differente locations for 3 knots

# %%
import random

def generate_pair_tuples(n=1, low=0, high=100):
    """
    Generate n tuples of 2 distinct integers in ascending order.
    """
    return [tuple(sorted(random.sample(range(low, high + 1), 2))) for _ in range(n)]

def generate_triplets(n=1, low=0, high=100):
    """
    Generate n lists of 3 distinct integers in ascending order.
    """
    return [sorted(random.sample(range(low, high + 1), 3)) for _ in range(n)]

def generate_4tuples(n=1, low=0, high=100):
    """
    Generate n tuples of 4 distinct integers in ascending order.
    """
    return [tuple(sorted(random.sample(range(low, high + 1), 4))) for _ in range(n)]

def generate_5tuples(n=1, low=0, high=100):
    """
    Generate n tuples of 5 distinct integers in ascending order.
    """
    return [tuple(sorted(random.sample(range(low, high + 1), 5))) for _ in range(n)]

def generate_6tuples(n=1, low=0, high=100):
    """
    Generate n tuples of 6 distinct integers in ascending order.
    """
    return [tuple(sorted(random.sample(range(low, high + 1), 6))) for _ in range(n)]


# %%
random.seed(42)

duples = generate_pair_tuples(100)
triplets = generate_triplets(100)
quadruplets = generate_4tuples(100)
quintuples = generate_5tuples(100)
sextuples = generate_6tuples(100)

challengers2 = {}
challengers3 = {}
challengers4 = {}
challengers5 = {}
challengers6 = {}

for v1, v2 in duples:
    challengers2[f'Spline_2_knot_{v1}_{v2}'] = SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [v1, v2]))

for v1, v2, v3 in triplets:
    challengers3[f'Spline_3_knot_{v1}_{v2}_{v3}'] = SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [v1, v2, v3]))

for v1, v2, v3, v4 in quadruplets:
    challengers4[f'Spline_4_knot_{v1}_{v2}_{v3}_{v4}'] = SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [v1, v2, v3, v4]))

for v1, v2, v3, v4, v5 in quintuples:
    challengers5[f'Spline_5_knot_{v1}_{v2}_{v3}_{v4}_{v5}'] = SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [v1, v2, v3, v4, v5]))

for v1, v2, v3, v4, v5, v6 in sextuples:
    challengers6[f'Spline_6_knot_{v1}_{v2}_{v3}_{v4}_{v5}_{v6}'] = SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [v1, v2, v3, v4, v5, v6]))

results_of_duples = get_models_results_df(challengers2, dataset)
results_of_duples['model_type'] = 'duples'
results_of_triplets = get_models_results_df(challengers3, dataset)
results_of_triplets['model_type'] = 'triplets'
results_of_quadruplets = get_models_results_df(challengers4, dataset)
results_of_quadruplets['model_type'] = 'quadruplets'
results_of_quintuples = get_models_results_df(challengers5, dataset)
results_of_quintuples['model_type'] = 'quintuples'
results_of_sextuples = get_models_results_df(challengers6, dataset)
results_of_sextuples['model_type'] = 'sextuples'

results = pd.concat([results_of_duples, results_of_triplets, results_of_quadruplets, results_of_quintuples, results_of_sextuples])

# %%


# %%
results.groupby('model_type')[['AIC','BIC','log_likelihood']].agg(['mean','min','median'])

# %%
results.sort_values(by='AIC').head(1)

# %%
results.sort_values(by='AIC').head(10)

# %%
## Recreating the best results
best_model = SplineFitter(np.percentile(dataset.time_df.loc[dataset.event_df.astype(bool)], [36, 47, 49]))
best_model = best_model.fit(dataset.time_df.values, dataset.event_df.values, label='Best_Spline_3_knot_36_47_49')


# %%
best_model.summary

# %%
fig = plt.figure()
gs = fig.add_gridspec(1,1)
ax = fig.add_subplot(gs[0,0])
best_model.plot_survival_function(ax=ax)
# overlay Kaplan-Meier on every subplot except when the subplot is the KM itself

if name != 'KaplanMeier':
    models['KaplanMeier'].plot_survival_function(
        ax=ax,
        ci_show=False,
        label='KaplanMeier',
        linestyle='--'
    )

plt.show()


