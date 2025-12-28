# %% [markdown]
# ## Fonte dos dados e dicionário
# 
# [Dados](https://www.kaggle.com/datasets/utkarshx27/breast-cancer-dataset-used-royston-and-altman)
# 
# Será utilizada, neste estudo, a base de dados intitulada "Breast Cancer Dataset 
# used  by  Royston  and  Altman",  disponibilizada  na  plataforma  Kaggle.  Esse 
# conjunto  de  dados  reúne  informações  clínicas de pacientes com diagnóstico 
# confirmado de câncer de mama, incluindo variáveis como idade, estadiamento 
# da doença, presença de metástases, entre outras características relevantes ao 
# prognóstico
# 
# ### Dicionário de dados
# 
# | Coluna     | Descrição                                                                 |
# |------------|---------------------------------------------------------------------------|
# | pid        | Identificador do paciente                                                 |
# | age        | Idade, em anos                                                            |
# | meno       | Status menopausal (0 = pré-menopausa, 1 = pós-menopausa)                 |
# | size       | Tamanho do tumor, em milímetros                                           |
# | grade      | Grau do tumor                                                             |
# | nodes      | Número de linfonodos positivos                                            |
# | pgr        | Receptores de progesterona (fmol/l)                                       |
# | er         | Receptores de estrogênio (fmol/l)                                         |
# | hormon     | Terapia hormonal (0 = não, 1 = sim)                                       |
# | rfstime    | Tempo livre de recorrência; dias até a primeira recorrência, morte ou último acompanhamento |
# | status     | 0 = vivo sem recorrência, 1 = recorrência ou morte                       |
# 
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter, WeibullFitter, LogNormalFitter, CoxPHFitter, ExponentialFitter, NelsonAalenFitter
from lifelines.plotting import qq_plot
from lifelines.statistics import logrank_test
from lifelines.utils import restricted_mean_survival_time, find_best_parametric_model
from lifelines.calibration import survival_probability_calibration

datapath = "data/gbsg.csv"

df = pd.read_csv(datapath)
df.drop(columns=['pid'], inplace=True)

# %%
event_col = ['status']
time_col = ['rfstime']
id_col = ['pid']

# %% [markdown]
# # 1. Analise exploratória dos dados
# 
# Evento em análise.
# Recorrência ou morte
# 
# Censura: 
# Para essa análise exploratória de dados, vale ressaltar que em todos os casos, são pacientes com nódulo positivo de câncer.
# 
# 

# %%
df.head()

# %%
df.isna().sum()

# %%
df.min()

# %%
df.max()

# %% [markdown]
# ## 1.1. Integridade dos dados
# 
# - Verificando dados faltantes
# - Verificando o tipo de dados (original)

# %%
df.shape

# %%
df.isna().sum()

# %%
df.dtypes

# %% [markdown]
# ## 1.2. Explorando as features isoladamente
# 
# Analisando as features isoladamente
# - Idade dos pacientes
# - Status de menopausa
# - Tamanho do tumor
# - Grau do tumor
# - Número de linfonodos positivos
# - Receptores de progesterona
# - Receptores de estrogênio
# - Terapia hormonal
# - Tempo livre de recorrência, dias até a primeira ocorrência, morte ou último acompanhamento
# - Status do evento

# %%
def setup_axes(ax: plt.Axes, title:str, remove_spines: list | str = [], remove_xticks: bool = False, remove_yticks: bool = False, title_loc: str = 'center') -> plt.Axes:
    ax.set_title(title, loc=title_loc)
    if isinstance(remove_spines, list):
        for spine in remove_spines:
            ax.spines[spine].set_visible(False)
    elif isinstance(remove_spines, str):
        if remove_spines == 'all':
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)
    if remove_xticks:
        ax.set_xticks([])
    if remove_yticks:
        ax.set_yticks([])
    return ax

def create_subplot_list(fig:plt.Figure, gs: plt.GridSpec,  slices: list[tuple[slice | int, slice | int]]) -> list[plt.Axes]:
    axes = [None for _ in range(len(slices))]
    for i, s in enumerate(slices):
        axes[i] = fig.add_subplot(gs[s[0], s[1]])
    return axes

# %%
num_cols = ['age','meno','size', 'grade', 'nodes', 'pgr', 'er','hormon','rfstime','status']

for col in num_cols:
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle(col.upper())
    
    gs = fig.add_gridspec(1, 3)
    axes = create_subplot_list(
        fig=fig,
        gs=gs,
        slices= [
            (0, slice(0,2)),
            (0, 2)
        ]
    )

    sns.histplot(df, x=col, ax=axes[0])
    sns.boxplot(df, y=col, ax=axes[1])

    mean = df[col].mean()
    axes[1].axhline(mean, color='red', linestyle='--', label=f"Média:\n{mean:.2f}")
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),  fontsize='small')

    setup_axes(axes[0], title=f"Histograma: {col}", remove_spines=['top','right'], title_loc='left')
    setup_axes(axes[1], title=f"Boxplot: {col}", remove_spines=['top','right'], title_loc='left')

    plt.tight_layout()
    plt.show()
    fig.savefig(fname=f"images/features/{col}_hist_box_plot.png")

# %% [markdown]
# ## 1.2. Explorando a relação entre as features e o status do evento

# %%
# num_cols = ['age','meno','size', 'grade', 'nodes', 'pgr', 'er','hormon','rfstime','status']


# for col in num_cols:
#     fig = plt.figure(figsize=(8, 4))
#     fig.suptitle(col.upper())
#     gs = fig.add_gridspec(1, 1)
#     axes = create_subplot_list(
#         fig=fig,
#         gs=gs,
#         slices= [
#             (0, 0)
#         ]
#     )

#     sns.kdeplot(df, x=col, hue='status',ax=axes[0])
#     setup_axes(axes[0], title=f"Histograma: {col} x status", remove_spines=['top','right'], title_loc='left')

#     plt.tight_layout()
#     plt.show()
#     fig.savefig(fname=f"images/features_x_status/kde_plot_features_x_status_{col}.png")

# %%
## Continuous columns
positions = [
        (0,0),
        (0,1),
        (0,2),
        (1,0),
        (1,1),
        (1,2)
]

columns = ['age','size','nodes', 'pgr','er','rfstime']
fig = plt.figure(figsize= (16, 9))
gs = fig.add_gridspec(2,3)
axes = create_subplot_list(
    fig=fig,
    gs=gs,
    slices=positions
)

fig.suptitle("Gráfico de distribuição x ocorrência do evento")

for i, ax in enumerate(axes):
    # Adiciona histograma
    ax.set_title(f"{columns[i]}")
    sns.histplot(
        data=df,
        x=columns[i],
        hue='status',
        stat='density',   # importante para combinar com a escala da kde
        element='step',   # você pode trocar por 'poly' ou 'bars' se quiser
        common_norm=False,  # para que cada hue tenha sua própria densidade
        palette='muted',
        ax=ax,
        alpha=0.5          # deixa mais transparente para visualizar bem a KDE
    )
    ax.set_yticklabels([])
plt.tight_layout()

# %%
# num_cols = ['age', 'size', 'grade', 'nodes', 'pgr', 'er', 'rfstime', 'status']

# for col in num_cols:
#     fig = plt.figure(figsize=(8, 4))
#     fig.suptitle(col.upper())
#     gs = fig.add_gridspec(1, 1)
#     axes = create_subplot_list(
#         fig=fig,
#         gs=gs,
#         slices=[(0, 0)]
#     )

#     ax = axes[0]

#     # Adiciona histograma
#     sns.histplot(
#         data=df,
#         x=col,
#         hue='status',
#         stat='density',   # importante para combinar com a escala da kde
#         element='step',   # você pode trocar por 'poly' ou 'bars' se quiser
#         common_norm=False,  # para que cada hue tenha sua própria densidade
#         palette='muted',
#         ax=ax,
#         alpha=0.3          # deixa mais transparente para visualizar bem a KDE
#     )

#     # Adiciona KDE
#     # sns.kdeplot(
#     #     data=df,
#     #     x=col,
#     #     hue='status',
#     #     common_norm=False,
#     #     ax=ax
#     # )

#     setup_axes(ax, title=f"{col} x status", remove_spines=['top', 'right'], title_loc='left')

#     plt.tight_layout()
#     plt.show()
#     fig.savefig(fname=f"images/features_x_status_no_kde/kde_hist_plot_features_x_status_{col}.png")

# %%
boolean_cols = ['meno','grade','hormon']

fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 3)
axes = create_subplot_list(
    fig=fig,
    gs=gs,
    slices= [
        (0, 0),
        (0, 1),
        (0, 2)
    ]
)
fig.suptitle("Variáveis categóricas x ocorrência do evento")
for i, ax in enumerate(axes):
    ctab = pd.crosstab(df[boolean_cols[i]], df['status'])

    sns.heatmap(ctab, annot=True, fmt=".0f", cmap="Blues", ax=ax,linecolor='white',linewidths=1)
    ax.set_title(boolean_cols[i])
    
plt.tight_layout()
plt.show()
fig.savefig(fname=f"images/heatmap_plot_cat_features_x_status.png")


# %%
boolean_cols = ['meno', 'grade', 'hormon']

fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 3)
axes = create_subplot_list(
    fig=fig,
    gs=gs,
    slices=[
        (0, 0),
        (0, 1),
        (0, 2)
    ]
)

fig.suptitle("Variáveis categóricas x porcentagem de ocorrência do evento")

for i, ax in enumerate(axes):
    # Tabela cruzada com normalização por coluna (status)
    ctab = pd.crosstab(df[boolean_cols[i]], df['status'], normalize='columns')

    # Gera o heatmap com formatação em porcentagem
    sns.heatmap(ctab, annot=True, fmt=".3f", cmap="Blues", ax=ax,
                linecolor='white', linewidths=1)

    ax.set_title(boolean_cols[i])
    ax.set_xlabel('Status')
    ax.set_ylabel(boolean_cols[i])

plt.tight_layout()
plt.show()
fig.savefig(fname="images/heatmap_plot_cat_features_x_status_percent_normalized.png")

# %% [markdown]
# ### Correlação entre variáveis

# %%
sns.pairplot(df)

plt.show()

# %% [markdown]
# # 2. Análise de sobrevivência

# %% [markdown]
# ## Análise de sobrevivência
# 
# Entender fatores de riscos ou prognósticos para evolução e piora dos casos de câncer de mama
# 
# Caracterização do nosso problema:
# 1. Evento: Recidiva ou morte
# 2. Tempo de estudo:
#     - Tempo inicial: Remoção do tumor
#     - Tempo para evento ou censura: Número de dias desde a cirurgia até evento/censura
# 3. Censura:
#     - Censura do tipo aleatório e à direita: O paciente foi removido do estudo antes da ocorrência do evento (Recuperação ou teve o último acompanhamento). O tempo máximo permitido nesse experimento é de 5 anos. Além disso, o evento de interesse está a direita do tempo registrado

# %%
# Ordena os dados por tempo para um visual mais organizado
df_sorted = df.sort_values(by='rfstime').reset_index(drop=True)

# Define os valores do eixo Y como o índice dos pacientes
y_vals = df_sorted.index
x_vals = df_sorted['rfstime']
status = df_sorted['status']

fig, ax = plt.subplots(figsize=(10, 6))

# Flags para controlar se o label já foi adicionado
added_labels = {'Evento': False, 'Censurado': False}

# Desenha cada linha individualmente com a cor correspondente
for i in range(len(df_sorted)):
    color = '#ff7f0e' if status[i] == 1 else '#1f77b4'

    # Linha
    ax.hlines(y=y_vals[i], xmin=0, xmax=x_vals[i], color=color, alpha=0.7)

    # Bolinha no fim (evento = cheia, censura = vazia)
    if status[i] == 1:  # Evento
        label = 'Status = 1 (Evento)' if not added_labels['Evento'] else None
        added_labels['Evento'] = True
        ax.plot(x_vals[i], y_vals[i], 'o', color=color, label=label)
    else:  # Censura
        label = 'Status = 0 (Censurado)' if not added_labels['Censurado'] else None
        added_labels['Censurado'] = True
        ax.plot(x_vals[i], y_vals[i], 'o', markerfacecolor='none', markeredgecolor=color, label=label)

# Ajustes visuais
ax.set_xlabel('Tempo de sobrevida')
ax.set_ylabel('Indivíduos ordenados')
ax.set_title('Gráfico tipo pirulito: tempo de sobrevida por indivíduo (colorido por status)')
ax.legend(title="Legenda", loc='lower right')
plt.tight_layout()
plt.show()
fig.savefig(fname=f"images/lolipop_plot.png")

# %%
kmf = KaplanMeierFitter()
kmf = kmf.fit(df["rfstime"], event_observed=df["status"])

# %%
fig = plt.figure(figsize=(9,6))
gs = fig.add_gridspec(1,1)
ax = fig.add_subplot(gs[0,0])

kmf.plot_survival_function(at_risk_counts=True,  censor_styles={'ms': 6, 'marker': 's'}, ax=ax)
plt.title("Curva de Sobrevivência - Kaplan-Meier")
plt.ylabel("Probabilidade de Sobrevivência")
plt.show()

fig.savefig(fname=f"images/kaplan_meier_surv_function.png")

# %%
f"Estimativa de tempo {kmf.median_survival_time_} dias, em que 50% dos pacientes não terão tido recorrência ou morte. (Survived)"

# %%
# Considerando tempo máximo = 5 anos
mean_survival_time = restricted_mean_survival_time(kmf, t=1825)
print("Tempo médio de sobrevivência estimado:", mean_survival_time)

# %%
kmf_summary_df = pd.concat([kmf.survival_function_, kmf.confidence_interval_], axis=1)

# %%
kmf_summary_df.tail(8)

# %% [markdown]
# ## Comparando as curvas de sobrevivência considerando diferentes categorias

# %%
from itertools import combinations

# %%
def comparar_sobrevivencia_multigrupo(df, time_col, event_col, cat_col):
    categorias = df[cat_col].dropna().unique()
    resultados = []

    for cat_1, cat_2 in combinations(categorias, 2):
        grupo_1 = df[df[cat_col] == cat_1]
        grupo_2 = df[df[cat_col] == cat_2]

        result = logrank_test(
            durations_A=grupo_1[time_col],
            durations_B=grupo_2[time_col],
            event_observed_A=grupo_1[event_col],
            event_observed_B=grupo_2[event_col]
        )

        resultados.append({
            'cat_1': cat_1,
            'cat_2': cat_2,
            't_0': result._kwargs['t_0'],
            'p_value': result.p_value,
            'test_statistic': result.test_statistic
        })

    return pd.DataFrame(resultados)

# %% [markdown]
# ### FEATURE: Idade

# %%
#bins = [0, 39, 59, float('inf')]
#labels = ['Jovem (<40)', 'Adulto (40–59)', 'Idoso (60+)']

#df['age_cat'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

# %%
#df['age_cat'].value_counts()

# %%
# grupo_col = "age_cat"  # qualquer variável categórica

# fig = plt.figure(figsize=(12,5))
# gs = fig.add_gridspec(1,2)

# ax_0= fig.add_subplot(gs[0,0])

# for grupo in df[grupo_col].unique():
#     kmf = KaplanMeierFitter()
#     mask = df[grupo_col] == grupo
#     kmf.fit(df[mask]["rfstime"], event_observed=df[mask]["status"], label=f"{grupo_col}={grupo}")
#     kmf.plot_survival_function(ax=ax_0,ci_show=False)

# plt.title(f"Curvas de Sobrevivência por {grupo_col}")
# plt.xlabel("Tempo")
# plt.ylabel("Probabilidade de Sobrevivência")
# plt.legend()


# ax_1 = fig.add_subplot(gs[0,1])
# ax_1.set_title("Contagem de pacientes por idade")
# sns.countplot(df, x='age_cat', ax=ax_1)

# plt.show()

# %%
# results_df = comparar_sobrevivencia_multigrupo(df=df, time_col='rfstime',event_col='status',cat_col='age_cat')

# %%
# results_df

# %% [markdown]
# ### FEATURE: Meno

# %%
grupo_col = "meno"  # qualquer variável categórica

fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 2)

# --- Curvas de sobrevivência ---
ax_0 = fig.add_subplot(gs[0, 0])

for grupo in df[grupo_col].unique():
    kmf = KaplanMeierFitter()
    mask = df[grupo_col] == grupo
    kmf.fit(df[mask]["rfstime"], event_observed=df[mask]["status"], label=f"{grupo_col}={grupo}")
    kmf.plot_survival_function(ax=ax_0, ci_show=False)

ax_0.set_title(f"Curvas de Sobrevivência por {grupo_col}")
ax_0.set_xlabel("Tempo")
ax_0.set_ylabel("Probabilidade de Sobrevivência")

ax_0.spines[['right', 'top']].set_visible(False)
ax_0.legend()

# --- Countplot ---
ax_1 = fig.add_subplot(gs[0, 1])
ax_1.set_title("Contagem de pacientes em menopausa")

bars = sns.countplot(df, x='meno', ax=ax_1)

# Adiciona rótulos acima de cada barra
for container in bars.containers:
    bars.bar_label(container, fmt='%d', label_type='edge', padding=3)
ax_1.spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.show()
fig.savefig(fname=f"images/features_surv_diff_test/meno.png")

# %%
results_df = comparar_sobrevivencia_multigrupo(df=df, time_col='rfstime',event_col='status',cat_col='hormon')

# %%
results_df

# %%
print(results_df.to_latex())

# %% [markdown]
# ### FEATURE: Grau

# %%
grupo_col = "grade"  # qualquer variável categórica

fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1,2)

ax_0= fig.add_subplot(gs[0,0])

for grupo in df[grupo_col].unique():
    kmf = KaplanMeierFitter()
    mask = df[grupo_col] == grupo
    kmf.fit(df[mask]["rfstime"], event_observed=df[mask]["status"], label=f"{grupo_col}={grupo}")
    kmf.plot_survival_function(ax=ax_0,ci_show=False)

plt.title(f"Curvas de Sobrevivência por {grupo_col}")
plt.xlabel("Tempo")
plt.ylabel("Probabilidade de Sobrevivência")
ax_0.spines[['right', 'top']].set_visible(False)
ax_0.legend()

ax_1 = fig.add_subplot(gs[0,1])
ax_1.set_title("Contagem de pacientes por grau do tumor")
bars = sns.countplot(df, x='grade', ax=ax_1)
# Adiciona rótulos acima de cada barra
for container in bars.containers:
    bars.bar_label(container, fmt='%d', label_type='edge', padding=3)
ax_1.spines[['right', 'top']].set_visible(False)

plt.show()
fig.savefig(fname=f"images/features_surv_diff_test/test_grade.png")

# %%
results_df = comparar_sobrevivencia_multigrupo(df=df, time_col='rfstime',event_col='status',cat_col='grade')
results_df

# %%
print(results_df.to_latex())

# %% [markdown]
# ### FEATURE: Tratamento hormonal

# %%
grupo_col = "hormon"  # qualquer variável categórica

fig = plt.figure(figsize=(14,6))
gs = fig.add_gridspec(1,2)

ax_0= fig.add_subplot(gs[0,0])

for grupo in df[grupo_col].unique():
    kmf = KaplanMeierFitter()
    mask = df[grupo_col] == grupo
    kmf.fit(df[mask]["rfstime"], event_observed=df[mask]["status"], label=f"{grupo_col}={grupo}")
    kmf.plot_survival_function(ax=ax_0,ci_show=False)

plt.title(f"Curvas de Sobrevivência por {grupo_col}")
plt.xlabel("Tempo")
plt.ylabel("Probabilidade de Sobrevivência")
ax_0.spines[['right', 'top']].set_visible(False)
ax_0.legend()

ax_1 = fig.add_subplot(gs[0,1])
ax_1.set_title("Contagem de pacientes realizando tratamento hormonal")
bars = sns.countplot(df, x=grupo_col, ax=ax_1)
for container in bars.containers:
    bars.bar_label(container, fmt='%d', label_type='edge', padding=3)
ax_1.spines[['right', 'top']].set_visible(False)
plt.show()
fig.savefig(fname=f"images/features_surv_diff_test/test_hormon.png")

# %%
results_df = comparar_sobrevivencia_multigrupo(df=df, time_col='rfstime',event_col='status',cat_col=grupo_col)
results_df

# %%
print(results_df.to_latex())

# %% [markdown]
# ## Métodos paramétricos

# %%
# Dados
T = df["rfstime"]           # tempo até o evento ou censura
E = df["status"]            # 1 = evento ocorreu, 0 = censurado

kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)

# Exponencial
print ("======")
exp = ExponentialFitter()
exp.fit(T, event_observed=E)
exp.print_summary()

# Weibull
print ("======")
wf = WeibullFitter()
wf.fit(T, event_observed=E)
wf.print_summary()

# Log-Normal
print ("======")
lnf = LogNormalFitter()
lnf.fit(T, event_observed=E)
lnf.print_summary()

# %%
fig = plt.figure(figsize=(12,7))
gs = fig.add_gridspec(1,1)
ax = fig.add_subplot(gs[0,0])

wf.plot_survival_function(label="Weibull", ax=ax,ci_show=False)
exp.plot_survival_function(label="Exponencial", ax=ax, ci_show=False)
lnf.plot_survival_function(label="Log-Normal", ax=ax,ci_show=False)
kmf.plot_survival_function(label="Kaplan-Meier", ax=ax,ci_show=False)

plt.title("Funções de sobrevivência ajustadas")
plt.legend()
plt.show()
fig.savefig(fname=f"images/parametric/all_models.png")

# %%
fig = plt.figure(figsize=(20,10))
gs = fig.add_gridspec(2,2)

ax0 = fig.add_subplot(gs[0,0])
ax0.set_title("Kaplan-Meier")
kmf.plot_survival_function(label="Kaplan-Meier", ax=ax0,ci_show=True)

ax1 = fig.add_subplot(gs[0,1])
ax1.set_title("Exponencial")
kmf.plot_survival_function(label="Kaplan-Meier", ax=ax1,ci_show=False)
exp.plot_survival_function(label="Exponencial", ax=ax1, ci_show=True, color='orange')


ax2 = fig.add_subplot(gs[1,0])
ax2.set_title("Log-Normal")
kmf.plot_survival_function(label="Kaplan-Meier", ax=ax2,ci_show=False)
lnf.plot_survival_function(label="Log-Normal", ax=ax2,ci_show=True, color='green')


ax3 = fig.add_subplot(gs[1,1])
ax3.set_title("Weibull")
kmf.plot_survival_function(label="Kaplan-Meier", ax=ax3,ci_show=False)
wf.plot_survival_function(label="Weibull", ax=ax3,ci_show=True, color='red')

plt.tight_layout()
plt.legend()
plt.show()

fig.savefig(fname=f"images/parametric/all_models_separeted.png")

# %%
# Tempos onde a função de sobrevivência é definida
times = kmf.survival_function_.index

# Valores das funções de sobrevivência
S_km = kmf.survival_function_["KM_estimate"].values
S_exp = exp.survival_function_at_times(times).values
S_wf = wf.survival_function_at_times(times).values
S_logn = lnf.survival_function_at_times(times).values


# %%
# KM vs Exponential
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(1,3)
ax_0 = fig.add_subplot(gs[0,0])

ax_0.scatter(S_km, S_wf, alpha=0.6, label="Exponential vs KM", color='orange')
ax_0.plot([0, 1], [0, 1], 'k--', label='y = x')
ax_0.set_xlabel("Kaplan-Meier")
ax_0.set_ylabel("Exponential")
ax_0.set_title("Comparação da função de sobrevivência:\nKM vs Exponential")
ax_0.legend()
ax_0.grid(True)

# KM vs Log-Normal
ax_1 = fig.add_subplot(gs[0,1])
ax_1.scatter(S_km, S_logn, alpha=0.6, label="Log-Normal vs KM", color='green')
ax_1.plot([0, 1], [0, 1], 'k--', label='y = x')
ax_1.set_xlabel("Kaplan-Meier")
ax_1.set_ylabel("Log-Normal")
ax_1.set_title("Comparação da função de sobrevivência:\nKM vs Log-Normal")
ax_1.legend()
ax_1.grid(True)

# KM vs Exponencial
ax_2 = fig.add_subplot(gs[0,2])
ax_2.scatter(S_km, S_exp, alpha=0.6, label="Exponencial vs KM", color='red')
ax_2.plot([0, 1], [0, 1], 'k--', label='y = x')
ax_2.set_xlabel("Kaplan-Meier")
ax_2.set_ylabel("Exponencial")
ax_2.set_title("Comparação da função de sobrevivência:\nKM vs Exponencial", loc='left', )
ax_2.legend()
ax_2.grid(True)

plt.tight_layout()
plt.show()

fig.savefig(fname=f"images/parametric/models_x_km.png")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 8))
axes = axes.reshape(3,)

fig.suptitle("QQ-Plots")
for i, model in enumerate([lnf, wf, exp]):
    qq_plot(model, ax=axes[i])
    
fig.savefig(fname=f"images/parametric/qq-plot.png")

# %%
best_model, best_aic_ = find_best_parametric_model(T, E, scoring_method="AIC")

print(best_model)

best_model.plot_hazard()

# %% [markdown]
# ## Modelos paramétricos - AFT

# %%
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter, LogNormalAFTFitter, GeneralizedGammaRegressionFitter
from lifelines.utils import k_fold_cross_validation

# %%
# wft = WeibullAFTFitter()
# loglogistic = LogLogisticAFTFitter()
# lognormal = LogNormalAFTFitter()
# gammareg = GeneralizedGammaRegressionFitter()

# wft = wft.fit(df, 'rfstime', 'status', ancillary=True)
# loglogistic = loglogistic.fit(df, 'rfstime', 'status', ancillary=True)
# lognormal = lognormal.fit(df, 'rfstime', 'status', ancillary=True)
# gammareg = gammareg.fit(df, 'rfstime', 'status', ancillary=True)

# %% [markdown]
# ## Regressão de Cox

# %%
from lifelines.statistics import proportional_hazard_test

# %% [markdown]
# ### Incorporando todas as features

# %%
if 'age_cat' in df.columns:
    df = df.drop(columns='age_cat')

# %%
cph = CoxPHFitter()
cph.fit(df, duration_col='rfstime', event_col='status')

# %%
cph.print_summary()

# %%
results = proportional_hazard_test(cph, df, time_transform='rank')
results.print_summary()

# %% [markdown]
# Avaliando as features com maior HR

# %%
cph = CoxPHFitter()
cph.fit(df, duration_col='rfstime', event_col='status')
cph.check_assumptions(df, p_value_threshold=0.05, show_plots=True)

# %%
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(1,1)
ax = fig.add_subplot(gs[0,0])

ax.set_title("Hazard ratio")
cph.plot(ax=ax)
fig.savefig("images/cox/all_features_hazard_ratios_strat.png")

# %%
## Comparando com KMf
kmf = KaplanMeierFitter()
kmf.fit(df["rfstime"], event_observed=df["status"], label=f"Kaplan Meier")

# %%
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(1,1)
ax=fig.add_subplot(gs[0,0])

ax.set_title("Efeito parcial na saída: Menopausa")
cph.plot_partial_effects_on_outcome(covariates='meno', values=[0, 1], ax=ax)
fig.savefig(f"images/cox/partial_meno.png")

# %%
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(1,1)
ax=fig.add_subplot(gs[0,0])

ax.set_title("Efeito parcial na saída: Menopausa")
cph.plot_partial_effects_on_outcome(covariates='meno', values=[0, 1], ax=ax)
kmf.plot_survival_function(ax=ax, ci_show=False, color='red', linestyle='--')
fig.savefig(f"images/cox/partial_meno_with_kmf.png")

# %%
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(1,1)
ax=fig.add_subplot(gs[0,0])

ax.set_title("Efeito parcial na saída: Grade")
cph.plot_partial_effects_on_outcome(covariates='grade', values=[1,2,3], ax=ax)

plt.show()
fig.savefig(f"images/cox/partial_grade.png")

# %%
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(1,1)
ax=fig.add_subplot(gs[0,0])

ax.set_title("Efeito parcial na saída: Grade")
cph.plot_partial_effects_on_outcome(covariates='grade', values=[1,2,3], ax=ax)
kmf.plot_survival_function(ax=ax, ci_show=False, color='red', linestyle='--')
plt.show()
fig.savefig(f"images/cox/partial_grade_with_kmf.png")

# %%
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(1,1)
ax=fig.add_subplot(gs[0,0])

ax.set_title("Efeito parcial na saída: Hormonal")
cph.plot_partial_effects_on_outcome(covariates='hormon', values=[0,1], ax=ax)
plt.show()
fig.savefig(f"images/cox/partial_hormon.png")

# %%
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(1,1)
ax=fig.add_subplot(gs[0,0])

ax.set_title("Efeito parcial na saída: Hormonal")
cph.plot_partial_effects_on_outcome(covariates='hormon', values=[0,1], ax=ax)
kmf.plot_survival_function(ax=ax, ci_show=False, color='red', linestyle='--')
plt.show()
fig.savefig(f"images/cox/partial_hormon_with_kmf.png")

# %% [markdown]
# ### Estratificado
# 

# %%
cph = CoxPHFitter()
cph.fit(df, duration_col='rfstime', event_col='status', strata=['grade','nodes','age'])
cph.check_assumptions(df, p_value_threshold=0.05, show_plots=True)

# %%
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(1,1)
ax = fig.add_subplot(gs[0,0])

ax.set_title("Hazard ratio")
cph.plot(ax=ax)
fig.savefig("images/cox/strat_all_features_hazard_ratios_strat.png")


