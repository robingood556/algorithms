import numpy as np
import scipy as sp
import sklearn as sk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from pgmpy.estimators import HillClimbSearch, TreeSearch, K2Score, BicScore
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

import networkx as nx
import pylab

df = pd.read_csv('heart.csv')
print(df.head(10))

df = df[ [
    'age',
    'trestbps',
    'chol',
    'sex',
    'cp',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak'
          ] ]


targets_df = df.iloc[:, :3].copy()
predictors_df = df.iloc[:, 3:].copy()
print('Targets:')
print(targets_df.head(5))


print('Predictors:')
print(predictors_df.head(5))

sns.set_theme(style='whitegrid', palette='pastel')
fig, axes = plt.subplots(1, 3, figsize=(25, 7))
for i, col_name in enumerate(targets_df):
    df = df.dropna()
    sns.histplot(df[col_name], kde=False, bins=30, stat='density', ax=axes[i], palette='Set3', color='blue')
    max_amount = targets_df[col_name].max()
    x = np.linspace(0, max_amount, 1000)

    mle_params = sp.stats.lognorm.fit(df[col_name], loc=1.1, scale=1.1)  # Make initial Guess
    pdf_mle = sp.stats.lognorm.pdf(x, *mle_params)


    axes[i].plot(x, pdf_mle, color='red')

plt.show()

fig, axes = plt.subplots(1, 3, figsize=(25, 7))
targets_df=targets_df.dropna()
params_1 = sp.stats.lognorm.fit(targets_df['age'], loc=1.1, scale=1.1)
x = np.linspace(0.001, 100, 1000)
ppf_1 = sp.stats.lognorm.ppf(x / 100.0, *params_1)
pdf_1 = sp.stats.lognorm.pdf(ppf_1, *params_1)
axes[0].plot(ppf_1, pdf_1 , 'r')
axes[0].set(xlabel='age')
synthetic_data=ppf_1[1:ppf_1.size-1]
sns.histplot(synthetic_data, kde=False, bins=30, stat='density', ax=axes[0], palette='Set3', color = 'blue')

params_2 = sp.stats.chi2.fit(targets_df['trestbps'])
x = np.linspace(0.001, 100, 1000)
ppf_2 = sp.stats.chi2.ppf(x / 100.0, *params_2)
pdf_2 = sp.stats.chi2.pdf(ppf_2, *params_2)
axes[1].plot(ppf_2, pdf_2 , 'r')
axes[1].set(xlabel='trestbps')
synthetic_data=ppf_2[1:ppf_2.size-1]
sns.histplot(synthetic_data, kde=False, bins=30, stat='density', ax=axes[1], palette='Set3', color = 'blue')

params_3 = sp.stats.lognorm.fit(targets_df['chol'], loc=1.1, scale=1.1)
x = np.linspace(0.001, 100, 1000)
ppf_3 = sp.stats.lognorm.ppf(x / 100.0, *params_3)
pdf_3 = sp.stats.lognorm.pdf(ppf_3, *params_3)
axes[2].plot(ppf_3, pdf_3 , 'r')
axes[2].set(xlabel='chol')
synthetic_data=ppf_3[1:ppf_3.size-1]
sns.histplot(synthetic_data, kde=False, bins=30, stat='density', ax=axes[2], palette='Set3', color = 'blue')

plt.show()

def gaussian(x, mu, sigma):
    return 1/np.sqrt(2*np.pi)/sigma * np.exp(-0.5*((x-mu)/sigma)**2)

# Lognormal PDF
def lognormal(x, s):
    return 1/(s*x*np.sqrt(2*np.pi)) * np.exp(-np.log(x)**2/(2*s**2))

fig, axes = plt.subplots(1, 3, figsize=(25, 7))

gaussian_1 = gaussian(ppf_1, 120, 35)
multiplier = 2.4
axes[0].plot(ppf_1, pdf_1 , 'r')
axes[0].plot(ppf_1, multiplier * gaussian_1 , 'b')
axes[0].set(xlabel='age', ylabel='Density')

gaussian_2 = gaussian(ppf_2, 70, 15)
multiplier = 1.2
axes[1].plot(ppf_2, pdf_2 , 'r')
axes[1].plot(ppf_2, multiplier * gaussian_2 , 'b')
axes[1].set(xlabel='trestbps', ylabel='Density')

gaussian_3 = gaussian(ppf_3, 75, 25)
multiplier = 1.8
axes[2].plot(ppf_3, pdf_3 , 'r')
axes[2].plot(ppf_3, multiplier * gaussian_3 , 'b')
axes[2].set(xlabel='chol', ylabel='Density')

plt.show()

fig, axes = plt.subplots(1, 3, figsize=(25, 7))

#number of instances
N = 100000
samples_1 = []
multiplier = 2.4
for _ in range(N):
    candidate = np.random.normal(120, 35)
    prob_accept = sp.stats.lognorm.pdf(candidate, *params_1) / (multiplier * gaussian(candidate, 23, 25))
    #accept with the calculated probability
    if np.random.random() < prob_accept:
        samples_1.append(candidate)

sns.histplot(samples_1, kde=False, bins=30, stat='density', ax=axes[0], palette='Set3', color = 'pink')
axes[0].plot(ppf_1, pdf_1 , 'r')
axes[0].plot(ppf_1, multiplier * gaussian_1 , 'b')
axes[0].set(xlabel='age', ylabel='Density')

samples_2 = []
multiplier = 1.2
for _ in range(N):
    candidate = np.random.normal(70, 15)
    prob_accept = sp.stats.chi2.pdf(candidate, *params_2) / (multiplier * gaussian(candidate, 48, 15))
    #accept with the calculated probability
    if np.random.random() < prob_accept:
        samples_2.append(candidate)

sns.histplot(samples_2, kde=False, bins=30, stat='density', ax=axes[1], palette='Set3', color = 'pink')
axes[1].plot(ppf_2, pdf_2 , 'r')
axes[1].plot(ppf_2, multiplier * gaussian_2 , 'b')
axes[1].set(xlabel='trestbps', ylabel='Density')

samples_3 = []
multiplier = 1.8
for _ in range(N):
    candidate = np.random.normal(75, 25)
    prob_accept = sp.stats.lognorm.pdf(candidate, *params_3) / (multiplier * gaussian(candidate, 64, 45))
    #accept with the calculated probability
    if np.random.random() < prob_accept:
        samples_3.append(candidate)

sns.histplot(samples_3, kde=False, bins=30, stat='density', ax=axes[2], palette='Set3', color = 'pink')
axes[2].plot(ppf_3, pdf_3 , 'r')
axes[2].plot(ppf_3, multiplier * gaussian_3 , 'b')
axes[2].set(xlabel='chol', ylabel='Density')

plt.show()

fig, axes = plt.subplots(1, 1, figsize=(20, 10))
sns.heatmap(df.corr()[['age','trestbps','chol']], cmap='Reds', annot=True)

plt.show()


def create_edges(connections):
    edges = []
    for outlet, inlet in connections.items():
        for inl in inlet:
            edges.append([outlet, inl])

    return edges


connections = {
    'age': [],
    'sex': ['age'],
    'cp': ['age', 'sex'],
    'trestbps': ['cp', 'age'],
    'chol': ['trestbps', 'cp', 'sex', 'age'],
    'fbs': ['age', 'trestbps', 'cp', 'sex','chol'],

}

edges = create_edges(connections)
vertices = ['age', 'sex', 'cp', 'trestbps','chol']
bn = {"V": vertices, "E": edges}

figure, ax = plt.subplots(1, 1, figsize=(10, 10))

graph = nx.DiGraph()
graph.add_edges_from(BayesianModel(edges).edges())

positions = nx.layout.circular_layout(graph)
nx.draw(graph, positions, with_labels=True, node_color='pink', node_size=5000)

plt.show()

df_transformed = df.copy()
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
df_discretized = discretizer.fit_transform(df.values[:])
df_transformed[:] = df_discretized
print(df_transformed)

def accuracy_params_restoration(bn, data):
    bn.fit(data)
    result = pd.DataFrame(columns=['Parameter', 'accuracy'])
    bn_infer = VariableElimination(bn)
    for j, param in enumerate(data.columns):
        accuracy = 0
        test_param = data[param].copy()
        test_data = data.drop(columns=param)
        evidence = test_data.to_dict('records')
        predicted_param = []
        for element in evidence:
            prediction = bn_infer.map_query(variables=[param], evidence=element, show_progress=False)
            predicted_param.append(prediction[param])
        accuracy = accuracy_score(test_param.values, predicted_param)
        result.loc[j,'Parameter'] = param
        result.loc[j, 'accuracy'] = accuracy
    return result

hc_search = HillClimbSearch(data=df_transformed)
hc_k2 = hc_search.estimate(scoring_method=K2Score(df_transformed))
hc_bic = hc_search.estimate(scoring_method=BicScore(df_transformed))

figure, ax = plt.subplots(1, 1, figsize=(10, 10))

graph = nx.DiGraph()
graph.add_edges_from(hc_k2.edges())
positions = nx.layout.circular_layout(graph)
nx.draw(graph, positions, with_labels=True, node_color='pink', node_size=5000)

plt.show()

figure, ax = plt.subplots(1, 1, figsize=(10, 10))

graph = nx.DiGraph()
graph.add_edges_from(hc_bic.edges())
positions = nx.layout.circular_layout(graph)
nx.draw(graph, positions, with_labels=True, node_color='pink', node_size=5000)

plt.show()

def sampling (bn, data, n):
    bn_new = BayesianModel(bn.edges())
    bn_new.fit(data)
    sampler = BayesianModelSampling(bn_new)
    sample = sampler.forward_sample(size=n)
    return sample


def draw_comparative_hist(parametr, original_data, data_sampled, axes=None):
    final_df = pd.DataFrame()

    df1 = pd.DataFrame()
    df1[parametr] = original_data[parametr]
    df1['Data'] = 'Original data'
    df1['Probability'] = df1[parametr].apply(
        lambda x: (df1.groupby(parametr)[parametr].count()[x]) / original_data.shape[0])

    df2 = pd.DataFrame()
    df2[parametr] = data_sampled[parametr]
    df2['Data'] = 'Synthetic data'
    df2['Probability'] = df2[parametr].apply(
        lambda x: (df2.groupby(parametr)[parametr].count()[x]) / data_sampled.shape[0])
    final_df = pd.concat([df1, df2])

    sns.barplot(ax=axes, x=parametr, y="Probability", hue="Data", data=final_df, palette='Reds')

sample_K2 = sampling(hc_k2, df_transformed, df_transformed.shape[0])
sample_Bic = sampling(hc_bic, df_transformed, df_transformed.shape[0])
print(classification_report(df_transformed.sex, sample_K2.sex))
print(classification_report(df_transformed.trestbps, sample_Bic.sex))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

draw_comparative_hist('sex', df_transformed, sample_K2, axes=axes[0])
draw_comparative_hist('sex', df_transformed, sample_Bic, axes=axes[1])

axes[0].set_title('K2_Score')
axes[1].set_title('Bic_Score')
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

draw_comparative_hist('exang', df_transformed, sample_K2, axes=axes[0])
draw_comparative_hist('exang', df_transformed, sample_Bic, axes=axes[1])

axes[0].set_title('K2_Score')
axes[1].set_title('Bic_Score')
plt.legend()
plt.show()