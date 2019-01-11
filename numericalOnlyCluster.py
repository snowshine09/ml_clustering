import pandas as pd
import numpy as np

import category_encoders as ce

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
# Likely = {
#     'Strongly disagree': -2,
#     'Somewhat disagree': -1,
#     'Neither agree nor disagree': 0,
#     'Somewhat agree': 1,
#     'Strongly agree': 2
# }
#
# Importance = {
#     'Not at all important': 0,
#     'A little important': 1,
#     'Somewhat important': 2,
#     'Moderately important': 3,
#     'Fairly important': 4,
#     'Pretty important': 5,
#     'Extremely important': 6
# }

x = pd.read_csv('/Users/nasun/workspace/selenium/wcR.csv') # this is in the Box folder I shared with you
# print(x.shape)
# print(x.columns)
# print(x.dtypes)
x_col = x.columns
col_non_num = [c for c in x_col if x[c].dtype == 'object']

x.drop(columns=col_non_num, inplace=True)


df = x.loc[:, ['age', 'enrollYear', 'progress']]
# fill with mode, mean, or median
df_mode, df_mean, df_median = df.mode().iloc[0], df.mean(), df.median()

df.fillna(df_median, inplace=True)
print(df.shape)
print(df.dtypes)
print(df.head(5))
data = df.values
print(data[0:100, :])

dend = shc.dendrogram(shc.linkage(data, method='ward'))


cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
print(cluster.labels_)


plt.title("Students Dendograms")
plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,2], c=cluster.labels_, cmap='rainbow')
plt.show()
