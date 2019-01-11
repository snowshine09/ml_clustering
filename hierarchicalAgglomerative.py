# from pymongo import MongoClient

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import category_encoders as ce
# client = MongoClient(port=27017)
# db = client.weare


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

x = pd.read_csv('/Users/nasun/workspace/selenium/wcR.csv')
# print(x.shape)
# print(x.columns)
# print(x.dtypes)
x_col = x.columns
col_non_num = [c for c in x_col if x[c].dtype == 'object']

x.drop(columns = ["WhyProfile", "otherChannels", "otherEmploy", "NonUS", "otherIndusry"], inplace=True)
SOCs = ['SOC'+str(x+1) for x in range(10)]
CCEs = ['CCE'+str(x+1) for x in range(24)]
ordinal_cols_mapping = [
]

important_scale = [
        ('Extremely important', 7),
        ('Pretty important', 6),
        ('Fairly important', 5),
        ('Moderately important', 4),
        ('Somewhat important', 3),
        ('A little important', 2),
        ('Not at all important', 1)
]

interest_scale = [
    ('Extremely interested', 5),
    ('Rather interested', 4),
    ('Somewhat interested', 3),
    ('A bit of interest', 2),
    ('Not at all interested', 1)
]

Agree_scale = [
        ('Strongly agree', 5),
        ('Somewhat agree', 4),
        ('Neither agree nor disagree', 3),
        ('Somewhat disagree', 2),
        ('Strongly disagree', 1)
]
for SOC in SOCs:
    ordinal_cols_mapping.append({
        "col": SOC,
        "mapping": Agree_scale
    })

for CCE in CCEs:
    ordinal_cols_mapping.append({
        "col": CCE,
        "mapping": Agree_scale
    })

for PeerInfo in ["PeerAvail", "PeerProfession", "PeerEdu", "PeerDetails"]:
    ordinal_cols_mapping.append({
        "col": PeerInfo,
        "mapping": important_scale
    })

ordinal_cols_mapping.append({
    "col": "Mconnected",
    "mapping": interest_scale
})

#marrital status, employment, degree, grade
encoder = ce.OrdinalEncoder(mapping=ordinal_cols_mapping, return_df=True)
df = encoder.fit_transform(x)

print(df.shape)
print(df.dtypes)
print(df.head(5))
# df = x.loc[:, ['age', 'enrollYear', 'progress']]
# fill with mode, mean, or median
df_mode, df_mean, df_median = df.mode().iloc[0], df.mean(), df.median()

df.fillna(df_median, inplace=True)
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

# cols_SOC_CCE = x.filter(regex=r'(SOC|CCE)\d+')



# for c in col_non_num:
#     x[c] = OneHotEncoder(handle_unknown='ignore').fit_transform(x[c])
# print(x.head(3))
# print(x.dtypes)

# cols_SOC_CCE = OrdinalEncoder.fit_transform(cols_SOC_CCE).toarray()
# print(x.columns)
# for doc in db.students.find({}):
#     obj = doc.copy()
#     for i in range(10):
#         obj['SOC'+ str(i+1)] = convertLikert(5, )
#     db.students.update_one({'_id': obj._id}, {'$set': obj})