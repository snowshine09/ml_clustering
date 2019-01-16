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
import re
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
x.dropna(thresh=5, inplace=True)

#number of courses, credits are too dirty as many entered text instead of a number
x.drop(columns=["surveyCompletion", "Duration (in seconds)", "NumCourses", "transferCredits", "NumTranCredits",
                "WhyProfile", "otherChannels", "otherEmploy", "NonUS", "otherIndusry", "KnownThroughProfile",
                "otherEth", "PPLinPerson", "otherEmail"], inplace=True)
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

print(df.head())
print(df.columns)
print(df.dtypes)

df['SOC_f'] = df.apply(lambda row: np.mean([row['SOC9'], row['SOC10']]), axis=1)
df['SOC_id'] = df.apply(lambda row: np.mean([row['SOC1'], row['SOC2'], row['SOC3'], row['SOC4'], row['SOC5']]), axis=1)
df['CCE_IR'] = df.apply(lambda row: np.mean([row['CCE13'], row['CCE10'], row['CCE19'], row['CCE9']]), axis=1)
df['CCE_Coor'] = df.apply(lambda row: np.mean([row['CCE24'], row['CCE20'], row['CCE22'], row['CCE21'], row['CCE17']]), axis=1)
df['CCE_SS'] = df.apply(lambda row: np.mean([row['CCE1'], row['CCE2'], row['CCE3'], row['CCE4']]), axis=1)

df.drop(columns=CCEs + SOCs, inplace=True)

print('no error')

#geo distance
# import geopy.distance
#
# coords_1 = (52.2296756, 21.0122287)
# coords_2 = (52.406374, 16.9251681)
# df['newcol'] = df.apply(lambda row: row['firstcolval'] * row['secondcolval'], axis=1)

print('no error1')

#dirty columns include: kids, courses
none_i = re.compile(r'none', flags=re.IGNORECASE)
# df.kids = none_i.sub(r'none\i', df.kids)
df['kids'].replace(none_i, 0, inplace=True)
df.kids = df['kids'].str.extract(r'^(\d+)', expand=False)

print(f'kids are {df.kids.unique()}')
print(f'gender are {df.gender.unique()}')

print(f'industry are {df.industry.unique()}')
print(f'military are {df.Military.unique()}')
# print(f'courses are {df.NumCourses.unique()}')


onehotecoder = ce.OneHotEncoder(cols=["gender", "InUS", "ethnicity", "Usstate", "marrital", "employment", "industry"], handle_unknown='impute')
df = onehotecoder.fit_transform(df)

col_non_num = [c for c in df.columns if df[c].dtype == 'object']

print('no error2')

df.drop(columns=col_non_num, inplace=True)
print(df.shape)
print(df.dtypes)
print(df.head(10))
# fill with mode, mean, or median
df_mode, df_mean, df_median = df.mode().iloc[0], df.mean(), df.median()

df.fillna(df_median, inplace=True)
# data = StandardScaler().fit_transform(df.values)
data = df.values
dend = shc.dendrogram(shc.linkage(data, method='ward'))


cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
print(f'labels are {cluster.labels_}')
print(len(cluster.labels_))



plt.title("Students Dendograms")
plt.figure(figsize=(10, 7))
plt.scatter(data[:,110], data[:,112], c=cluster.labels_, cmap='rainbow') #110 is age, and 112 is progress; 102, 103 is lat and long
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