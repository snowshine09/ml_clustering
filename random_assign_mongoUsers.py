from pymongo import MongoClient
from random import randint
from bson.json_util import dumps
import pandas as pd
import numpy as np
import re
import category_encoders as ce
from scipy.spatial.distance import pdist,squareform, jaccard, cosine
# import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

client = MongoClient(port=27017)
db = client.weare
student_queries = db.students.find({})
students = pd.DataFrame(list(student_queries))
print(f'students info: {students.shape}')
print(students.columns)
print(students.dtypes)
print(students.head(5))
# remove = re.compile(r'none', flags=re.IGNORECASE)
students['id']=students.Email.str.split("@", n=1, expand=True)[0]

# df['kids'].replace(none_i, 0, inplace=True)
del students['_id']

students.dropna(thresh=5, inplace=True)
students = students.replace('', np.nan)
#number of courses, credits are too dirty as many entered text instead of a number
students.drop(columns=["surveyCompletion", "Duration (in seconds)", "NumCourses", "transferCredits", "NumTranCredits",
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

df = encoder.fit_transform(students)

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
print(f'Before None processing, kids are {df.kids.unique()}')
print('no error1')

#dirty columns include: kids, courses
none_i = re.compile(r'none', flags=re.IGNORECASE)
df['kids'].replace(none_i, 0, inplace=True)
none_i = re.compile(r'zero', flags=re.IGNORECASE)
df['kids'].replace(none_i, 0, inplace=True)
stringany = re.compile(r'[a-zA-Z ()]+', flags=re.IGNORECASE)
print(f'Before digit processing, kids are {df.kids.unique()}')
df.kids = df['kids'].replace(stringany, '')

print(f'kids are {df.kids.unique()}')
print(f'gender are {df.gender.unique()}')

print(f'industry are {df.industry.unique()}')
print(f'military are {df.Military.unique()}')
# print(f'courses are {df.NumCourses.unique()}')

categorical_cols = ["gender", "InUS", "ethnicity", "Usstate", "marrital", "employment", "industry"]
df_c_mode = df[categorical_cols].mode()
print(f'mode listed are {df_c_mode.iloc[0]}')
print(len(df_c_mode))
# df.loc[:, categorical_cols].fillna(df_c_mode.iloc[0], inplace=True)
for col in categorical_cols+['kids']:
    df[col].fillna(df[col].mode().iloc[0], inplace=True)
print(f'kids are {df.kids.unique()}')
print(f'gender are {df.gender.unique()}')
onehotecoder = ce.OneHotEncoder(cols=categorical_cols, handle_unknown='ignore')
df = onehotecoder.fit_transform(df)

col_non_num = [c for c in df.columns if df[c].dtype == 'object']

print('no error2')

df.drop(columns=col_non_num, inplace=True)


# fill with mode, mean, or median
df_mode, df_mean, df_median = df.mode().iloc[0], df.mean(), df.median()

df.fillna(df_median, inplace=True)
print(df.shape)
print(df.dtypes)
print(df.columns)
print(df.head(10))
print(df.describe().to_string())
data = StandardScaler().fit_transform(df.values)


# data = df.values

A_sparse = sparse.csr_matrix(data)

similarities = cosine_similarity(A_sparse)
print('pairwise dense output:\n {}\n'.format(similarities))

#also can output sparse matrices
similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarities_sparse))

print('the matrix of distance using Euclidean------------')

euclidean_matrix = pd.DataFrame(squareform(pdist(data, metric='euclidean')), columns=students['id'], index=students['id'])
print(euclidean_matrix)
print(euclidean_matrix.shape)
print(euclidean_matrix.describe().to_string())

print('the matrix of distance using Cosine*****************')
print(squareform(pdist(data, metric='cosine')))
cosine_matrix = pd.DataFrame(squareform(pdist(data, metric='cosine')), columns=students['id'], index=students['id'])
print(cosine_matrix.shape)
print(cosine_matrix.describe().to_string())


cos = cosine_matrix.to_dict(orient='records')  # Here's our added param..

db.cosine_matrix.insert_many(cos)

euc = euclidean_matrix.to_dict(orient='records2q34')  # Here's our added param..

db.euclidean_matrix.insert_many(euc)

