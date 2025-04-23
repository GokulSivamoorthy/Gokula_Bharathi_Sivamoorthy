import pandas as pd
import numpy as np

"Step 1: Data Loading"
df = pd.read_csv('asd.csv')
print(df.head(2))

"Step 2: Initial Data Exploration"
print(df.shape)
df.info()
print(df.describe())

"Finding null values"
print(df.isnull().sum())

"removing duplicates"
print(df.drop_duplicates(inplace=True))


"Step 3: Data Preprocessing"

"finding values to encode the categorical data with numbers"
print(df['ethnicity'].value_counts())
print(df['relation'].value_counts())
df = df.replace({'yes':1, 'no':0, '?':'Others', 'others':'Others'})
print(df)

"Step 4: Exploratory Data Analysis (EDA)"

import matplotlib.pyplot as plt
import seaborn as sns

plt.pie(df['Class/ASD'].value_counts().values,autopct='%1.1f%%')
plt.show()

"Identify Numeric and Categorical Columns"

num = []
obj = []
dec = []

for col in df.columns:
    if df[col].dtype == int:
        num.append(col)
    elif df[col].dtype == object:
        obj.append(col)
    else:
        dec.append(col)

num.remove('ID')
num.remove('Class/ASD')

"Count Plots for Numerical Features"

melt = df.melt(id_vars=['ID','Class/ASD'],value_vars=num,var_name = 'col',value_name = 'value')

plt.figure(figsize=(15,15))

for i, col in enumerate(num):
    plt.subplot(5,3,i+1)
    sns.countplot(x='value', hue='Class/ASD', data=melt[melt['col'] == col])

plt.tight_layout()
plt.show()

"Count Plots for Categorical Features"

plt.figure(figsize=(15,15))

for i, col in enumerate(obj):
    plt.subplot(5,3,i+1)
    sns.countplot(x=col, hue='Class/ASD', data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

"Country wise comparison"

plt.figure(figsize=(15,5))
sns.countplot(data=df, x='contry_of_res',hue='Class/ASD')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15,5))

"Distribution and Boxplots for Decimal Features"

for i, col in enumerate(dec):
    plt.subplot(1,2,i+1)
    sns.distplot(df[col])

plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))

for i, col in enumerate(dec):
    plt.subplot(1,2,i+1)
    sns.boxplot(df[col])

plt.tight_layout()
plt.show()

"Step 5: Correlation Heatmap"
from sklearn.preprocessing import LabelEncoder

def encode_labels(data):
    for col in data.columns:

        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    return data


df = encode_labels(df)

plt.figure(figsize=(10, 10))
sns.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

"Step 6: Data Modeling (Supervised Learning)"

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


x = df.drop(['ID', 'age_desc', 'used_app_before', 'Class/ASD'],axis=1)
y =df['Class/ASD']

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=10)

random = RandomOverSampler(sampling_strategy='minority',random_state=0)
X, Y = random.fit_resample(X_train,Y_train)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for model in models:
    model.fit(X,Y)
    print(f'{model} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(Y, model.predict(X)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_test, model.predict(X_test)))
    print()

"Step 7: Unsupervised Learning – KMeans Clustering by Country"

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


new_col = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score','A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score','age', 'result']
country = df.groupby('contry_of_res')[new_col].mean().dropna()


scaler = StandardScaler()
scaled_features = scaler.fit_transform(country)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

print(clusters)

country['Cluster'] = clusters

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis', s=100)
for i, country in enumerate(country.index):
    plt.text(reduced_features[i, 0], reduced_features[i, 1], country, fontsize=8)
plt.title('Country-wise Clustering based on Survey Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()