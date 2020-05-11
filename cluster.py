import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans,Birch
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_excel('2017all(noreg).xlsx')
regions_df = pd.DataFrame.copy(df[['region']])
df.drop(['region', 'Avrpension','Avrsalary', 'Percapconsumspend'], 1, inplace = True)
#df.fillna(method = 'bfill', inplace = True)
df.interpolate(axis = 1, inplace = True)



#with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
#   print(df)

save_df = df.corr()
#export_csv = save_df.to_csv(r'D:\Olq\4 thesis\Try\Correlation.csv', index = None)

labels = [c[:2] for c in save_df.columns]
figr = plt.figure(figsize= (12, 12))
ax = figr.add_subplot(111)
ax.matshow(save_df, cmap = plt.cm.RdYlGn)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.show()

X = np.array(df.astype(float))
#X = preprocessing.scale(X)
X = StandardScaler().fit_transform(X)

clf = KMeans(n_clusters=3, random_state=0, max_iter = 500)
cluster = clf.fit_predict(X)

df['cluster'] = cluster

df = df.join(regions_df)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(df[['region', 'cluster']])
