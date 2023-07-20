# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:56:12 2023

@author: janya
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import model_selection
from sklearn import tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import pandas as pd
import seaborn as sns
import random
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from umap import UMAP
from sklearn import cluster # Clustering algorithms such as K-means and agglomerative
from sklearn.metrics import silhouette_score


random.seed(19123302)
data = pd.read_csv("musicData.csv", delimiter=",")

#%% Preparing Data For Classification


for column in data.columns:
    #there are also question marks scattered randomly throughout
    data[column] = data[column].replace(["?"], np.NaN)
cleandata = data.dropna(inplace = False)


predcols = np.unique(cleandata["music_genre"])
#need to create new variables to represent each of the classes of music genre
cleandata["MG"] = 0
i = 1
for genre in predcols:
    cleandata["MG"] = np.where(cleandata["music_genre"] == genre, i, cleandata["MG"])
    i += 1
    

#need to create new variables for mode of music
cleandata["mode"] = np.where(cleandata["mode"] == "Major", 1, 0)

#need to deal with a key as well
cleandata[['Letter', 'Sharp']] = cleandata['key'].str.split('', 3, expand =True).iloc[:,1:3]

for l in np.unique(cleandata["Letter"]):
    cleandata[l] = np.where(cleandata["Letter"] == l, 1, 0)
    
    
cleandata["Sharp"] = np.where(cleandata["Sharp"] == "#", 1, 0)

cleandata0 = cleandata

#%%
#Prepping Data

#create train test split
testt = np.array([cleandata.columns], dtype = object)

trainn = np.array([cleandata.columns], dtype = object)


for i in range(len(predcols)):
    filtered_label = cleandata.loc[cleandata["music_genre"] == predcols[i]]
    filt_train, filt_test = model_selection.train_test_split(filtered_label, test_size = 0.1, random_state=19123302)
    testt= np.concatenate((testt, filt_test),axis = 0)
    trainn= np.concatenate((trainn, filt_train),axis = 0)

trainn = pd.DataFrame(trainn[1:],columns=cleandata.columns)
testt = pd.DataFrame(testt[1:],columns=cleandata.columns)

trainn["duration_ms"] = np.where(trainn["duration_ms"] == -1, np.mean(trainn["duration_ms"]), trainn["duration_ms"])
testt["duration_ms"] = np.where(testt["duration_ms"] == -1, np.mean(trainn["duration_ms"]), testt["duration_ms"])

zscoring = ['popularity',
       'acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness','liveness', 'loudness', 'speechiness', 'tempo', 'valence']
for col in zscoring:
    trainn[col] = trainn[col].astype(float)
    testt[col] = testt[col].astype(float)

    mean = sum(trainn[col]) / len(trainn[col])
    differences = [(value - mean)**2 for value in trainn[col]]
    sum_of_differences = sum(differences)
    standard_deviation = (sum_of_differences / (len(trainn[col]) - 1)) ** 0.5
    trainn[col] = [(value - mean) / standard_deviation for value in trainn[col]]
    testt[col] = [(value - mean) / standard_deviation for value in testt[col]]



##Now before I go on, I want to create a mapping from artist name to their most common genre of their previous songs
## previous songs based on X_train to avoid leakage
mapp = []
artists = np.unique(trainn['artist_name'])
for art in artists:
    types = trainn.loc[trainn['artist_name'] == art]["MG"]
    most = np.unique(types, return_counts=True)[0][0]
    if art == '!!!':
        most = 0
    mapp += [[art, most]]
    
mapp = pd.DataFrame(mapp, columns = ["artist_name", "most common genre"])

trainn = pd.merge(mapp, trainn, how = "right", on = "artist_name")
  
testt = pd.merge(mapp, testt, how = "right", on = "artist_name")
testt["most common genre"].fillna(0, inplace = True)
#%%

Y_train = trainn["MG"]
X_train = trainn.drop(['instance_id', 'artist_name', 'track_name', 'key', 'obtained_date', 'Letter', 'MG' ], axis=1)


Y_test = testt["MG"]
X_test = testt.drop(['instance_id', 'artist_name', 'track_name', 'key', 'obtained_date', 'Letter', 'MG' ], axis=1)



X_train_gen = X_train["music_genre"]

del X_train["music_genre"]
del X_test["music_genre"]

X_train=X_train.astype(float)
X_test=X_test.astype(float)
Y_train = Y_train.astype(float)
Y_test = Y_test.astype(float)


#%% Some Data Exploration
cov = np.cov(X_train.T)
eigvals, eigvec = np.linalg.eig(cov)
print(eigvals)

sns.set(rc={"figure.figsize": (15, 10)})
corr_matrix = trainn.corr()
sns.heatmap(corr_matrix, annot=False)

#%%

lda = LinearDiscriminantAnalysis(n_components = 9).fit(X_train, Y_train)
perc_eigVals = lda.explained_variance_ratio_
rotatedData_train = lda.fit_transform(X_train, Y_train)
rotatedData_test = lda.transform(X_test)


plt.figure()
numClasses = 9
plt.bar(np.linspace(1,9,9),perc_eigVals)
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.title("Linear Discriminant Analysis from 21 to 9 Components")
plt.show()

rotatedData2_train = pd.DataFrame(rotatedData_train)
rotatedData2_train["Genre"] = X_train_gen
    
plt.figure()
plt.title("2 Dimensional Data from LDA")
plt.plot(rotatedData_train[:,0],rotatedData_train[:,1],'o',markersize=5)
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.figure()
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(predcols))]
for i in range(len(predcols)):
    filtered_label = rotatedData2_train.loc[rotatedData2_train.Genre == predcols[i]]
    plt.scatter(filtered_label[0] , filtered_label[1], color = colors[i], label = predcols[i])
plt.legend()
plt.title("Components 1, 2 from LDA")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

plt.figure()
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(predcols))]
for i in range(len(predcols)):
    filtered_label = rotatedData2_train.loc[rotatedData2_train.Genre == predcols[i]]
    plt.scatter(filtered_label[1] , filtered_label[2], color = colors[i], label = predcols[i])
plt.legend()
plt.title("Components 2, 3 from LDA")
plt.xlabel('Component 2')
plt.ylabel('Component 3')
plt.show()

plt.figure()
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(predcols))]
for i in range(len(predcols)):
    filtered_label = rotatedData2_train.loc[rotatedData2_train.Genre == predcols[i]]
    plt.scatter(filtered_label[0], filtered_label[2], color = colors[i], label = predcols[i])
plt.legend()
plt.title("Components 1, 3 from LDA")
plt.xlabel('Component 1')
plt.ylabel('Component 3')
plt.show()
# Creating figure

fig = plt.figure(figsize = (15, 9))
ax = plt.axes(projection ="3d")
 
# Creating plot
for i in range(len(predcols)):
    filtered_label = rotatedData2_train.loc[rotatedData2_train.Genre == predcols[i]]
    ax.scatter3D(filtered_label[0] , filtered_label[1], filtered_label[2], color = colors[i], label=predcols[i])
plt.legend()
plt.title("Components 1, 2, 3 from LDA")
ax.set_xlabel('Component 1', fontweight ='bold')
ax.set_ylabel('Component 2', fontweight ='bold')
ax.set_zlabel('Component 3', fontweight ='bold')
plt.show()

fig = plt.figure(figsize = (15, 9))
ax = plt.axes(projection ="3d")
 
# Creating plot
for i in range(len(predcols)):
    filtered_label = rotatedData2_train.loc[rotatedData2_train.Genre == predcols[i]]
    ax.scatter3D(filtered_label[2] , filtered_label[3], filtered_label[4], color = colors[i], label=predcols[i])
plt.legend()
plt.title("Components 3, 4, 5 from LDA")
ax.set_xlabel('Component 3', fontweight ='bold')
ax.set_ylabel('Component 4', fontweight ='bold')
ax.set_zlabel('Component 5', fontweight ='bold')
plt.show()


#%% Trying K-Means

silhouettes = np.zeros(11)

# Same code as before.
for k in range(2, 12):
    print(k)
    kmeans = cluster.KMeans(k, n_init = 20, random_state=19123302)
    labels = kmeans.fit_predict(rotatedData_train[:,:5])
    silhouettes[k-1] = silhouette_score(rotatedData_train[:,:5], labels)
    
plt.plot(np.arange(2, 13, 1), silhouettes, 'r-o', lw = 2)
plt.xlabel(r'Clusters $k$')
plt.ylabel(r'Silhouette Average')
plt.title(r'Silhouette Curve for K-means')
plt.grid()
plt.show()


data_2d = pd.DataFrame(rotatedData_train[:,:5])

k = 4
kmeans = cluster.KMeans(k, random_state=19123302)
data_2d["label"] = kmeans.fit_predict(data_2d)
cents = kmeans.cluster_centers_
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(predcols))]
for i in [0,1,2,3]:
 
    filtered_label = data_2d.loc[data_2d.label == i]
    ax.scatter3D(filtered_label[0] , filtered_label[1], filtered_label[2], color = colors[i])
plt.title("KMeans on LDA Data")
ax.set_xlabel('Component 1', fontweight ='bold')
ax.set_ylabel('Component 2', fontweight ='bold')
ax.set_zlabel('Component 3', fontweight ='bold')
plt.show()

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
for i in [0,1,2,3]:
 
    filtered_label = data_2d.loc[data_2d.label == i]
    ax.scatter3D(filtered_label[2] , filtered_label[3], filtered_label[4], color = colors[i])
plt.title("KMeans on LDA Data")
ax.set_xlabel('Component 3', fontweight ='bold')
ax.set_ylabel('Component 4', fontweight ='bold')
ax.set_zlabel('Component 5', fontweight ='bold')
plt.show()


#%% #Just to get an idea of model performance on original data


import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


Y_train = pd.DataFrame(Y_train.astype(float))
Y_test = pd.DataFrame(Y_test.astype(float))
'''

bdt = AdaBoostClassifier(
    tree.DecisionTreeClassifier(max_depth=4), algorithm="SAMME", n_estimators=75, learning_rate=0.5)
bdt.fit(X_train, Y_train)


pred = bdt.predict(X_test)

f1_true = metrics.f1_score(Y_test, pred, average=None)
print("F1 score of adaBoost Classifier: ", f1_true)
print("Accuracy Score of Model: ", metrics.accuracy_score(Y_test, pred))

print()

y_pred_proba = bdt.predict_proba(X_test)

for i in range(0,10):
    y_tst = np.where(Y_test["MG"] == i+1, 1, 0)
    auc_true = metrics.roc_auc_score(y_tst, y_pred_proba[:, i])
    fpr, tpr, _ = metrics.roc_curve(y_tst,  y_pred_proba[:, i])
    fig, ax = plt.subplots()
    plt.plot(fpr,tpr, label="AUC="+str(auc_true))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC Curve for AdaBoost Classifier: " +str( i+1))
    ax.axline([0, 0], [1, 1], color = "black", linestyle = "dotted")
    ax.axline([0, 0], [0, 1], color = "black", linestyle = "dotted")
    ax.axline([0, 1], [1, 1], color = "black", linestyle = "dotted")
    plt.xlim(-0.01, 1.01) 
    plt.ylim(-0.01, 1.01)
    plt.legend()
    plt.show()
'''
#%%

#So now I'm going to do a grid search for the adaBoostClassifier on the rotated data from LDA


results1 = []

for md in range(3,5):
    for ne in range(40,80, 7):
        for lr in np.linspace(0.3, 1, 6):
            bdt = AdaBoostClassifier(
                tree.DecisionTreeClassifier(max_depth=md), algorithm="SAMME", n_estimators=ne, learning_rate=lr)
            bdt.fit(X_train, Y_train)
            pred = bdt.predict(X_test)
            y_pred_proba = bdt.predict_proba(X_test)
            auc=[]
            for i in range(0,10):
                y_tst = np.where(Y_test["MG"] == i+1, 1, 0)
                auc_true = metrics.roc_auc_score(y_tst, y_pred_proba[:, i])
                auc += [auc_true]
            f1_true = metrics.f1_score(Y_test, pred, average=None)
            acc = metrics.accuracy_score(Y_test, pred)
            results1 += [["OG Data", md, ne, lr, f1_true, acc, auc]]

rotated_X_train = rotatedData_train[:,:5]
rotated_X_test = rotatedData_test[:,:5]


for md in range(3,5):
    for ne in range(40,80, 7):
        for lr in np.linspace(0.3, 1, 6):
            bdt = AdaBoostClassifier(
                tree.DecisionTreeClassifier(max_depth=md), algorithm="SAMME", n_estimators=ne, learning_rate=lr)
            bdt.fit(rotated_X_train, Y_train)
            pred = bdt.predict(rotated_X_test)
            y_pred_proba = bdt.predict_proba(rotated_X_test)
            auc=[]
            for i in range(0,10):
                y_tst = np.where(Y_test["MG"] == i+1, 1, 0)
                auc_true = metrics.roc_auc_score(y_tst, y_pred_proba[:, i])
                auc += [auc_true]
            f1_true = metrics.f1_score(Y_test, pred, average=None)
            acc = metrics.accuracy_score(Y_test, pred)
            results1 += [["LDA Data", md, ne, lr, f1_true, acc, auc]]
            
'''            
rotated_X_train = rotatedData2_train.iloc[:,:3]
rotated_X_test = rotatedData2_test.iloc[:,:3]

            
for md in range(3,5):
    for ne in range(40,80, 7):
        for lr in np.linspace(0.3, 1, 6):
            bdt = AdaBoostClassifier(
                tree.DecisionTreeClassifier(max_depth=md), algorithm="SAMME", n_estimators=ne, learning_rate=lr)
            bdt.fit(rotated_X_train, Y_train)
            pred = bdt.predict(rotated_X_test)
            y_pred_proba = bdt.predict_proba(rotated_X_test)
            auc=[]
            for i in range(0,10):
                y_tst = np.where(Y_test["MG"] == i+1, 1, 0)
                auc_true = metrics.roc_auc_score(y_tst, y_pred_proba[:, i])
                auc += [auc_true]
            f1_true = metrics.f1_score(Y_test, pred, average=None)
            acc = metrics.accuracy_score(Y_test, pred)
            results1 += [["UMAP Data", md, ne, lr, f1_true, acc, auc]]
'''            
            
res = pd.DataFrame(results1, columns = ["Set of Data", "Max Depth", "N_estimators", "Learning Rate", "f1", "acc", "AUC"])

#%%
#lets calculate the means of the f1 scores and AUC to easily compare models
auc_a = []
f1_a = []
for i in results1:
    auc_a += [np.mean(i[6])]
    f1_a += [np.mean(i[4])]
    
res["Avg. F1"] = f1_a    
res["Avg. AUC"] = auc_a

#%% final model


Y_train = pd.DataFrame(Y_train.astype(float))
Y_test = pd.DataFrame(Y_test.astype(float))


bdt = AdaBoostClassifier(
    tree.DecisionTreeClassifier(max_depth=4), algorithm="SAMME", n_estimators=47, learning_rate=0.3)
bdt.fit(X_train, Y_train)


pred = bdt.predict(X_test)

f1_true = metrics.f1_score(Y_test, pred, average=None)
print("F1 score of adaBoost Classifier: ", f1_true)
print("Accuracy Score of Model: ", metrics.accuracy_score(Y_test, pred))

print()
ig, ax = plt.subplots()
y_pred_proba = bdt.predict_proba(X_test)

for i in range(0,10):
    y_tst = np.where(Y_test["MG"] == i+1, 1, 0)
    auc_true = metrics.roc_auc_score(y_tst, y_pred_proba[:, i])
    fpr, tpr, _ = metrics.roc_curve(y_tst,  y_pred_proba[:, i])
    
    plt.plot(fpr,tpr, label=predcols[i]+" -> "+str(round(auc_true,3)))

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ROC Curve for Best AdaBoost Classifier for Class -> AUC")
ax.axline([0, 0], [1, 1], color = "black", linestyle = "dotted")
ax.axline([0, 0], [0, 1], color = "black", linestyle = "dotted")
ax.axline([0, 1], [1, 1], color = "black", linestyle = "dotted")
plt.xlim(-0.01, 1.01) 
plt.ylim(-0.01, 1.01)
plt.legend(ncols = 2)
plt.show()

#%%
'''

pred = bdt.predict(X_test)

f1_true = metrics.f1_score(Y_test, pred, average=None)
print("F1 score of adaBoost Classifier: ", f1_true)
print("Accuracy Score of Model: ", metrics.accuracy_score(Y_test, pred))

print()

y_pred_proba = bdt.predict_proba(X_test)

for i in range(0,10):
    y_tst = np.where(Y_test[0] == i+1, 1, 0)
    auc_true = metrics.roc_auc_score(y_tst, y_pred_proba[:, i])
    fpr, tpr, _ = metrics.roc_curve(y_tst,  y_pred_proba[:, i])
    fig, ax = plt.subplots()
    plt.plot(fpr,tpr, label="AUC="+str(auc_true))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC Curve for AdaBoost Classifier: " +str( i+1))
    ax.axline([0, 0], [1, 1], color = "black", linestyle = "dotted")
    ax.axline([0, 0], [0, 1], color = "black", linestyle = "dotted")
    ax.axline([0, 1], [1, 1], color = "black", linestyle = "dotted")
    plt.xlim(-0.01, 1.01) 
    plt.ylim(-0.01, 1.01)
    plt.legend()
    plt.show()
    


auc_true = metrics.roc_auc_score(Y_test, y_pred_proba)

fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)

#create ROC curve
fig, ax = plt.subplots()
plt.plot(fpr,tpr, label="AUC="+str(auc_true))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ROC Curve for AdaBoost Classifier")
ax.axline([0, 0], [1, 1], color = "black", linestyle = "dotted")
ax.axline([0, 0], [0, 1], color = "black", linestyle = "dotted")
ax.axline([0, 1], [1, 1], color = "black", linestyle = "dotted")
plt.xlim(-0.01, 1.01) 
plt.ylim(-0.01, 1.01)
plt.legend()
plt.show()
'''


#%% Trying t-SNE
'''
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2, perplexity=25, n_jobs=-1, init="pca", method = "barnes_hut").fit_transform(X_train)
print(X_embedded.shape)

rotatedData2_train = pd.DataFrame(X_embedded)
rotatedData2_train["Genre"] = X_train_gen


plt.figure()
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(predcols))]
for i in range(len(predcols)):
    filtered_label = rotatedData2_train.loc[rotatedData2_train.Genre == predcols[i]]
    plt.scatter(filtered_label[0], filtered_label[1], color = colors[i])

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
for i in range(len(predcols)):
    filtered_label = rotatedData2_train.loc[rotatedData2_train.Genre == predcols[i]]
    ax.scatter3D(filtered_label[0] , filtered_label[1], filtered_label[2], color = colors[i])

ax.set_xlabel('X-axis', fontweight ='bold')
ax.set_ylabel('Y-axis', fontweight ='bold')
ax.set_zlabel('Z-axis', fontweight ='bold')
plt.show()
'''
 #%% dBScan on LDA
'''
 data_2d = pd.DataFrame(rotatedData_train[:,:5])

 for e in np.linspace(0.05,3,10):
     for m in range(4,15, 2):
         db = cluster.DBSCAN(eps = e, min_samples = m).fit(data_2d)
         labels = db.labels_
         data_2d["label"] = labels
         unique_labels = set(labels)
         n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
         colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))] 
         fig = plt.figure(figsize = (10, 7))
         ax = plt.axes(projection ="3d")
         for i in unique_labels:
             filtered_label = data_2d.loc[data_2d.label == i]
             ax.scatter3D(filtered_label[0] , filtered_label[1], filtered_label[2], color = colors[i])
         ax.set_xlabel('Component 1', fontweight ='bold')
         ax.set_ylabel('Component 2', fontweight ='bold')
         ax.set_zlabel('Component 3', fontweight ='bold')
         plt.title(f"Estimated number of clusters when eps = {e} and m = {m}: {n_clusters_}" )
         plt.show()
         data_2d = data_2d.iloc[:,:5]

 '''
 
#%%
'''

 #Clustering Data Using 
 import scipy.cluster.hierarchy as sch


 plt.figure(figsize=(15,6))
 plt.title('Dendrogram')
 plt.xlabel('Customers')
 plt.ylabel('Euclidean distances')
 plt.hlines(y=250,xmin=0,xmax=200000,lw=3,linestyles='--')
 #plt.text(x=900,y=220,s='Horizontal line crossing 5 vertical lines',fontsize=20)
 #plt.grid(True)
 dendrogram = sch.dendrogram(sch.linkage(rotatedData_train[:,:3], method = 'ward'))
 plt.show()

 from sklearn.cluster import AgglomerativeClustering
 hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
 y_hc = hc.fit_predict(rotatedData_train[:,:3])

 data_2d = pd.DataFrame(rotatedData_train[:,:3])
 data_2d["label"] = y_hc


 fig = plt.figure(figsize = (10, 7))
 ax = plt.axes(projection ="3d")
 colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(predcols))]
 for i in [0,1]:
  
     filtered_label = data_2d.loc[data_2d.label == i]
     ax.scatter3D(filtered_label[0] , filtered_label[1], filtered_label[2], color = colors[i])
 plt.title("Hierarchal on PCA Data ")
 ax.set_xlabel('X-axis', fontweight ='bold')
 ax.set_ylabel('Y-axis', fontweight ='bold')
 ax.set_zlabel('Z-axis', fontweight ='bold')
 plt.show()
 plt.show()
 '''
 
#%%
'''
 umap_model = UMAP(n_components=3, n_neighbors=50,min_dist=0.001,random_state=19123302)
 umap_model.fit(X_train)
 X_umap = umap_model.transform(X_train)
 X_umap2 = umap_model.transform(X_test)

 rotatedData2_train = pd.DataFrame(X_umap)
 rotatedData2_test = pd.DataFrame(X_umap2)

 rotatedData2_train["Genre"] = X_train_gen


 plt.figure()
 colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(predcols))]
 for i in range(len(predcols)):
     filtered_label = rotatedData2_train.loc[rotatedData2_train.Genre == predcols[i]]
     plt.scatter(filtered_label[0], filtered_label[1], color = colors[i])

 plt.xlabel('Component 1')
 plt.ylabel('Component 2')
 plt.show()

 fig = plt.figure(figsize = (10, 7))
 ax = plt.axes(projection ="3d")
 colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(predcols))] 
 # Creating plot
 for i in range(len(predcols)):
     filtered_label = rotatedData2_train.loc[rotatedData2_train.Genre == predcols[i]]
     ax.scatter3D(filtered_label[0] , filtered_label[1], filtered_label[2], color = colors[i], label=predcols[i])
 plt.legend()
 plt.title("Resulting 3 Components from UMAP")
 ax.set_xlabel('Component 1', fontweight ='bold')
 ax.set_ylabel('Component 2', fontweight ='bold')
 ax.set_zlabel('Component 3', fontweight ='bold')
 plt.show()


 #%%#%%

 #Clustering Data Using dBScan


 data_2d = pd.DataFrame(X_umap)

 for e in np.linspace(0.5,3,4):
     plt.figure()
     db = cluster.DBSCAN(eps = e, min_samples = 3).fit(data_2d)
     labels = db.labels_
     data_2d["label"] = labels
     unique_labels = set(labels)
     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
     
     fig = plt.figure(figsize = (10, 7))
     ax = plt.axes(projection ="3d")
     colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(predcols))]
     for i in unique_labels:
         filtered_label = data_2d.loc[data_2d.label == i]
         ax.scatter3D(filtered_label[0] , filtered_label[1], filtered_label[2], color = colors[i])
     plt.title(f"Estimated number of clusters from dBScan when eps = {e} and min_samples = {3}: {n_clusters_}" )
     plt.show()
     data_2d = data_2d.iloc[:,:3] 
     
     ax.set_xlabel('X-axis', fontweight ='bold')
     ax.set_ylabel('Y-axis', fontweight ='bold')
     ax.set_zlabel('Z-axis', fontweight ='bold')
     plt.show()

 '''