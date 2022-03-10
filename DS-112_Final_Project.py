# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:33:09 2021

@author: janya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simple_linear_regress_func import simple_linear_regress_func 
from sklearn import linear_model
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

data0 = pd.read_csv('middleSchoolData.csv')
data = data0.to_numpy()

#%%1) What is the correlation between the number of applications and admissions to HSPHS?
#First, I will create a seperate array of just applications and admissions
appsAndAds = data[:,2:4]
#convert to array of float64
appsAndAds2 = appsAndAds.astype(np.float64)
#now I will find the correlation coefficient
corr = np.corrcoef(appsAndAds2[:,0], appsAndAds2[:,1])
rSquared = corr[0,1] ** 2
#and do linear regression
output = simple_linear_regress_func(appsAndAds2)
yHat = output[0]*appsAndAds2[:,0] + output[1]
#correlation coefficient of r = 0.80172654
#rsquared = 0.6427654402453512
plt.plot()
plt.plot(appsAndAds2[:,0],appsAndAds2[:,1], 'o',markersize=.75)
plt.plot(appsAndAds[:,0], yHat, color = 'orange', linewidth = 0.5)
plt.xlabel('Applications')
plt.ylabel('Admissions')





#%%2) What is a better predictor of admission to HSPHS? Raw number of applications or
#application *rate*?


#find the application rates for each school
schoolSizes = data[:,20]
schoolSizes1 = schoolSizes.astype(np.float64)
schoolSizes2 = schoolSizes1[np.isfinite(schoolSizes1)]
appls = appsAndAds2[:,0]


applRates = np.empty([len(data),1])
applRates[:] = np.NAN
for i in range(len(data)):
    applRates[i,0] = appls[i,]/schoolSizes1[i,]
    
applRates1 = applRates[np.isfinite(applRates)]

#so there are 2 NaN values in the arrays
    
#create array
rateVadmit = np.empty([len(data),2])
rateVadmit[:] = np.NaN
for i in range(len(data)):
    rateVadmit[i,0] = applRates[i,0]
    rateVadmit[i,1] = appsAndAds2[i,1]
    
#clean up that array
noGoodData = []

for i in range(len(data)):
    if np.isnan(applRates[i,0]):
        noGoodData += [i]
        
newrateVadmit = np.delete(rateVadmit, noGoodData, axis=0 )
        
        


#now do basically the same thing as above (1) but with rates
corr2 = np.corrcoef(newrateVadmit[:,0], newrateVadmit[:,1])
rSquared2 = corr2[0,1] ** 2
#and do linear regression
output2 = simple_linear_regress_func(newrateVadmit)
yHat2 = output2[0]*newrateVadmit[:,0] + output2[1]
#correlation coefficient of r = .65875075
#rsquared = 0.4339525544466692

plt.figure()
plt.plot()
plt.plot(newrateVadmit[:,0],newrateVadmit[:,1], 'o',markersize=.75)
plt.plot(newrateVadmit[:,0], yHat2, color = 'orange', linewidth = 0.5)
plt.xlabel('Application Rate')
plt.ylabel('Admissions')

#%%3) Which school has the best *per student* odds of sending someone to HSPHS?
    
#calculate the odds for each school
#(accepted/applied)/(rejected/applied) = (accepted/applied)/(1-(accepted/applied))

oddsPerSchool = np.empty([len(data), 3]) 
oddsPerSchool[:] = np.NaN
for i in range(len(data)):
    if appsAndAds[i,0] != 0:
        oddsPerSchool[i,0] = appsAndAds[i,1]/appsAndAds[i,0]
        oddsPerSchool[i,1] = 1-(appsAndAds[i,1]/appsAndAds[i,0])
        oddsPerSchool[i,2] = oddsPerSchool[i,0]/oddsPerSchool[i,1]

    
# Find the maximum of the third column
maxLocation = 0
maxOdds = 0
for i in range(len(data)):
    if oddsPerSchool[i,2] > maxOdds:
        maxOdds = oddsPerSchool[i,2]
        maxLocation = i
        
#maxlocation = 304
#maxOdds = 4.4565

#%%4) Is there a relationship between how students perceive their school (as reported in columns
#L-Q) and how the school performs on objective measures of achievement (as noted in
#columns V-X).

#First, I will pull out the two groups of columns
schoolPerception = data[:,11:17]
schoolPerception = schoolPerception.astype(np.float64)
noGoodData1 = []
for i in range(len(schoolPerception)):
    if np.isnan(schoolPerception[i,0]) or np.isnan(schoolPerception[i,1]) or np.isnan(schoolPerception[i,2])or np.isnan(schoolPerception[i,3]) or np.isnan(schoolPerception[i,4]) or np.isnan(schoolPerception[i,5]):
        noGoodData1 += [i]
schoolAchievement = data[:,21:24]
schoolAchievement = schoolAchievement.astype(np.float64)
totalData = np.empty([len(data), 9])
#also merge the two groups
for i in range(len(data)):
    totalData[i,0] = schoolPerception[i,0]
    totalData[i,1] = schoolPerception[i,1]
    totalData[i,2] = schoolPerception[i,2]
    totalData[i,3] = schoolPerception[i,3]
    totalData[i,4] = schoolPerception[i,4]
    totalData[i,5] = schoolPerception[i,5]
    totalData[i,6] = schoolAchievement[i,0]
    totalData[i,7] = schoolAchievement[i,1]
    totalData[i,8] = schoolAchievement[i,2]
    
    
#I do a row reduction based on no good data 1 because those school just didn't take that survey at all

totalData = np.delete(totalData, noGoodData1, axis=0 )
schoolAchievement1 = np.delete(schoolAchievement, noGoodData1, axis=0)
schoolAchievement1 = np.delete(schoolAchievement, noGoodData1, axis=0)

noGoodData2 = []
for i in range(len(schoolAchievement1)):
    if np.isnan(schoolAchievement1[i,0]) or np.isnan(schoolAchievement1[i,1]) or np.isnan(schoolAchievement1[i,2]):
        noGoodData2 += [i]
#after taking into consideration noGoodData1, i look at the problems with school Achievement (noGoodData2)
#either all of the 3 values are NaN or just the first value
#First, definitely doing a row reduction for the ones that are missing all 3
noGoodData2 = []
for i in range(len(schoolAchievement1)):
    if np.isnan(schoolAchievement1[i,0]) or np.isnan(schoolAchievement1[i,1]) or np.isnan(schoolAchievement1[i,2]):
        noGoodData2 += [i]
        
totalData = np.delete(totalData, noGoodData2, axis=0 )


        
schoolAchievement1 = np.delete(schoolAchievement1, noGoodData2, axis = 0)

#total data is now completely clean
#PCA on school Perception
r0 = np.corrcoef(totalData[:,:6],rowvar=False) # True = variables are rowwise; False = variables are columnwise

# Plot the data:
plt.figure()
plt.imshow(r0) 
plt.colorbar()
  
zscoredData0 = stats.zscore(totalData[:,:6])

pca0 = PCA().fit(zscoredData0)

eigVals0 = pca0.explained_variance_

loadings0 = pca0.components_

rotatedData0 = pca0.fit_transform(zscoredData0)

covarExplained0 = eigVals0/sum(eigVals0)*100

numClasses = 6
plt.figure()
plt.bar(np.linspace(1,6,6),eigVals0)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,numClasses],[1,1],color='red',linewidth=1)

#very clearly, the first component i most important. it is the only one that meets the 1 threshold.


whichPrincipalComponent0 = 0
plt.figure()
plt.bar(np.linspace(1,6,6),loadings0[whichPrincipalComponent0,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')

#PCA on school Achievement 1
r = np.corrcoef(totalData[:,6:],rowvar=False) # True = variables are rowwise; False = variables are columnwise

# Plot the data:
plt.figure()
plt.imshow(r) 
plt.colorbar()
  
zscoredData = stats.zscore(totalData[:,6:])

pca = PCA().fit(zscoredData)

eigVals = pca.explained_variance_

loadings = pca.components_

rotatedData = pca.fit_transform(zscoredData)

covarExplained = eigVals/sum(eigVals)*100

numClasses = 3
plt.figure()
plt.bar(np.linspace(1,3,3),eigVals)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,numClasses],[1,1],color='red',linewidth=1)

whichPrincipalComponent = 0
plt.figure()
plt.bar(np.linspace(1,3,3),loadings[whichPrincipalComponent,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')

#so basically there is one valuable principle component with eigenvalue over one
#and those values are in the first column of rotated data

corr3 = np.corrcoef(rotatedData0[:,0], rotatedData[:,0])
perceptionVlackAchievement = np.transpose(np.array([rotatedData0[:,0], rotatedData[:,0]]))
output3 = simple_linear_regress_func(perceptionVlackAchievement)

yHat3 = output3[0]*perceptionVlackAchievement[:,0] + output3[1]
plt.figure()
plt.plot()
plt.plot(perceptionVlackAchievement[:,0],perceptionVlackAchievement[:,1], 'o',markersize=.75)
plt.plot(perceptionVlackAchievement[:,0], yHat3, color = 'orange', linewidth = 0.5)
plt.xlabel('General School Perception')
plt.ylabel('School Achievement')

#%%5)Test a hypothesis of your choice as to which kind of school (e.g. small schools vs. large
#schools or charter schools vs. not (or any other classification, such as rich vs. poor school))
#performs differently than another kind either on some dependent measure, e.g. objective
#measures of achievement or admission to HSPHS (pick one).

#Null Hypothesis: Small schools and large schools perform the same with respect to admission to HSPHS
#Alternate Hypothesis: large schools have a larger admission rate (accepted/applied) than 
#small schools

#defined large school as school with greater than or equal to the median number students

#school sizes 1 is our array with sizes
y = np.median(schoolSizes2[:,])

tempData = np.empty([len(data), 3])
tempData[:] = np.NaN
for ii in range(len(data)):
    tempData[ii,0] = schoolSizes1[ii,]
    if schoolSizes[ii,] >= y:
        tempData[ii,2] = 1
    else: 
        tempData[ii,2] = 0
    if appsAndAds2[ii,0] != 0:
        tempData[ii,1] = appsAndAds2[ii,1]/appsAndAds2[ii,0]
        

#clean up the data
    
noGoodData3 = []
for i in range(len(tempData)):
    if np.isnan(tempData[i,0]) or np.isnan(tempData[i,1]) or np.isnan(tempData[i,2]):
        noGoodData3 += [i]
        
cleanData = np.delete(tempData, noGoodData3, axis=0)


#now we can do the test

small = []
large = []
for i in range(len(cleanData)):
    if cleanData[i,2] == 0:
        small += [cleanData[i,1]]
    else:
        large += [cleanData[i,1]]
        
small = np.array(small)
large = np.array(large)
        
descriptivesContainer = np.empty([2,4])
descriptivesContainer[:] = np.NaN

descriptivesContainer[0,0] = np.mean(small[:,]) # mu
descriptivesContainer[0,1] = np.std(small[:,]) # sigma
descriptivesContainer[0,2] = len(small[:,]) # n
descriptivesContainer[0,3] = descriptivesContainer[0,1]/np.sqrt(descriptivesContainer[0,2]) # se

descriptivesContainer[1,0] = np.mean(large[:,]) # mu
descriptivesContainer[1,1] = np.std(large[:,]) # sigma
descriptivesContainer[1,2] = len(large[:,]) # n
descriptivesContainer[1,3] = descriptivesContainer[1,1]/np.sqrt(descriptivesContainer[1,2]) # se


t1,p1 = stats.ttest_ind(small,large)

#%%5)Test a hypothesis of your choice as to which kind of school (e.g. small schools vs. large
#schools or charter schools vs. not (or any other classification, such as rich vs. poor school))
#performs differently than another kind either on some dependent measure, e.g. objective
#measures of achievement or admission to HSPHS (pick one).

#Null Hypothesis: Small schools and large schools perform the same with respect to admission to HSPHS
#Alternate Hypothesis: large schools have a larger admission rate (accepted/applied) than 
#small schools

#defined large school as school with greater than or equal to the median number students

#school sizes 1 is our array with sizes
y = np.median(schoolSizes2[:,])

tempData = np.empty([len(data), 3])
tempData[:] = np.NaN
for ii in range(len(data)):
    tempData[ii,0] = schoolSizes1[ii,]
    if schoolSizes[ii,] >= y:
        tempData[ii,2] = 1
    else: 
        tempData[ii,2] = 0
    if appsAndAds2[ii,0] != 0:
        tempData[ii,1] = appsAndAds2[ii,1]/appsAndAds2[ii,0]
    else:
        tempData[ii,1] = 0
        

#clean up the data
    
noGoodData3 = []
for i in range(len(tempData)):
    if np.isnan(tempData[i,0]) or np.isnan(tempData[i,1]) or np.isnan(tempData[i,2]):
        noGoodData3 += [i]
        
cleanData = np.delete(tempData, noGoodData3, axis=0)


#now we can do the test

small = []
large = []
for i in range(len(cleanData)):
    if cleanData[i,2] == 0:
        small += [cleanData[i,1]]
    else:
        large += [cleanData[i,1]]
        
small = np.array(small)
large = np.array(large)
        
descriptivesContainer = np.empty([2,4])
descriptivesContainer[:] = np.NaN

descriptivesContainer[0,0] = np.mean(small[:,]) # mu
descriptivesContainer[0,1] = np.std(small[:,]) # sigma
descriptivesContainer[0,2] = len(small[:,]) # n
descriptivesContainer[0,3] = descriptivesContainer[0,1]/np.sqrt(descriptivesContainer[0,2]) # se

descriptivesContainer[1,0] = np.mean(large[:,]) # mu
descriptivesContainer[1,1] = np.std(large[:,]) # sigma
descriptivesContainer[1,2] = len(large[:,]) # n
descriptivesContainer[1,3] = descriptivesContainer[1,1]/np.sqrt(descriptivesContainer[1,2]) # se


t1,p1 = stats.ttest_ind(small,large)


#%%6) Is there any evidence that the availability of material resources (e.g. per student spending
#or class size) impacts objective measures of achievement or admission to HSPHS?

#Null Hypothesis: Public schools with higher class sizes have the same admission to HSPHS as schools with lower class sizes.

#first, pull out the data we need
data6 = np.transpose(np.array([data[:,3], data[:,5]]))

data6 = data6.astype(np.float64)
#clean up data

noBuenoData = []
for i in range(len(data6)):
    if np.isnan(data6[i,0]) or np.isnan(data6[i,1]):
        noBuenoData += [i]
        
data62 = np.delete(data6, noBuenoData, axis=0)

x = np.median(data62[:,1])

low = []
high = []
for i in range(len(data62)):
    if data62[i,1] >= x:
        high += [data62[i,0]]
    else:
        low += [data62[i,0]]
        
low = np.array(low)
high = np.array(high)


descriptivesContainer2 = np.empty([2,4])
descriptivesContainer2[:] = np.NaN

descriptivesContainer2[0,0] = np.mean(low[:,]) # mu
descriptivesContainer2[0,1] = np.std(low[:,]) # sigma
descriptivesContainer2[0,2] = len(low[:,]) # n
descriptivesContainer2[0,3] = descriptivesContainer2[0,1]/np.sqrt(descriptivesContainer2[0,2]) # se

descriptivesContainer2[1,0] = np.mean(high[:,]) # mu
descriptivesContainer2[1,1] = np.std(high[:,]) # sigma
descriptivesContainer2[1,2] = len(high[:,]) # n
descriptivesContainer2[1,3] = descriptivesContainer2[1,1]/np.sqrt(descriptivesContainer2[1,2]) # se

t2,p2 = stats.ttest_ind(low,high)

#%%7) What proportion of schools accounts for 90% of all students accepted to HSPHS? 
#pull out school dbn and acceptances

data7 = np.transpose(np.array([data[:,0], data[:, 3]]))
data7[:,1] = data7[:,1].astype(np.float64)
#ordering the data from most accepted to least accepted

data72 = data7[data7[:,1].argsort()]

#flip it
data73 = []
for i in range(len(data72)):
    data73 += [data72[(593-i),:]]
    
data73 = np.array(data73)

z = np.sum(data73[:,1])

#4014.9 
sum = 0
counter = 0
while sum < (.90*z):
    sum += data73[counter,1]
    counter += 1
    
t = np.sum(data73[:123,1])

#so the first 123 schools (up to index 122) accounts for 90% of the students

#now, i will build a bar graph 

plt.figure()
plt.bar(np.linspace(0,149,150), data73[:150,1], width = 1)
plt.plot([123,123],[0, 200],color='red',linewidth=1)
plt.xlabel('Index in Ordered List')
plt.ylabel('Accepted')

'''
#%%8) Build a model of your choice – clustering, classification or prediction – that includes all
#factors – as to what school characteristics are most important in terms of a) sending
#students to HSPHS, b) achieving high scores on objective measures of achievement?

#I want to build a clustering

yOutcomes = data0.acceptances.to_numpy()
predictors = data0[['per_pupil_spending','avg_class_size','asian_percent', 'black_percent', 'hispanic_percent', 'multiple_percent', 'white_percent', 'rigorous_instruction', 'collaborative_teachers', 'supportive_environment', 'effective_school_leadership', 'strong_family_community_ties', 'trust', 'disability_percent','poverty_percent', 'ESL_percent', 'school_size', 'student_achievement', 'reading_scores_exceed', 'math_scores_exceed']].to_numpy()

noBuenoData2 = []

for i in range(len(predictors)): 
    for c in range(20):
        if np.isnan(predictors[i,c]):
            noBuenoData2 += [i]
            break
        
predictors = np.delete(predictors, noBuenoData2, axis = 0)
        
yOutcomes = np.delete(yOutcomes, noBuenoData2, axis = 0)

plt.figure()
r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r) 
plt.colorbar()

zscoredData = stats.zscore(predictors)
pca = PCA().fit(zscoredData)
eigValues = pca.explained_variance_ 
loadings = pca.components_ *-1
origDataNewCoordinates = pca.fit_transform(zscoredData) *-1

# Scree plot:
plt.figure()
numPredictors = 20
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.plot([0,numPredictors],[1,1],color='red',linewidth=1)

plt.figure()

plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:]) # minority

plt.figure()

plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:]) # community

plt.figure()
plt.plot(origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],'o',markersize=1)
plt.xlabel('Minority')
plt.ylabel('Community and Support')

X = np.transpose(np.array([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]]))

numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
Q = np.empty([numClusters,1]) # init container to store sums
Q[:] = np.NaN # convert to NaN

plt.figure()
# Compute kMeans:
for ii in range(2, 3): # Loop through each cluster (from 2 to 10!)
    kMeans = KMeans(n_clusters = int(ii)).fit(X) # compute kmeans
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(X,cId) # compute the mean silhouette coefficient of all samples
    Q[ii-2] = sum(s) # take sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,500)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(Q[ii-2]))) # sum rounded to nearest integer
    
plt.figure()    
indexVector = np.linspace(1,len(np.unique(cId)),len(np.unique(cId))) 
for ii in indexVector:
    plotIndex = np.argwhere(cId == int(ii-1))
    plt.plot(origDataNewCoordinates[plotIndex,0],origDataNewCoordinates[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Minority')
    plt.ylabel('Community and Support')    
    

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],yOutcomes)

#%%8) Build a model of your choice – clustering, classification or prediction – that includes all
#factors – as to what school characteristics are most important in terms of a) sending
#students to HSPHS, b) achieving high scores on objective measures of achievement?

#I want to build a clustering

yOutcomes = data0.acceptances.to_numpy()
predictors = data0[['asian_percent', 'black_percent', 'hispanic_percent', 'multiple_percent', 'white_percent', 'rigorous_instruction', 'collaborative_teachers', 'supportive_environment', 'effective_school_leadership', 'strong_family_community_ties', 'trust', 'disability_percent','poverty_percent', 'ESL_percent', 'school_size', 'student_achievement', 'reading_scores_exceed', 'math_scores_exceed']].to_numpy()

noBuenoData2 = []

for i in range(len(predictors)): 
    for c in range(18):
        if np.isnan(predictors[i,c]):
            noBuenoData2 += [i]
            break
        
predictors = np.delete(predictors, noBuenoData2, axis = 0)
        
yOutcomes = np.delete(yOutcomes, noBuenoData2, axis = 0)

plt.figure()
r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r) 
plt.colorbar()

zscoredData = stats.zscore(predictors)
pca = PCA().fit(zscoredData)
eigValues = pca.explained_variance_ 
loadings = pca.components_ *-1
origDataNewCoordinates = pca.fit_transform(zscoredData) *-1

# Scree plot:
plt.figure()
numPredictors = 18
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.plot([0,numPredictors],[1,1],color='red',linewidth=1)

plt.figure()

plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:]) # minority

plt.figure()

plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:]) # community

plt.figure()
plt.plot(origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],'o',markersize=1)
plt.xlabel('Minority')
plt.ylabel('Community and Support')

X = np.transpose(np.array([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]]))

numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
Q = np.empty([numClusters,1]) # init container to store sums
Q[:] = np.NaN # convert to NaN

plt.figure()
# Compute kMeans:
for ii in range(2,3): # Loop through each cluster (from 2 to 10!)
    kMeans = KMeans(n_clusters = int(ii)).fit(X) # compute kmeans
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(X,cId) # compute the mean silhouette coefficient of all samples
    Q[ii-2] = sum(s) # take sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,500)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(Q[ii-2]))) # sum rounded to nearest integer
    
plt.figure()    
indexVector = np.linspace(1,len(np.unique(cId)),len(np.unique(cId))) 
for ii in indexVector:
    plotIndex = np.argwhere(cId == int(ii-1))
    plt.plot(origDataNewCoordinates[plotIndex,0],origDataNewCoordinates[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Minority')
    plt.ylabel('Community and Support')    
    

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],yOutcomes)


plt.scatter(X[:,0], X[:,1], s = None, c = yOutcomes)



#%%8) Build a model of your choice – clustering, classification or prediction – that includes all
#factors – as to what school characteristics are most important in terms of a) sending
#students to HSPHS, b) achieving high scores on objective measures of achievement?

#I want to build a clustering

yOutcomes = data0.acceptances.to_numpy()
predictors = data0[['asian_percent', 'black_percent', 'hispanic_percent', 'multiple_percent', 'white_percent', 'rigorous_instruction', 'collaborative_teachers', 'supportive_environment', 'effective_school_leadership', 'strong_family_community_ties', 'trust', 'disability_percent','poverty_percent', 'ESL_percent', 'school_size']].to_numpy()

noBuenoData2 = []

for i in range(len(predictors)): 
    for c in range(15):
        if np.isnan(predictors[i,c]):
            noBuenoData2 += [i]
            break
        
predictors = np.delete(predictors, noBuenoData2, axis = 0)
        
yOutcomes = np.delete(yOutcomes, noBuenoData2, axis = 0)

plt.figure()
r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r) 
plt.colorbar()

zscoredData = stats.zscore(predictors)
pca = PCA().fit(zscoredData)
eigValues = pca.explained_variance_ 
loadings = pca.components_ *-1
origDataNewCoordinates = pca.fit_transform(zscoredData) *-1

# Scree plot:
plt.figure()
numPredictors = 15
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.plot([0,numPredictors],[1,1],color='red',linewidth=1)

plt.figure()

plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:]) # minority

plt.figure()

plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:]) # community

plt.figure()
plt.plot(origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],'o',markersize=1)
plt.xlabel('Minority')
plt.ylabel('Community and Support')

X = np.transpose(np.array([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]]))

numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
Q = np.empty([numClusters,1]) # init container to store sums
Q[:] = np.NaN # convert to NaN

plt.figure()
# Compute kMeans:
for ii in range(3,4): # Loop through each cluster (from 2 to 10!)
    kMeans = KMeans(n_clusters = int(ii)).fit(X) # compute kmeans
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(X,cId) # compute the mean silhouette coefficient of all samples
    Q[ii-2] = sum(s) # take sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,500)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(Q[ii-2]))) # sum rounded to nearest integer
    
plt.figure()    
indexVector = np.linspace(1,len(np.unique(cId)),len(np.unique(cId))) 
for ii in indexVector:
    plotIndex = np.argwhere(cId == int(ii-1))
    plt.plot(origDataNewCoordinates[plotIndex,0],origDataNewCoordinates[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Minority')
    plt.ylabel('Community and Support')    
    

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],yOutcomes)


plt.scatter(X[:,0], X[:,1], s = None, c = yOutcomes)






#%%8) Build a model of your choice – clustering, classification or prediction – that includes all
#factors – as to what school characteristics are most important in terms of a) sending
#students to HSPHS, b) achieving high scores on objective measures of achievement?

#I want to build a clustering

yOutcomes = data0.acceptances.to_numpy()
predictors = data0[['per_pupil_spending','avg_class_size','asian_percent', 'black_percent', 'hispanic_percent', 'multiple_percent', 'white_percent', 'rigorous_instruction', 'collaborative_teachers', 'supportive_environment', 'effective_school_leadership', 'strong_family_community_ties', 'trust', 'disability_percent','poverty_percent', 'ESL_percent', 'school_size']].to_numpy()

noBuenoData2 = []

for i in range(len(predictors)): 
    for c in range(17):
        if np.isnan(predictors[i,c]):
            noBuenoData2 += [i]
            break
        
predictors = np.delete(predictors, noBuenoData2, axis = 0)
        
yOutcomes = np.delete(yOutcomes, noBuenoData2, axis = 0)

plt.figure()
r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r) 
plt.colorbar()

zscoredData = stats.zscore(predictors)
pca = PCA().fit(zscoredData)
eigValues = pca.explained_variance_ 
loadings = pca.components_ *-1
origDataNewCoordinates = pca.fit_transform(zscoredData) *-1

# Scree plot:
plt.figure()
numPredictors = 17
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.plot([0,numPredictors],[1,1],color='red',linewidth=1)

plt.figure()

plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:]) # minority
plt.xlabel('Question')
plt.ylabel('Loading')
plt.figure()

plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:]) # community
plt.xlabel('Question')
plt.ylabel('Loading')



plt.figure()
plt.plot(origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],'o',markersize=1)
plt.xlabel('Minority')
plt.ylabel('Community and Support')

X = np.transpose(np.array([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]]))

numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
Q = np.empty([numClusters,1]) # init container to store sums
Q[:] = np.NaN # convert to NaN

plt.figure()
# Compute kMeans:
for ii in range(3, 4): # Loop through each cluster (from 2 to 10!)
    kMeans = KMeans(n_clusters = int(ii)).fit(X) # compute kmeans
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(X,cId) # compute the mean silhouette coefficient of all samples
    Q[ii-2] = sum(s) # take sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,500)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(Q[ii-2]))) # sum rounded to nearest integer
    
plt.figure()    
indexVector = np.linspace(1,len(np.unique(cId)),len(np.unique(cId))) 
for ii in indexVector:
    plotIndex = np.argwhere(cId == int(ii-1))
    plt.plot(origDataNewCoordinates[plotIndex,0],origDataNewCoordinates[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Minority')
    plt.ylabel('Community and Support')    
    

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],yOutcomes)

plt.figure()
plt.scatter(X[:,0], X[:,1], s = None, c = yOutcomes)
plt.xlabel('Minority')
plt.ylabel('Community and Support') 
plt.colorbar()


'''

#%%8) Build a model of your choice – clustering, classification or prediction – that includes all
#factors – as to what school characteristics are most important in terms of a) sending
#students to HSPHS, b) achieving high scores on objective measures of achievement?

#I want to build a clustering

#First, I will pull out the two groups of columns

schoolAchievement = data[:,21:24]
schoolAchievement = schoolAchievement.astype(np.float64)


noGoodData2 = []
for i in range(len(schoolAchievement)):
    if np.isnan(schoolAchievement[i,0]) or np.isnan(schoolAchievement[i,1]) or np.isnan(schoolAchievement[i,2]):
        noGoodData2 += [i]


predictors = data0[['per_pupil_spending','avg_class_size','asian_percent', 'black_percent', 'hispanic_percent', 'multiple_percent', 'white_percent', 'rigorous_instruction', 'collaborative_teachers', 'supportive_environment', 'effective_school_leadership', 'strong_family_community_ties', 'trust', 'disability_percent','poverty_percent', 'ESL_percent', 'school_size']].to_numpy()

noBuenoData2 = []

for i in range(len(predictors)): 
    for c in range(17):
        if np.isnan(predictors[i,c]):
            noBuenoData2 += [i]
            break
        
badData = noGoodData2 + noBuenoData2

badDataSimple = []
for x in badData:
    if x not in badDataSimple:
        badDataSimple += [x]

predictors = np.delete(predictors, badDataSimple, axis = 0)
schoolAchievement1 = np.delete(schoolAchievement, badDataSimple, axis = 0)

#so now, i can do a pca on the achievements

r = np.corrcoef(schoolAchievement1,rowvar=False) # True = variables are rowwise; False = variables are columnwise

# Plot the data:
plt.figure()
plt.imshow(r) 
plt.colorbar()
  
zscoredData = stats.zscore(schoolAchievement1)

pca = PCA().fit(zscoredData)

eigVals = pca.explained_variance_

loadings = pca.components_

rotatedData = pca.fit_transform(zscoredData)

covarExplained = eigVals/np.sum(eigVals)*100

numClasses = 3
plt.figure()
plt.bar(np.linspace(1,3,3),eigVals)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,numClasses],[1,1],color='red',linewidth=1)

whichPrincipalComponent = 0
plt.figure()
plt.bar(np.linspace(1,3,3),loadings[whichPrincipalComponent,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')




plt.figure()
r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r) 
plt.colorbar()

zscoredData = stats.zscore(predictors)
pca = PCA().fit(zscoredData)
eigValues = pca.explained_variance_ 
loadings = pca.components_ *-1
origDataNewCoordinates = pca.fit_transform(zscoredData) *-1

# Scree plot:
plt.figure()
numPredictors = 17
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.plot([0,numPredictors],[1,1],color='red',linewidth=1)

plt.figure()

plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:]) # minority
plt.xlabel('Question')
plt.ylabel('Loading')
plt.figure()

plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:]) # community
plt.xlabel('Question')
plt.ylabel('Loading')



plt.figure()
plt.plot(origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],'o',markersize=1)
plt.xlabel('Minority')
plt.ylabel('Community and Support')

X = np.transpose(np.array([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]]))

numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
Q = np.empty([numClusters,1]) # init container to store sums
Q[:] = np.NaN # convert to NaN

plt.figure()


# Compute kMeans:
for ii in range(2, 10): # Loop through each cluster (from 2 to 10!)
    kMeans = KMeans(n_clusters = int(ii)).fit(X) # compute kmeans
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(X,cId) # compute the mean silhouette coefficient of all samples
    Q[ii-2] = np.sum(s) # take sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,500)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(Q[ii-2]))) # sum rounded to nearest integer

plt.subplots_adjust(hspace=1.5)
    
plt.figure()    
indexVector = np.linspace(1,len(np.unique(cId)),len(np.unique(cId))) 
for ii in indexVector:
    plotIndex = np.argwhere(cId == int(ii-1))
    plt.plot(origDataNewCoordinates[plotIndex,0],origDataNewCoordinates[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Minority')
    plt.ylabel('Community and Support')    
    

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],rotatedData[:,0])

plt.figure()
plt.scatter(X[:,0], X[:,1], s = None, c = rotatedData[:,0])
plt.xlabel('Minority')
plt.ylabel('Community and Support') 
plt.colorbar()



X = np.transpose([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]])
Y =  rotatedData[:,0]
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
rSqr = regr.score(X,Y)
betas = regr.coef_ # m
yInt = regr.intercept_  # b

import matplotlib.pyplot as plt
# Visualize: actual vs. predicted income (from model)
y_hat = betas[0]*origDataNewCoordinates[:,0] + betas[1]*origDataNewCoordinates[:,1] + yInt
plt.plot(y_hat,rotatedData[:,0],'o',markersize=.75) # y_hat, income
plt.xlabel('Prediction from model') 
plt.ylabel('Actual')  
plt.title('R^2: {:.3f}'.format(rSqr)) 