from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt

'''
Reading the data set which has both target data and the features, and then removing the synthetic
data features
''' 
print("Reading the data set which has both target data and the features, and then removing the synthetic data features")
data = pd.read_csv("combined.csv")
data = data.drop(["X_6","X_7","X_8","X_9","X_10"],axis=1)

'''
First clustering the data and predicting the class labels for each of the instance
'''
print("First clustering the data and predicting the class labels for each of the instance")
M = 10
kmeans = KMeans(n_clusters=M, random_state=421)
kmeans_data = kmeans.fit_predict(data.iloc[:,1:])
data = data.join(pd.DataFrame(kmeans_data,columns=["kmean_cluster_number"]))

'''
2D stratified sampling on the target value and the cluster number so that the algorithm which we will 
implement will have fair chances of learning all types of data.
'''
print("2D stratified sampling on the target value and the cluster number so that the algorithm which we will implement will have fair chances of learning all types of data.")
train,test_val = train_test_split(data,test_size = 0.2,stratify=data[["Target","kmean_cluster_number"]],random_state=42)
val,test = train_test_split(test_val,test_size = 0.5,stratify=test_val[["Target","kmean_cluster_number"]],random_state=24)

'''
Cluster number is not required now
'''
print("Removing Cluster number column since is not required now")
train = train.drop(["kmean_cluster_number"],axis=1)
test = test.drop(["kmean_cluster_number"],axis=1)
val = val.drop(["kmean_cluster_number"],axis=1)

mu = kmeans.cluster_centers_

'''
Splitting the labels form the data to train the data
'''
train_lab = train.iloc[:,0]
trainData = train.iloc[:,1:]
test_lab = test.iloc[:,0]
testData = test.iloc[:,1:]
val_lab = val.iloc[:,0]
valData = val.iloc[:,1:]

num_basis = len(mu)

def covar(trainData,num_basis):
    ''' 
    Getting the covar over the training data based on number of basics we have implemented
    Changed the spread for Gaussian radial basis function
    '''
    print("Using Uniform Gaussian radial basis function")
    train_transpose = np.transpose(trainData)
    iden = np.identity(np.shape(train_transpose)[0])
    holdResult = []
    for i in range(0,np.shape(train_transpose)[0]):
        holdRow = []
        for j in range(0,len(trainData)):
            holdRow.append(train_transpose.iloc[i,j])
        #iden[i] = np.dot(iden[i],np.dot(np.dot(0.1,i),np.var(holdRow)))
       	iden[i] = np.dot(iden[i],np.dot(np.dot(200,i),np.var(holdRow)))
    return iden
print(" Getting the covar over the training data based on number of basics we have implemented")
covarMat = covar(trainData,num_basis)

def genPhi(train,covarMat,num_basis):
    '''
    Getting the Phi based on the covariance and number of basis
    '''
    phiMat = np.zeros((len(train),int(num_basis))) 
    covarMatInv = np.linalg.pinv(covarMat)
    for i in range(0,num_basis):
        for j in range(0,len(train)):
            subsResult = (np.subtract(train.iloc[j,],mu[i,]))
            L = np.dot(np.transpose(subsResult),covarMatInv)
            R = np.dot(L,subsResult)
            phiMat[j][i] = math.exp(-np.dot(0.5,R))
    return phiMat

print("Computing Phi will take lot of time. Please wait...")

# Gen Phi Data
phiMat = genPhi(trainData,covarMat,num_basis)
test_phi = genPhi(testData,covarMat,num_basis)
val_phi = genPhi(valData,covarMat,num_basis)

print("Calculated Test Phi, Val Phi, and Training Phi Matrix")

lam =0.01

print("Getting the weights based on the Lambda and phi matrix")
def getWeights(train_lab,phiMat,lam):
    '''
    Getting the weights based on the Lambda and phi matrix
    '''
    iden = np.identity(len(phiMat[0]))
    ft = np.dot(iden,lam)
    st = np.dot(np.transpose(phiMat),phiMat)
    fmt = np.add(ft,st)
    fmt = np.linalg.pinv(fmt)
    smt = np.dot(np.transpose(phiMat),np.asarray(train_lab))
    wReg = np.dot(fmt,smt)
    return wReg

weights = getWeights(train_lab,phiMat,lam)

# Using same functions as the TA code had to check accuracy
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

TR_TEST_OUT  = GetValTest(phiMat,weights)
VAL_TEST_OUT = GetValTest(val_phi,weights)
TEST_OUT     = GetValTest(test_phi,weights)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,np.asarray(train_lab)))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,np.asarray(val_lab)))
TestAccuracy       = str(GetErms(TEST_OUT,np.asarray(test_lab)))

print ('UBITname      = pkubal')
print ('Person Number = 50290804')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------Basic Implementation----------------')
print ("M = 10 \nLambda = 0.01")
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))

np.random.seed(589)

alpha = 0.001
lam = 0.03
'''
Random initilization of weights
'''

def updateWeights(weights,phiMat,train_lab,alpha,lam):  
    midT = np.dot(np.transpose(prev_weight),phiMat)
    deltaL = -(np.subtract(train_lab,midT))
    deltaD = np.dot(deltaL,phiMat)
    deltaE = deltaD + np.dot(lam,prev_weight)

    delta = np.dot(-alpha,deltaE)
    new_weight = prev_weight + np.dot(delta,prev_weight)
    return new_weight


train_lab = np.asarray(train_lab)
log_erms_val = []
log_erms_train = []
log_erms_test = []
prev_weight = np.random.rand(np.shape(weights)[0])
print("Output images are stored in un_sgd_10_high_var_outputs folder (INCREMENTAL OUTPUTS)")
for i in range(0,400):
    print("Iteration: "+str(i))
    prev_weight = updateWeights(prev_weight,phiMat[i],train_lab[i],alpha,lam)
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(phiMat,prev_weight) 
    Erms_TR       = GetErms(TR_TEST_OUT,np.asarray(train_lab))
    log_erms_train.append(float(Erms_TR.split(',')[1]))
    print ('---------TrainingData Accuracy: ' + Erms_TR + '--------------')

    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(val_phi,prev_weight) 
    Erms_Val      = GetErms(VAL_TEST_OUT,np.asarray(val_lab))
    log_erms_val.append(float(Erms_Val.split(',')[1]))
    print ('---------ValidationData Accuracy: ' + Erms_Val + '--------------')

    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(test_phi,prev_weight) 
    Erms_Test = GetErms(TEST_OUT,np.asarray(test_lab))
    log_erms_test.append(float(Erms_Test.split(',')[1]))

    df = pd.DataFrame(log_erms_train)
    ax = df.plot(figsize=(10,15))
    ax.ticklabel_format(useOffset=False)

    plt.savefig('./un_sgd_10_high_var_outputs/log_erms_train.png',bbox_inches='tight')

    df = pd.DataFrame(log_erms_val)
    ax = df.plot(figsize=(10,15))
    ax.ticklabel_format(useOffset=False)

    plt.savefig('./un_sgd_10_high_var_outputs/log_erms_val.png',bbox_inches='tight')
    plt.close("all")

print ('----------Gradient Descent Solution--------------------')
print ("M = 10 \nLambda  = 0.03\neta=0.001")
print ("E_rms Training   = " + str(np.around(min(log_erms_train),5)))
print ("E_rms Validation = " + str(np.around(min(log_erms_val),5)))
print ("E_rms Testing    = " + str(np.around(min(log_erms_test),5)))
