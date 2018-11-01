from numpy.random import seed
seed(111)
    
from tensorflow import set_random_seed
set_random_seed(125)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
import keras
import numpy as np
import pandas as pd

def fizzbuzz(n):
    
    # Logic Explanation
    # If n / a ; where a is any number greater than 1 has the remainder as 0 which means it is divisible by n. Therefore,
    # in the first condition we check if it is divisible by both 3 and 5, If it isn't then we check if it is divisible
    # by only 3 and if not then try 5. If it is not divisible by either 3 or 5 then we return "Other". All the other return
    # statements work on the FizzBuzz Game.
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'

def createInputCSV(start,end,filename):
    
    # Why list in Python?
    # Amongst other data types in Python, List is an example of mutable object. We need a mutable object to
    # modify the data such as to Normalize the data set, also Machine Learning is based on repetative examples
    # if we use a tuple then it wouldn't be possible. Since Data frames use dictionary of lists to write data to csv
    # this is another case where we need them
    inputData   = []
    outputData  = []
    
    # Why do we need training Data?
    # We need training data to teach the model about what answer is expected from it. Example to make it learn what is 
    # a cat we have to show it pictures of cat.
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # Why Dataframe?
    # Data frames enable us to view multi-column data in two dimensions which is useful here because one part is the
    # input while the other is the output
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")

def processData(dataset):
    
    # Why do we have to process?
    # If we just input the numbers as they are then we won't be able to show patterns in the data. Each input will be 
    # connected to n hidden layers, which is useless. Another case is of dimensionality, a 4 labelled output data cannot
    # be expressed in 1dimensions(viz. Using only 1 input). We need atleast 4 or more than that so that the vectors
    # span all the dimensions
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel

def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # Why do we have number 10?
        # Because we are planning to have 10 input Layer neurons/nodes also the max number which we have to train for is
        # 1000 which takes up 10 bits after converting it to binary.
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)

def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)

input_size = 10
drop_out = 0.2
first_dense_layer_nodes  = 256
second_dense_layer_nodes = 128
final_dense_layer_nodes = 4

def get_model():

    
    
    # Why do we need a model?
    # Model is like a black box, we need it basically to teach it how to translate the inputs to required output. A model for us could
    # be a mathematical function which basically applies matrix multiplication to get the result
    
    # Why use Dense layer and then activation?
    # Dense layer is like a structure. Formally a dense layer implemnts kernel i.e The weight matrix which is used
    # while running it iterations.
    # Activation is a function of input vectors and the kernel. Therefore, we first need kernel ie. The weight matrix
    # or the dense layer before activation
    # https://keras.io/layers/core/ 
    
    # Why use sequential model with layers?
    # Sequential model allows us to design the network layer by layer. It is enough for simple cases like we have now
    # https://jovianlin.io/keras-models-sequential-vs-functional/
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('relu'))
    
    # Why dropout?
    # Avoids the model being overfitted for the training data such that it fails to generalize for new 
    # incidences.
    model.add(Dropout(drop_out))
    
    model.add(Dense(final_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax?
    # Since this is a multi class classification problem, for the last output layer we need to translate the continous signal 
    # into a discrete signal which will give us the classes. The softmax activation converts a vector of arbitary
    # real values into a vector of values in a specific range from which we can get categories.
    # Important thing to note is that the predicted class will belong to only one class, therefore sum of 
    # probabalities sums to one. For Multi label problem we have to use a different activation, perhaps 
    # sigmoid.
    
    model.summary()
    
    # Why use categorical_crossentropy?
    # Here if we have to use crossentropy loss functions we have various choices such as Binary Crossentropy, categorical
    # crossentropy and Sparse Crossentropy. Here we have One-hot Encoded the input therefore we have to use categorical
    # crossentropy.
    # https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
    
    adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0)
    
    model.compile(loss='categorical_crossentropy', optimizer=adadelta,metrics=['categorical_accuracy'])
    
    return model

# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')

model = get_model()

validation_data_split = 0.2
num_epochs = 10000
model_batch_size = 128
tb_batch_size = 128
early_patience = 200

tensorboard_cb   = TensorBoard(log_dir='./logs/kerasLogs/default/', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )

print("Average Categorical Accuracy Before Early Stopping: "+ str( np.mean(history.history['categorical_accuracy'][-early_patience:])))
print("Average Validation Accuracy Before Early Stopping: "+ str(np.mean(history.history['val_categorical_accuracy'][-early_patience:])))

# %matplotlib inline
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))

def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"

from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)

processedTestLabel = encodeLabel(testData['label'].values)

predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))
    
conf = confusion_matrix(testData['label'].values, predictedTestLabel,labels=("Fizz","Buzz","FizzBuzz","Other"))
print("The confusion matrix is:\n")
print(pd.DataFrame(conf, index=['true:Fizz', 'true:Buzz','true:FizzBuzz','true:Other'], columns=['pred:Fizz','pred:Buzz','pred:FizzBuzz','pred:Other']))

#plt.imshow(conf, cmap='binary', interpolation='None')
plt.show()

wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))
    
    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

# Please input your UBID and personNumber 
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "pkubal")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50290804")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')


M = 10 
Lambda = 0.9
E_rms Training   = 0.6427042724780092
E_rms Validation = 0.6420852168429999
E_rms Testing    = 0.6391975050630525

