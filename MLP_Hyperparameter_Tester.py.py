import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from datetime import datetime

# importing dataset and storing it in arrays
fashion_data = keras.datasets.fashion_mnist
(train_imgs, train_lbls) , (test_imgs, test_lbls) = fashion_data.load_data()

# flattening to a 1d array
train_imgs = train_imgs.reshape(60000, 784)
test_imgs = test_imgs.reshape(10000, 784)

train_imgs = train_imgs.astype('float32')
test_imgs = test_imgs.astype('float32')

# normalizing the data for faster processing
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# arrays to load hyperparameters
hidden_layers = [1]
neurons = [64, 128]
learning_rate = [0.05]
epochs = [10]

# keeps track of Results objects
results = []
accuracy = []

# class used to store the network objects (models) and their parameters and results
class Results :
    def __init__(self) :
        pass
    
    def setValues(self, model, parameters, train_loss, train_acc, test_loss, test_acc):
        self.model = model
        self.parameters = parameters
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.test_loss = test_loss
        self.test_acc = test_acc
    
    def display(self) :
        string = "\nHyperparameters: {} \nTrain loss: {:.4f} \nTrain accuracy: {:.4f} \nTest loss: {:.4f} \nTest accuracy: {:.4f}".format(self.parameters, self.train_loss, self.train_acc, self.test_loss, self.test_acc)
        return string

def createGraphs(result, parameters) :
    figure = plt.figure()
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title(parameters)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.grid()
    figure.savefig('./graphs/loss/' + parameters + '.png', bbox_inches='tight')
    plt.close()
    
    figure2 = plt.figure()
    plt.plot(result.history['accuracy'])
    plt.plot(result.history['val_accuracy'])
    plt.title(parameters)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.grid()
    figure2.savefig('./graphs/accuracy/' + parameters + '.png', bbox_inches='tight')
    plt.close()

# creates the neural networks, trains and evaluates them, creates csv files and calls the createGraphs method
def testNetwork(num_layers, neurons, lr, num_epochs, train_imgs, train_lbls, test_imgs, test_lbls) :
    if num_layers == 1:
        model = keras.Sequential([
            keras.layers.Dense(784,input_shape=(784,)),
            keras.layers.Dense(neurons, activation='sigmoid'),
            keras.layers.Dense(10)
        ])
    # two hidden layers
    else :
        model = keras.Sequential([
            keras.layers.Dense(784,input_shape=(784,)),
            keras.layers.Dense(neurons, activation='sigmoid'),
            keras.layers.Dense(neurons, activation='sigmoid'),
            keras.layers.Dense(10)
        ])
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate = lr), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy']
    )
    
    parameters = "Layers " + str(num_layers) + ", Neurons " + str(neurons) + ", Learning Rate " + str(lr) + ", Epochs " + str(num_epochs)
    
    # used as a callback to store results of each epoch
    csv_logger = CSVLogger('./csv/log ' + parameters + '.csv', append=True, separator=',')

    # training the network
    result = model.fit(train_imgs, train_lbls, epochs=num_epochs, validation_split=0.2, callbacks=[csv_logger], verbose=1)

    # saving the values at the end of training
    train_loss = result.history['loss'][-1]
    train_acc = result.history['accuracy'][-1]

    # evaluating and storing test results on unseen images
    test_loss, test_acc = model.evaluate(test_imgs,  test_lbls, verbose=1)
     
    createGraphs(result, parameters)
    
    # storing the model and its metrics into an object
    output = Results()
    output.setValues(model, parameters, train_loss, train_acc, test_loss, test_acc)
    
    return output

# loops to switch hyperparameters and test
for h in range(len(hidden_layers)) :
    for i in range(len(neurons)) :
        for j in range(len(learning_rate)) :
            for k in range(len(epochs)) :
                result = testNetwork(hidden_layers[h], neurons[i], learning_rate[j], epochs[k], train_imgs, train_lbls, test_imgs, test_lbls)
                results.append(result)
                accuracy.append(result.test_acc)

# finding the highest accuracy
best_accuracy = max(accuracy)
best_result = Results()

# finding the results object that has the network with the best accuracy
for i in range(len(results)) :
    if best_accuracy == results[i].test_acc:
        best_result = results[i]
        break

print("\nResults Summary:")
for i in range(len(results)) :
    print(results[i].display())

print("\nBest result:" + best_result.display())

# saving the results summary to a txt file
with open('./summary/' + datetime.now().strftime("%d-%m-%Y, %H-%M-%S") + '.txt', 'a') as f:
    for r in results :
        print(r.display(), file=f)
    print("\nBest result:" + best_result.display(), file=f)

# storing the best network
best_model = best_result.model

# Add a softmax layer to convert to an easier to understand format for predictions
prob_model = tf.keras.Sequential([best_model, tf.keras.layers.Softmax()])

# array with the names of the labels as they are not included in the labels arrays
lbl_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# prompting the user to select an img from the test set and the network makes a prediction
while True :
    x = input("\nSelect an image from test dataset (0-9999): ")
    if x.isdigit() and 0 <= int(x) <= 9999:
        x = int(x)
        img = test_imgs[x]
        # Add the image to a batch for processing
        img = (np.expand_dims(img,0))
        # make prediction
        single_prediction = prob_model.predict(img)
        # get the value with the highest probability
        result = np.argmax(single_prediction[0])
        print("Predicted: ", lbl_names[result], result, "\nActual: ", lbl_names[test_lbls[x]], test_lbls[x])
    
    else :
        print("Invalid input! Please select a number in the range 0-9999: ")

    if input("Would you like to exit? y/n: ") == 'y' :
            break