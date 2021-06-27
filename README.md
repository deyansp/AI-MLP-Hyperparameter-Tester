# Overview
The aim of this project was to give an in-depth look into the classic neural network: multi-layered perceptron (MLP) by measuring the performance of different network configurations on a multi-class image classification task. Each network has a different set of hyperparameters (hidden layers, neurons, learning rate, and epochs).

# Methodology
The program uses the Fashion MNIST dataset which contains 70,000 28x28 grayscale images of clothing in 10 categories. The `TensorFlow` and `Keras` libraries are used to import the dataset and to create, train and evaluate the neural networks. The dataset is split into 60 000 images for training, 12 000 (20%) of which are used as a validation set to continuously monitor performance on unseen images after each epoch. The remaining 10 000 images are used during the evaluation phase to measure the performance of the fully trained network.

The process begins by storing the training and test images and labels into 4 arrays. Image arrays are flattened from 2D 28x28 arrays to a one dimensional 784-element array, which is a suitable format for the neural network to process. 

A `testNetwork` method is used to create a network by using `TensorFlow` and the hyperparameter arrays as input. Then the model is trained and evaluated with the appropriate library methods. The results from each epoch are logged into a CSV file. 

`Matplotlib` is used to create loss and accuracy graphs through a `createGraphs` method. A `Results` object is used to store the network, a string with its hyperparameters, and its train and test loss and accuracy values. The `testNetwork` method is called inside nested for loops which loop through the hyperparameter arrays and change one parameter at a time. Each `Results` object is stored in an array. 

![Example Graphs Generated](https://raw.githubusercontent.com/deyansp/AI-MLP-Hyperparameter-Tester/main/sample-graphs.PNG)

After all network configurations have been evaluated, the results array is looped through to find the network with the highest test accuracy. A summary text file is generated with all the results and the best one is highlighted. The user is prompted to select an image index from the test data and the best network is used to make predictions. The program displays the network's prediction along with the correct value so that the user can verify the network's capabilities. 

# Results and Conclusion
From comparing different configurations, the learning rate and the number of epochs had the biggest effect on network performance. The number of layers and neurons did have an impact, especially on very high or very low learning rates. 

![Results Table]( https://raw.githubusercontent.com/deyansp/AI-MLP-Hyperparameter-Tester/main/results-table.PNG)

As we can see from the table above, a network with just 64 neurons and 1 hidden layer achieved almost the same accuracy as ones with 256 neurons and 2 layers. For this dataset, a large number of neurons or multiple hidden layers are not necessary as they are more computationally expensive and do not appear to add a significant benefit. 

It would be more feasible to explore using the smaller networks with 64-128 neurons and a learning rate of 0.05 and decreasing this rate over time to possibly achieve a higher accuracy. A higher number of neurons and hidden layers would be more useful for a larger dataset, but the Fashion MNIST dataset is relatively small with 70 000 images. 

This project provided a good baseline for possible hyperparameter optimizations in the future. More experimentation would be necessary to achieve a better accuracy and a lower loss. The program created for this project can be quite versatile as it can easily be used to test other configurations by changing the array values. The reports and graphs it generates are highly beneficial in assessing performance.
