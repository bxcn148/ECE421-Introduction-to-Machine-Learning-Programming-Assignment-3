Download link :https://programming.engineering/product/ece421-introduction-to-machine-learning-programming-assignment-3/


# ECE421-Introduction-to-Machine-Learning-Programming-Assignment-3
ECE421: Introduction to Machine Learning Programming Assignment 3
Objectives

The purpose of this assignment is to investigate the classification performance of neural networks. You will implement, train and evaluate your neural network models using Pytorch on the notMNIST dataset. The functions are provided and you will need to fill-in after theTODO comment instead of coding them from scratch. In particular, in the first part, you will implement a fully connected neural networks (FNN). In the second part, you will implement the model training loop. You will also be asked to answer several questions related to your implementations. You are encouraged to look up Pytorch documentation for useful utility functions, at: https://pytorch.org/docs/stable/index.html. You can also refer to the demo covered in the lecture. The notMNIST dataset is stored as notMNIST.npz inside in the assignment folder. We highly recommend students to look at the tutorial files for this assignment.

To avoid any potential installation issue, you are encouraged to develop your solution using Google Colab notebooks. It is highly recommended that you train the neural network using a GPU.

Unlike in the tutorial video which implemented convolutional neural networks, we simply im-plemented a fully-connected network for the image recognition task.

Requirements

In your implementations, please use the function prototype provided (i.e. name of the function, inputs and outputs) in the detailed instructions presented in the remainder of this document. We will be testing your code using a test function that will evoke the provided function prototype. If our testing file is unable to recognize the function prototype you have implemented, you can lose significant portion of your marks. In the assignment folder, the following files are included in thestarter code folder:

NeuralNetPyTorch.py

notMNIST.npz: this is a dataset and you can upload this file to Google Colab. We provide functions to read the data in NeuralNetPyTorch.py.

These files contain the test function and an outline of the functions that you will be implementing. You also need to submit a separate PA3 qa.pdf file that answer questions related to your implementations.

Abbreviations

Following the definition in our code, we use the following abbreivations in this document:

BATCH SIZE refers to the batch size of images we are using for training. In this assignment, we set BATCH SIZE=32.

F is an abbreviated import name of torch.nn.functional.

nn is an abbreviated import name of torch.nn.

Preliminaries

In this part, we explain several helper functions/classes in the NeuralNetPyTorch.py file. Do not modify any of these functions/classes.

Function 1: loadData(datafile=”notMNIST.npz”)

– Inputs: datafile

The default input is ””notMNIST.npz”, which is the dataset we are using in this assignment.

– Output: trainData, validData, testData, trainTarget, validTarget, testTarget The outputs are images and annotations in the form of Numpy matrices. trainData and trainTarget are the images and annotations for training. Similarly, validData and validTarget are the images and annotations for validation and testData and testTarget are the images

and annotations for testing.

– Functionality:

This function loads the notMNIST dataset and splits it into training, validation and testing set.

Class 1: class notMNIST(Dataset)

– Functionality:

This class implements a PyTorch Dataset class for the notMNIST dataset.

Read the description of the function below as you will use it in this assignment

Function 2: def experiment(learning rate=0.0001, num epochs=50, verbose=False):

– Inputs: learning rate, num epochs, verbose

The input learning rate is a scalar of type float that specifies the learning rate. The input num epochs is an integer that specifies how many times your model will train through the whole training set. The input verbose is a Boolean that will let you print the training process if is True and nothing if is False.

– Output: trained model and training history

This function returns a trained model, which is a torch.nn.Module class and its training history. The training history is a dictionary that contains the accuracy on training, validation and test sets at each epoch, which is returned after calling the train function. Note: You will implement the train function in Part 3.

– Functionality:

This function will build your model, train it on the notMNIST dataset using the specified hyperparameters and return a trained model with its training history.

Part 1: Fully Connected Neural Networks

In this part, you will be implementing a Fully Connected Neural Network using PyTorch. The model is defined in class FNN(nn.Module). The neural network architecture that you will be implementing is as the following order:

An input layer for the images of size (BATCH SIZE x 1 x 28 x 28). The second dimension is the image channel, which is 1 in this case since we are using gray-scale images. The third and forth dimension are the width and height of the input respectively. We will transform the (BATCH SIZE x 1 x 28 x 28) matrix into a batch of 1D arrays of size (BATCH SIZE x 784).

A fully connected layer followed by a ReLU activation. The size of the weight matrix is 784×10 where 784 is the size of the input array and 10 is the size of the 1st hidden layer.

A fully connected layer followed by a ReLU activation. The size of the weight matrix is 10×10 where 10 is the size of the 1st hidden layer and 10 is the size of the 2nd hidden layer.


A fully connected layer (without softmax activation). The size of the weight matrix is 10×10 where 10 is the size of the 2nd hidden layer and 10 is the size of the output layer.

Specifically, you will be implementing two functions in class FNN(nn.Module), which are detailed in the following:

• Function 1: def init (self)

– Inputs: This function does not require any input

– Output: This function does not return any output

The purpose of this function is to setup the variables for this class. As shown in the tutorial, you need to define all the layers you will be using in this function.

– Function implementation considerations:

You will use the following PyTorch functions to set up the layers in your network: nn.Linear(). The nn.Linear() function is a fully connected layer and is defined by the dimension of the input and output. For example, nn.Linear(3, 4) is a fully-connected layer for an input of dimension (BATCH SIZE, 3) and an output of dimension (BATCH SIZE, 4).

Function 2: forward(self, x)

– Inputs: self, x

The input x is the batch of images of size (BATCH SIZE, 1, 28, 28). The input self represents the instance of the class.

– Output: out

This function computes the logits for each image in the batch. The output has a size of (BATCH SIZE, 10), where each (i,j) position represent the class-logit score j of image i. The order of the layers (and activations) in this function must follow the described network architecture above.

– Function implementation considerations:

You will find the following PyTorch functions helpful: torch.flatten(x, start dim=1), F.relu(). The torch.flatten(x, start dim=1) operation will flatten your input x of size (BATCH SIZE, 1, 28, 28) to the size of (BATCH SIZE, 784). Functions F.relu() applies the ReLU() activation to the input respectively.

The following is the mark breakdown for Part 1:

Test file successfully runs implemented function: 15 marks

Output is close to the expected output from the test file: 15 marks

Part 2: Model Training and Experiments

In this part, you will be implementing the training procedure for your neural networks. You will use the cross-entropy loss, Adam optimization method. Implement (Complete) the following functions:

Function 1: def get accuracy(model, dataloader):

– Inputs: model, dataloader

model is an instance of the neural network class. dataloader is an instance of the notMNIST dataloader (see the function experiment).

– Output:

The output of this function is a scalar, which is the accuracy over all images in dataloader.

– Function implementation considerations:

This function will calculate the classification accuracy over the dataloader we specifed. It will be called whenever your model finishes one training epoch in the train function. Remember to set model.eval() to get the model into evaluation mode.


Function 2: def train(model, device, learning rate, train loader, val loader, test loader, num epochs=50, verbose=False):

– Inputs: model, device, learning rate, train loader, val loader, test loader, num epochs=50, verbose

model is an instance of the neural network class. To train the neural networks with GPU, device should be “cuda:0” as shown in the experiment function. The learning rate is a float scalar specifying the learning rate at training. train loader, val loader, test loader are the notMNIST dataloader for training, validation and testing set respec-tively. num epochs is the number of passes through the dataset we would like in our training (default is 50 in this assignment). Finally, set verbose to True will print out the training progress.

– Output: model, acc hist

This function will return a trained model and the evolution of training, validation and testing accuracy.

– Function implementation considerations:

First, you need to specify the cross-entropy loss through the criterion variable that we will use as an optimization objective. After that, set the AdamW optimizer through the optimizer variable (PyTorch Link) using the learning rate and the model’s weights (to be optimized). After that, complete the backpropagation process (forward, backward, update weights) in the training loop in the function. Remember to set the weight’s gradients to zero.

Complete the following experiments, write and save your answer in a separate PA3 qa.pdf file. You will also need to write functions that prints/plots these experiment in your NeuralNetPyTorch.py sub-mission. Remember to submit this file together with your code. Use theexperiment function to complete these experiments. For simplicity, we will discard the validation accuracy since fine-tuning the hyper-parameters for neural networks is a very time-consuming process. To run the experiment, use the provided experiment function.

Experiment: In this experiment, you will train the FNN in the image recognition task. Train your FNN model for a batch size of 32 (default) for 50 epochs (default) and the AdamW optimizer (Link to AdamW) for learning rate of 0.1, 0.01, 0.001, 0.0001. Plot and compare the training/testing history of these models, namely we want to see how the training progresses with different learning rates. The function name for this experiment will be compare lr(). Comment on what you observed from the experiment and explain.

The following is the mark breakdown for Part 2:

Test file successfully runs implemented function: 6 marks

Output is close to the expected output from the test file: 18 marks

Questions are answered correctly: 6 marks
