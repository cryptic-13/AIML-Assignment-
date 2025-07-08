# 🧠 AIML Assignment – Neural Network & Ensemble Learning

This project contains two main components:

1. **Neural Network from Scratch**: A fully connected neural network implemented using NumPy to classify handwritten digits from the MNIST dataset.
2. **Ensemble Learning Model**: Combines XGBoost and Random Forest classifiers to predict obesity levels using a dataset with 17 features.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Instructions To Run](#instructions-to-run)
5. [Expected Output](#expected-output)
---

## 📘 Project Overview

This repository demonstrates:
- A low-level implementation of a feedforward neural network with backpropagation and custom activation functions (ReLU, Softmax).
- An ensemble learning approach that merges Random Forest and XGBoost classifiers to improve prediction accuracy on health-related data.

---

## ✨ Features

### `neural-network.py`
- Using Numpy, Matplotlib and Tensorflow to build a neural network from scratch on the MNIST Database, for number recognition on images of handwritten digits. 
- Trains a 3-layer neural network on MNIST.
- Implements:
  - Xavier weight initialization
  - ReLU and Softmax activations
  - Forward and backward propagation
  - Manual accuracy calculation

Reading through this project demonstartates a thorough breakdown of how neural networks, receive, process and train on data from a dataset to predict outputs. 

This code works on the MNIST database, containing images of handwritten digits, which the model trains to read and identify. It doesnt recognize shapes, or characteristic parts of the numbers, like say, narrowing the possible numbers down to 2,3,5,6,8,9 or 0 as a curved feature was present and so on. It makes predictions based on pixel brightness and activates neurons in each layer accordingly. The activation level of each neuron in a particular layer plays a role in deciding the activation level of every other neuron in the next layer. This is known is forward propagation. 

The final layer consists as many neurons as there are total number of possible outputs, for recognizing a digit, there are 10 possible outcomes (0-9) and thus the final layer has 10 neurons. The one with the maximum acitvation out of these 10 neurons becomes the predicted number. 

While training the neural network, if the predicted number does not match the actual label, back propagation adjusts the weights and biases such that the correct prediction is made using gradient descent to minimize the losses and the gradients computed thus inform how much each paramter contributes to the loss (error in prediction). 

After training on all the examples in the dataset, the weights and biases which produce correct identification for maximum number of training examples are selected as the 'optimized parameters' and a prediction made on any new input (not part of the training dataset) is made using these final parameters. New inputs are used to gauge the performance of the model (based on accuracy and indications of overfitting) to see how well what the neural network has learned from training applies to new data inputs. 

Use the following flowchart to visualze how a neural network works in steps. 

![image](https://github.com/user-attachments/assets/79867849-45e5-4433-be36-ed54f36233cc)

The program makes the output more comprehendable by printing the image taken from the training dataset, displaying its true label and the model's prediction. This is a more layman and visual way of understanding how the model works, as an added feature. 

---

### `ensemble2.py`
- Creating an ensemble model using XGBoost and Random Forest as base models, for multi-class classification of obesity levels on a dataset of 17 paramteres of various data types. 
- Trains a Random Forest and XGBoost model on an obesity dataset (`dataset.csv`)
- Performs predictions, calculates accuracy, and visualizes confusion matrices
- Merges predictions using majority voting (ensemble)

For both projects, performance metrics used were: Confusion Matrix, Precision, Recall, Accuracy and F1 score 


---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/cryptic-13/AIML-Assignment-.git
cd AIML-Assignment-
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate     # On Windows use: venv\Scripts\activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

---


## Instructions To Run 
- Ensure the environment you are running these files on : TensorFlow, Keras, Numpy, Matplotlib, Seaborn, Scikit-Learn, xgboost and Pandas.
- If not, 'pip-install' suitable versions to your environment, and ensure you are working in a python version between 3.9 and 3.12 as TensorFlow is not compatible with more recent versions.



## Expected Output

### Expected Output of ensemble2.py 
![image](https://github.com/user-attachments/assets/c24c4724-0dd4-48d2-8234-6cd28161f53b)


## Expected Output of of neural-network.py
![image](https://github.com/user-attachments/assets/61c584e6-bc6a-47a8-a850-41be86d17408)
![image](https://github.com/user-attachments/assets/6650c19d-2460-4dfd-bc50-d5a679e17356)
![image](https://github.com/user-attachments/assets/83a6c099-ede2-40e4-8bc1-e9a63765ba15)
![image](https://github.com/user-attachments/assets/e00c3846-ef97-4d89-abf0-150cb9f2b20d) 


