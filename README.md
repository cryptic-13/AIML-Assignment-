# Neural-Network-From-Scratch And Ensemble Learning 
## Using Numpy, Matplotlib and Tensorflow to build a neural network from scratch on the MNIST Database.
## Creating an ensemble model using XGBoost and Random Forest as base models. 
## For both projects, performance metrics used were --> Confusion Matrix, Precision, Recall, Accuracy and F1 score 


## Instructions To Run --> 
-- Ensure the environment you are running these files on have--> TensorFlow, Keras, Numpy, Matplotlib, Seaborn, Scikit-Learn, xgboost and Pandas.
-- If not, 'pip-install' suitable versions to your environment, and ensure you are working in a python version between 3.9 and 3.12 as TensorFlow is not compatible with more recent versions. 

Reading through this project demonstartates a thorough breakdown of how neural networks, receive, process and train on data from a dataset to predict outputs. 


This code works on the MNIST database, containing images of handwritten digits, which the model trains to read and identify. 


It doesnt recognize shapes, or characteristic parts of the numbers, like say, narrowing the possible numbers down to 2,3,5,6,8,9 or 0 as a curved feature was present and so on. 
It makes predictions based on pixel brightness and activates neurons in each layer accordingly. The activation level of each neuron in a particular layer plays a role in deciding the activation level of every other neuron in the next layer. This is known is forward propagation. 


The final layer consists as many neurons as there are total number of possible outputs, for recognizing a digit, there are 10 possible outcomes (0-9) and thus the final layer has 10 neurons. The one with the maximum acitvation out of these 10 neurons becomes the predicted number. 


While training the neural network, if the predicted number does not match the actual label, back propagation adjusts the weights and biases such that the correct prediction is made using gradient descent to minimize the losses and the gradients computed thus inform how much each paramter contributes to the loss (error in prediction). 


After training on all the examples in the dataset, the weights and biases which produce correct identification for maximum number of training examples are selected as the 'optimized parameters' and a prediction made on any new input (not part of the training dataset) is made using these final parameters. 


New inputs are used to gauge the performance of the model (based on accuracy and indications of overfitting) to see how well what the neural network has learned from training applies to new data inputs. 
Use the following flowchart to visualze how a neural network works in steps. 


![image](https://github.com/user-attachments/assets/79867849-45e5-4433-be36-ed54f36233cc)

The program makes the output more comprehendable by printing the image taken from the training dataset, displaying its true label and the model's prediction. This is a more layman and visual way of understanding how the model works, as an added feature. 

## Expected Output on running the ensemble2.py file --> 
![image](https://github.com/user-attachments/assets/c24c4724-0dd4-48d2-8234-6cd28161f53b)

## Expected Output on running the neural network file--> 
![image](https://github.com/user-attachments/assets/61c584e6-bc6a-47a8-a850-41be86d17408)
![image](https://github.com/user-attachments/assets/6650c19d-2460-4dfd-bc50-d5a679e17356)
![image](https://github.com/user-attachments/assets/83a6c099-ede2-40e4-8bc1-e9a63765ba15)
![image](https://github.com/user-attachments/assets/e00c3846-ef97-4d89-abf0-150cb9f2b20d) 








