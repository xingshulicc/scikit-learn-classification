# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Dec 05 14:50:56 2017

@author: xingshuli
"""
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata

from sklearn.neural_network import MLPClassifier

#load data
mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


#create mlp classifier
mlp = MLPClassifier(hidden_layer_sizes = (50,), activation = 'relu', 
                    solver = 'sgd', alpha = 1e-4, batch_size = 32, learning_rate = 'adaptive', 
                    learning_rate_init = 0.001, max_iter = 300, shuffle = True, 
                    random_state = 1, tol = 1e-4, verbose = 300, momentum = 0.9, 
                    nesterovs_momentum = True)

#train the model
mlp.fit(X_train, y_train)

#draw loss figure
MLP_loss = mlp.loss_curve_
Epoches = len(MLP_loss)
x_axis = range(1, (Epoches + 1))
y_axis = MLP_loss
plt.plot(x_axis, y_axis, 'b:')

print('Training set score: %f' % mlp.score(X_train, y_train))
print('Test set score: %f' % mlp.score(X_test, y_test))

plt.show()




