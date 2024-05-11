#!/usr/bin/env python3

import numpy as np
import json



def relu( x ):
    return ( ( x > 0 ) + ( x < 0 ) * .01 ) * x
def drelu( x ):
    return ( x > 0 ) + ( x < 0 ) * .01
def sigmoid( x ):
    return 1 / ( 1 + np.exp(-x) )
def dsigmoid( x ):
    return sigmoid(x) * ( 1 - sigmoid(x) )
act = relu
dact = drelu
"""
act = sigmoid
dact = dsigmoid
"""


# load data

with open( "wordmap.json", mode="r" ) as f:
    wordmap = json.load( f )

vocab_size = 10000
samples = 10000
x = []
y = []

with open( "reviews.txt", 'r' ) as f:
    for line in f:
        bag = np.zeros(vocab_size)
        words = [
            wordmap[word]
            for word in line.strip().split(" ")
            if word and word in wordmap and wordmap[word] < vocab_size
        ]
        for word in words:
            bag[word] += 1 / len(words)
        x.append(bag)
        if len(x) == samples:
            break

with open( "labels.txt", 'r' ) as f:
    for line in f:
        y.append( [1] if line.strip() == "positive" else [0] )
        if len(y) == samples:
            break

x = np.array(x)
y = np.array(y)

x_train, x_test = x[:9000], x[9000:]
y_train, y_test = y[:9000], y[9000:]

obss, obss_test = x_train, x_test
targets, targets_test = y_train, y_test

# define funcs

def predict( obs, weights ):
    res = act( weights @ obs )
    return res

def loss( obs, weights, target ):
    pureloss = predict( obs, weights ) - target
    return pureloss.T @ pureloss

def check( obs, weights, target ):
    return abs( predict( obs, weights )[0] - target[0] ) < .5

def test_eval( obss, weights, targets ):
    return sum(
        check( obs, weights, target )
        for obs, target in zip(obss, targets)
    ) / targets.shape[0]

def dldw( obs, weights, target ):
    z_out = weights @ obs
    pred = act( z_out )

    error_pred = 2 * ( pred - target ) * dact(z_out)

    dlossdw = np.outer( error_pred, obs )

    return dlossdw


# init network

np.random.seed(1)
weights = np.random.rand(1, vocab_size) - .5
weights *= .2
a = 0.05

# training loop

try:
    for epoch in range(101):
        for i, obs in enumerate(obss):
            target = targets[i]
            dw = dldw( obs, weights, target )
            weights -= a * dw
        print(
            "epoch", epoch,
            "- test accuracy", f"{test_eval( obss_test, weights, targets_test ) * 100:.1f}%"
        )

except KeyboardInterrupt:
    print("ending training!")

# save the weights to a file

np.savez( "weights.npz", weights=weights )

# to load the weights

with np.load( "weights.npz", allow_pickle=True ) as f:
    weights = f["weights"]

