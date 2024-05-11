#!/usr/bin/env python3

import numpy as np
import json
import sys


def relu( x ):
    return ( ( x > 0 ) + ( x < 0 ) * .01 ) * x
act = relu


# load wordmap

with open( "wordmap.json", mode="r" ) as f:
    wordmap = json.load( f )

vocab_size = 10000

# define funcs

def encode_review( review ):
    bag = np.zeros(vocab_size)
    words = [
        wordmap[word]
        for word in review.strip().split(" ")
        if word and word in wordmap and wordmap[word] < vocab_size
    ]
    for word in words:
        bag[word] += 1 / len(words)
    return bag

def predict( obs, weights ):
    res = act( weights @ obs )
    return res

# load the weights

with np.load( "weights.npz", allow_pickle=True ) as f:
    weights = f["weights"]

print( predict( encode_review(sys.argv[1]), weights ) )

