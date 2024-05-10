#!/usr/bin/env python3

import numpy as np
import json

"""
# load dataset

with np.load( "dataset.npz", allow_pickle=True ) as f:
    weights = [ f["w1"], f["w2"] ]
"""

with np.load( "words.npz", allow_pickle=True ) as f:
    words = f["words"]

with open( "wordmap.json", mode="r" ) as f:
    wordmap = json.load( f )


