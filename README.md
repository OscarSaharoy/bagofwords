this is a simple single layer sentiment analysis classifier based on a bag of words input vector from a 10000 word dictionary. the simple architecture is able to reach 84% accuracy on the imdb reviews dataset while remaining very explainable.

## training the network

make sure numpy is installed:
```
python3 -m pip install numpy
```
run the training process:
```
python3 network.py
```
after this the weights will be saved to `weights.npz`.

## inference

you can also run the network for different inputs like this, where a high output is a positive prediction:
```
$ python3 infer.py "This is a great movie, loved every second and will be watching again!"
[2.15298487]
$ python3 infer.py "I am not sure about this movie, maybe skip this one"
[-0.00422019]
```

## results

by running the infer script with single words, you can read out the weights for those words in the network. these are embeddings in a 1-dimensional sentiment space, where higher numbers are positive and lower numbers are negative:

```
great    11.23366169
amazing  3.94121445
award    0.93138511
loved    3.71383831

bad      -14.9690355
terrible -5.77687847
boring   -6.71697652
hated    -0.51254948
```

the model has also learned a gender bias:

```
actor   -1.23702685
actress -0.39061272

he      1.87330988
she     2.40016936

him     2.24494896
her     3.67306853
```

and the network can also generalise to non-review inputs:

```
$ ./infer.py "you are my everything"
[2.22616088]
$ ./infer.py "you smell really bad and its kind of unpleasant to be around you"
[-0.00081939]
```

## further work

by removing all weights close to 0 and all the corresponding words from the input vector, you could potentially reduce the size of the network and increase training and inference speed.

