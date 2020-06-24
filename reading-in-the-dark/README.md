# Reading In The Dark: Classifying Encrypted Digits with Functional Encryption

This code implements the cryptographic part of
[**Partially Encrypted Machine Learning using Functional Encryption**](https://arxiv.org/abs/1905.10214) (to appear at
NIPS 2019). The code for the Collateral Learning component can be found
[**here**](https://github.com/LaRiffle/collateral-learning).

## Running the provided code

### Testing

Start by running a simple end-to-end test.
~~~~
cd mnist
python3 ml_test.py
~~~~
This will pick a random digit from the MNIST test set, encrypt it, and evaluate the provided model on it using functional decryption.

### Generating the confusion matrix

~~~~
cd mnist
python3 confusion_matrix.py
~~~~
This operates on cleartext data.

### Benchmarking the scheme on your hardware

~~~~
cd mnist
python3 -O benchmark.py
~~~~

## Project structure

```
.
├── core    # The code for the Quadratic FE scheme, optimized for 1 hidden layer Quadratic Networks.
│   ├── discretelogarithm.py    # A discrete logarithm solver that uses the Baby Step Giant Step method and interacts
|   |                           # with a database to store and retrieve procomputations.
│   ├── make_keys.py    # A simple key generation function.
│   ├── models.py   # Classes and methods for vectors, cryptographic keys and machine learning models.
│   ├── scheme.py   # The implementation of the Quadratic Functional Encryption scheme.
│   └── utils.py    # Defines some utilities, e.g. for serializing objects and batch exponentiations.
└── mnist   # Code testing and benchmarking the implementation on the MNIST dataset.
    ├── benchmark.py    # Runs encryption and decryption on the whole MNIST test set and reports the timings.
    ├── confusion_matrix.py # Generates the confusion matrix found in the paper.
    ├── initialization.py   # A run once script to run before running anything else. Generates keys and fills
    |                       # the database with discrete logarithm precomputations, which takes some time.
    ├── ml_test.py  # Picks a MNIST digit at random, encrypts it and functionally decrypts it.
    ├── objects # A directory of directories where objects are stored durably. Should initially contain an
    |           # instantiation (a choice of Pairing Group with base elements) and a pretrained MNIST model.
    └── setup.py    # Used to make importing from core easier.
```
