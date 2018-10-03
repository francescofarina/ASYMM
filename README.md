# ASYMM

This repository contains a python implementation of ASYMM: the ASYnchronous Method of Multipliers.
ASYMM is an algorithm for distributed constrained nonconvex optimization which has been presented in

>[Francesco Farina, Andrea Garulli, Antonio Giannitrapani, Giuseppe Notarstefano. Asynchronous Distributed Method of Multipliers for Constrained Nonconvex Optimization. In press in European Control Conference (ECC) 2018, Preprint in arXiv, 2018.](https://arxiv.org/pdf/1803.06482.pdf).

>[Francesco Farina, Andrea Garulli, Antonio Giannitrapani, Giuseppe Notarstefano. Distributed Constrained Nonconvex Optimization: the Asynchronous Method of Multipliers. 2018.](https://arxiv.org/pdf/1803.06482.pdf)

## Requirements
Required packages for running the code in this repository are reported in [requirements.txt](./requirements.txt) and are:

- numpy
- matplotlib
- scipy
- networkx

## Content
The class Node, which is used for simulating nodes in the considered network, is defined in the file [node_definition.py](./node_definition.py).

The script in [launcher.py](./launcher.py) simulates a network of N nodes which have to localize an emitting source in a 2D environment.
