# ChessAI

## Introduction

This is my final project for my deep learning course for the fall semester 2018.
The goal of the project is to figure out different ways to create an AI
for chess that can beat me.

Four different AIs were created which are as follows:
	
	* Alpha-beta algorithm with classical value function
	* Alpha-beta algorithm with value network (pyTorch)
	* Policy and value network without Monte Carlo Tree Search
	* Policy and value network with Monte Carlo Tree Search

## Requirements:

Python3 modules: flask, pytorch, numpy, python-chess, Keras, tensorflow

## Usage:

To create dataset for supervised learning put all the game files (pgn files) in
the folder called "data". Then create an empty folder called "processed" to store
the processed training data.

To create dataset for the pyTorch model (value network):

	* python generate_training_set.py --type pytorch -n 1000000

To create dataset for the keras model (policy and value network):

	* python generate_training_set.py --type keras -n 10000

** NOTE: The processed dataset for the keras model will take lots of hard disk space.


To play against the classical model with no deep learning (with alpha beta algorithm),
execute the following command:

	* python play.py

To play against the pytorch model, execute the following command:
	
	* python DLplay.py

To play against the keras model without Monte Carlo Tree Search, execute the following command (faster):
	
	* python DLplay_keras.py

To play against the keras model with Monte Carlo Tree Search, execute the following command (slower):
	
	* python DLplay_keras_MCTS.py

## Model Architecture

pyTorch model architecture is given below.

![pytorch architecture](https://github.com/nightstorm0909/ChessAI/blob/master/images/pyTorch_model.png)

Keras model architecture is given below. It is a residual network with 5 residual
modules. It has both policy and value network.

![keras architecture](https://github.com/nightstorm0909/ChessAI/blob/master/images/model.png)