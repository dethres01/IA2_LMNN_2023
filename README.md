# **Neural Network with Levenberg-Marquardt Algorith**

## Overview

This Python code implements a neural network using the Levenberg-Marquardt algorithm for binary classification. The neural network has two inputs and one output, making it suitable for binary classification tasks. The sigmoid function is used as the activation function, and the mean squared error is employed as the loss function.

The Levenberg-Marquardt algorithm is an optimization technique for solving nonlinear least squares problems.
These problems are common in many areas of science and engineering, such as curve fitting, solving systems of nonlinear equations and function optimization.
This algorithm combines two methods: Gauss-Newton and gradient descent.
The Gauss-Newton method is fast but can be unstable, while the gradient descent method is slow but stable.
The Levenberg-Marquardt algorithm attempts to maximize the advantages of both methods by adjusting a parameter (λ) that determines how much to rely on each method at each step of the algorithm.


![Formula Levenberg-Marquardt](image.png)

## Requirements

### To compile and run this code, you need:

Python: Ensure you have Python installed (version 3.x).

Required Libraries: Install the necessary libraries using the following:


    pip install numpy matplotlib


## How to Run

    Copy the code into a Python environment.
    Ensure the required libraries are installed.
    Run the script.

>[!NOTE]
>Creating a neural network with the Levenberg-Marquardt algorithm involves several steps. Below is a step-by-step guide to help you understand how to implement a neural network with the Levenberg-Marquardt algorithm:

## Results

The code trains a neural network using the Levenberg-Marquardt algorithm and visualizes the decision boundary along with the sine wave. The training error plot is also displayed.

Note: This code is designed for educational purposes and may need further optimization for production use.