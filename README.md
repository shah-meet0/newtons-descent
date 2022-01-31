# Information
Contains python code for numerical estimation of gradient and Hessian, along with Newton's Method and Gradient Descent. Part of a project on parallelization in statistics in ST444- Computational Data Science

Co-authors: Xiaoyi Zhu, David Ni

Contains embarassingly parallel implementations of numerical estimations of gradient and Hessian.

# Contents
  - [gradientLib](./gradientLib.py): This module contains code for numerically estimating the gradient and Hessian of a given function at a point. Also contains a decorater for timing functions. 
  - [optimLib](./optimLib.py): This module contains parallelized and non-parallelized implementations of Gradient Descent and Newton's Method. 
  - [functionLib](./functionLib.py): This module contains some common scalable functions used for testing optimization algorithms. 
  - [working_example](./working_example.py): Module contains some short usage scenarios, and tests parallelization versus non-parallelization. 
