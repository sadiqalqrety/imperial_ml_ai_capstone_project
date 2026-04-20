# imperial_ml_ai_capstone_project
A capstone project for the educational course; Professional Certificate in Machine Learning and Artificial Intelligence


# Project Overview

## Description and purpose

There are eight BBO functions. The number of inputs vary from two to eight variables per function, and there is one output per function. The objective for every function is maximisation. The domains vary from chemical yields, cooking recipes and model performances. In short, the relationship between input and output is non-linear, and the landscape of output vary in terms of single and/or multiple peaks and valleys.

The project models real-world challenges in machine learning. Many at times, in specialised domains, involving complex modelling, the unintuitive and ambiguous relationship between input and output remains difficult to optimise for. Hence, the ability to reason about such problems is key to building machine learning models that are able to perform robustly and scale to unseen data.

The project is an insightful hands-on experience to these low-level mathematical challenges, and will greatly improve the ability to fine-tune foundation models for domain specific tasks.

## Inputs and outputs

The inputs are numerical values between zero and one, see example inputs, per function, below.

Function 1:	[0.839195, 0.748743]
Function 2:	[0.575757, 0.999999]
Function 3:	[0.000015, 0.999999, 0.999999]
Function 4:	[0.000050, 0.406177, 0.000050, 0.999950]
Function 5:	[0.000050, 0.000050, 0.066897, 0.999959]
Function 6:	[0.453794, 0.184473, 0.988142, 0.998891, 0.050898]
Function 7:	[0.000050, 0.411098, 0.228833, 0.144114, 0.346375, 0.736693]
Function 8:	[0.000050, 0.061737, 0.126822, 0.000050, 0.552527, 0.573729, 0.000050, 0.241814]




See example outputs, below.

Function 1:	-1.3793248852179627e-42
Function 2:	0.08084127857059048
Function 3:	-0.501755912728561
Function 4:	-27.92656455534856
Function 5:	190.72970845678554
Function 6:	-0.8394620905844289
Function 7:	1.5250340981229598
Function 8:	9.8208672019139

We are further informed of the domains for each of the functions, see below.

Function 1:	Maximisation, detect likely contamination sources in a two-dimensional area, such as a radiation field, where only proximity yields a non-zero reading.
Function 2:	Maximisation, a black box, or a mystery ML model, that takes two numbers as input and returns a log-likelihood score.
Function 3:	Maximisation, a drug discovery project, testing combinations of three compounds to create a new medicine
Function 4:	Maximisation, the challenge of optimally placing products across warehouses for a business with high online sales, where accurate calculations are costly and only feasible biweekly.
Function 5:	Maximisation, optimising a four-variable black-box function that represents the yield of a chemical process in a factory.
Function 6:	Maximisation, optimising a cake recipe using a black-box function with five ingredient inputs, for example flour, sugar, eggs, butter and milk.
Function 7:	Maximisation, optimising an ML model by tuning six hyperparameters, for example learning rate, regularisation strength or number of hidden layers.
Function 8:	Maximisation, optimising an eight-dimensional black-box function, where each of the eight input parameters affects the output, but the internal mechanics are unknown. 




# Challenge objectives

The objective for every function is maximisation. The constraint at hand is compute. We are able to submit one input candidate per function per week. This reflects real world constriants in terms of computational time or affordability, and hence, the strategy to reason about viable testing candidates is critical.




# Technical approach

The number of dimensions influence our approach to a certain degree. The higher the number of dimensions the more computationally expensive is the iteration, and hence, ingenuity about the domain and early iterations become critical in guiding future iterations.

In general, the initial iterations have been focused on exploration as opposed to exploitation. This is critical to explore the landscape at hand, and discovering unknown regions with high uncertainty, thereby, enabling a wider horizon exploration and exploitation.

The candidates for the respective functions were generated using Bayesian optimisation with acquisition techniques such as UCB and EI. Furthermore, the function domain guided our intuitions about the potential landscapes such as unimodal in the context of chemical yields. For the functions with four or less dimensions, UCB was deemed as a suitable starting algorithm considering it is relatively performant with low number of dimensions as we are able to cheaply explore the peaks and valleys. Whereas, for the functions with more than four dimensions, the ideal candidate would be EI since we could leverage clever mathematical techniques to avoid expensive hyper-dimensional landscapes searches, by using methods such as 'L-BFGS-B'.

SVM will be leveraged in future iterations when the focus becomes exploitation heavy. The purpose of SVM in this context is to qualify or rule-out potential candidates. The ideal is to "explore" in the early stages, and then "exploit" and qualify candidates using SVM. The intuition is that, after exploration, the landscape has been mapped out and SVM is able to effectively identify the boundaries between different peaks and valleys, and hence, with reasonable confidence, qualify or reject candidates for each function.

Finally, throughout the development process, the hyper-parameters such as exploration, alpha, noise, length-scales and the likes, are continuously fine-tined as per weekly outputs. The clearer the landscape becomes, the better guided our intutions are.


