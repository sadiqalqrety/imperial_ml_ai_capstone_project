# Model overview
Model name: BBO Hyperparameter Bayesian Optimisation
Type: Gaussian Process
Version: 1.0.0
Task: Propose function input variables for optimal yields
Deployment: Used as part of Jupyter Notebook workflows
License: MIT
Developer: https://github.com/sadiqalqrety

# Intended use
Primary tasks: BBO hyperparameter bayesian optimisation for functions ranging from two to eight dimensions
Target users: Researchers, students, and machine learning enthusiasts
Use case: Functions whereby inputs are numerical floating point numbers between zero and one with six decimal place precision, and output is floating point numbers
Limitations: Not suitable for non-numerical data nor for input numbers less than zero or greater than one

# Training data
Size: The initial data was sourced from the educational course provider, Imperial Executive Education
Sources: Domain specific function data, as outlined in main [README](/README.md) provided by educational course provider, Imperial Executive Education
Features: Function feature inputs and outputs

# Inputs and outputs
Input: Floating point numbers precise to six decimal places greater than zero and less than one
Output: Floating point numbers, arbitrary yield numbers depicting performance (peak or valley)
Strategy: First four to five rounds were exploration focused, and the following four rounds were exploitation focused, and the final two rounds were exploration focused
Techniques: Bayesian Optimisation using Gaussian Process coupled with kernels such as RBF, and acquisiton functions such as UCB and EI

Assumptions
Constraints
Potential Failure Modes
Summaries - Results across functions

Evaluation - performance across different "functions" of same nature cooking/chemical

# Ethical considerations
Bias risks: Function 1 and 2 were subjected to manual input following visual inspection of heat-maps and 3D surface plot, this was intentionally to force explore unexplored regions
Transparency: The parameters such as exploration parameter and alpha were not tracked for previous rounds, and the notebooks simply reflect the latest round of iteration
Recommendation: Leverage a versioned workflow tool such as MLFlow to track configuration per round of iteration

Risk mitigation strategies
Risks

Reproducibility
Adaptation


# Distributions
Model card and code available in repository for scrutiny and improvement, under MIT license.


    
