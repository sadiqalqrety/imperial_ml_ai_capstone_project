# Model overview
Model name: BBO Hyperparameter Bayesian Optimisation 

Type: Gaussian Process

Version: 1.0.0

Task: Black-box hyperparmeter optimisation

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

Strategy: The strategy depends on your intended behaviour; exploration or exploitation, and this is dictated by the exploration parameter and choice of acquisiton function

Techniques: Bayesian Optimisation using Gaussian Process coupled with kernels such as RBF, and acquisiton functions such as UCB and EI

Assumptions: Inherent statisical assumptions embedded in Gaussian Process

Constraints: Current approach is only suitable for functions ranging from 2D to 8D

Potential Failure Modes: The underlying Gaussian Process has a tendency to explore universe boundaries and this can be mitigated by use of "local-boundaries" per dimension

Summaries - yield results for functions across the ten iterations exist in the respective function jupyter notebooks, see below Function 7 with nine iteration
![Function 7](/function_7_chart_9th_iteration.png)

Evaluation - Each function has a specific approach as per number of dimensions, for example, Sobol Sequence is only used in the functions with more than six dimensions, hence, inter-function evaluation is not suitable.

# Ethical considerations
Bias risks: Function 1 and 2 were subjected to manual input following visual inspection of heat-maps and 3D surface plot, this was intentionally to force explore unexplored regions

Transparency: The parameters such as exploration parameter and alpha were not tracked for previous rounds, and the notebooks simply reflect the latest round (10) of iteration

Recommendation: Leverage a versioned workflow tool such as MLFlow to track configuration per round of iteration

Risk mitigation strategies: Document all manual interventions

Risks: An experimental approach, with unstable numerical stability beyond six decimal places, and output yield ranges are varied - hence, a careful mathematical framework must be in place to ensure "numerical stability" across the functions

Reproducibility: Poor as previous workflow iterations were not documented

Adaptation: The model may be adapted for functions with similar dimensions and input restrictions


# Distributions
Model card and code available in repository for scrutiny and improvement, under MIT license.


    
