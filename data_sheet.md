# Datasheet template

This datasheet helps you document your optimisation decisions, learning and reasoning for every function in the black-box optimisation project. 

Provide concise, reflective answers. Bullet points are acceptable unless otherwise specified.

## Function overview

Describe the specific black-box function you are optimising.

1. Which function does this datasheet describe? (Function 1Ð8)
State the function number and name.

2. What real-world scenario does this function simulate? 
Summarise the domain (e.g., contamination detection, chemical yield).

3. What is the dimensionality of the input? 
E.g., 2D, 3D, 5D, etc.

4. How many initial data points were provided? 
Refer to the dataset shape.

5. What does the output represent? 
E.g., yield, adverse reaction severity, performance score, etc.
 
## Nature of the data

Describe how the dataset is structured and evolves across iterations.

1. Describe the structure of the initial dataset. 
State the shapes of input and output arrays.

2. How does the dataset evolve as you add new queries weekly?
Mention the number of new points, the exploration pattern, and shifts in sampling.

3. Does the function include noise or randomness? 
Explain if repeated evaluations give different results and how this affected your strategy.

4. Based on observations, does the function appear unimodal, multimodal, noisy, or smooth? 
State your reasoning (plots, surrogate behaviour, GP variance, etc.).

## Your optimisation strategy

Explain the method you designed.

1. Which optimisation method(s) did you use? 
Random search, grid search, Bayesian optimisation (GP/EI/UCB), manual reasoning, etc.

2. Why did you choose this method for this particular function? 
Tie your reasoning to noise level, dimensionality, local optima, etc.

3. How did you balance exploration and exploitation? 
Mention acquisition functions, search heuristics or heuristics.

4. Did your strategy change over the weeks? Why? 
Describe adaptations due to insights or failures.

## Data handling and preprocessing

Explain how you prepared data for modelling or decision-making.

1. Did you rescale or normalise inputs? Why or why not?

2. Did you train any surrogate models? 
GP, regression, tree-based model, neural model, etc.

3. If yes, what preprocessing did the surrogate require? 
Kernel choice, encoding choices, noise modelling and hyperparameter tuning.

4. Did you handle outliers or unusual data points? 
Explain your criteria and actions taken. 

## Weekly iteration and learning

Reflect on learning over time.

1. How did new data points change your understanding of the function landscape?

2. Did you encounter local optima? How did you detect them?

3. Which queried inputs were most informative and why? 
E.g., boundary points, uncertain regions, high-gradient regions.

4. If you restarted, what would you do differently? 
Strategy, heuristics, exploration schedule, model choice.

## Performance and results

Summarise your optimisation outcomes.

1. What is the best output value you achieved?

2. Which input vector produced this value?

3. How confident are you that this is near the global maximum? Why?
Refer to variance, stability, surrogate predictions, and exploration coverage.

4. Did your results align with expectations for this function? Based on the problem description.

## Ethical, practical and general considerations

Reflect on the broader implications of the optimisation task.

1. How does this black-box optimisation task relate to real-world applications?

2. What limitations arise from the synthetic nature of the function?

3. Would your strategy scale to more serious or more expensive problems? Why or why not?

4. What risks or pitfalls should a future user be aware of when analysing this function?

