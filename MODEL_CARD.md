# Model Card

## Description

**Input:** Each of the eight black-box functions takes a vector of floating-point numbers in `[0, 1]`, precise to six decimal places, with dimensionality specific to the function: 2D (Functions 1–2), 3D (Function 3), 4D (Function 4-5), 5D (Function 6), 6D (Function 7), 8D (Function 8). At each weekly step the surrogate model also consumes the full cumulative history of previously observed `(X, y)` pairs for that function, since the Gaussian Process is refit from scratch on all data seen so far before proposing the next candidate.

**Output:** For any candidate input, the GP surrogate outputs a predicted mean and standard deviation of the (unknown) objective value. These are combined by an acquisition function into a single score used to rank candidates. The practical deliverable each week is one recommended input vector — the single candidate to actually query against the real black-box function, given the one-query-per-function-per-week budget.

**Model Architecture:** A **Gaussian Process Regressor** is used as the surrogate for every function, with the kernel adapted per function.

On top of the GP, hand-implemented **Expected Improvement (EI)** and **Upper Confidence Bound (UCB)** acquisition functions score candidate points. The next point is found via dense grid search or a handful of `L-BFGS-B` restarts for the lower-dimensional functions, and via **Sobol-sequence multi-start optimisation** (32–64 starts) followed by `L-BFGS-B` refinement for the higher-dimensional functions (6D–8D), where grid search is infeasible. 

All models have an associated neural network (PyTorch feed-forward with hand-labelled points) at the end to "predict" the proposed candidates for high or low performance. This is an auxiliary optimisation of the pipeline and does not impact the acquisition strategy directly.

## Performance

Because these are genuine black-box functions with no known ground truth optimum, performance is measured empirically as the best objective value achieved (`max(y)`, or "closest to zero" for the negatively-scaled Function 3) across the initial dataset plus the thirteen weekly real-world queries recorded directly in each notebook.

A secondary, more model-centric performance signal comes from the GP fit diagnostics: for Functions 1, 3, 6, 7 and 8, `sklearn` raised `ConvergenceWarning`s because the optimised length-scale (or noise level) landed on its search bound, indicating the kernel bounds were mis-specified relative to the true landscape and needed manual widening or tightening between iterations — i.e. the model repeatedly signalled its own uncertainty about whether it had the right smoothness assumptions.

In the course cohort leaderboard, the model performance varied from best performing 9th place (Function 4) to worst performing 38th (Function 2). 

## Limitations

- **Very small sample sizes for the dimensionality involved.** Each GP is fit on roughly 20–40 observations total (10–30 initial + up to 13 weekly points), which is thin for reliably estimating a Matérn kernel with independent length scales per dimension in up to 8D — most visible in Function 8, where one of eight length scales repeatedly saturated at its upper bound
- **No held-out validation of the surrogate itself.** The GP is refit on all available history each week and used immediately to propose the next point; its predictive accuracy is never checked against a held-out set - so "performance" above reflects the objective values found, not how well the GP predicts unseen inputs
- **Manual, undocumented hyperparameter tuning.** `kappa`, `xi`, kernel type and length-scale bounds were adjusted week-to-week by hand based on visual inspection of heatmaps/surface plots, and the specific values used in earlier rounds were not logged — only the final round's settings remain in each notebook, so the process is not fully reproducible
- **Auxiliary NN saliency is illustrative only.** The PyTorch classifiers are trained on a handful of hand-labelled points per function and are not validated — their "importance scores" should not be read as statistically reliable feature importances

## Trade-offs

- **Kernel smoothness vs. flexibility.** RBF assumes a very smooth objective and can under-fit sharp, narrow peaks, whereas Matérn was used for the higher-dimensional functions since it is more flexible for rugged landscapes - but needs more data to fit reliably
- **Multi-start thoroughness vs. compute.** Sobol multi-start with 32–64 restarts in higher dimensional spaces provided better coverage of the acquisition surface than the single- or few-restart `L-BFGS-B` used on lower-dimensional functions but costs proportionally more compute per suggested point — a reasonable trade given evaluations of the *real* function are the actual bottleneck
- **Manual tuning vs. reproducibility.** Adjusting acquisition and kernel parameters by hand each week following review of function performance — valuable when data is this scarce — but it makes the process hard to reproduce or transfer to a new problem space

