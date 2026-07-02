# Datasheet 
General project-wide context (data source, licence, weekly-query constraint) is in [here](archive/DATASHEET.md); this file focuses on the per-function optimisation reasoning.

## Function 1: Contamination Detection

### Function overview
- Detecting likely contamination sources in a 2D area (e.g. a radiation field), where only proximity to the source yields a non-zero reading
- Two dimensions
- There are ten initial points (`initial_inputs.npy` shape `(10, 2)`), growing to ~23 after weekly additions
- The output represents a proximity/detection reading — effectively zero almost everywhere, with a sharp non-zero signal only very close to the source

### Nature of the data
- Initial `X` shape `(10, 2)`, `y` shape `(10,1)`, values in `[0, 1]`.
- One point was appended per week; early points explored broadly, later points converged tightly around `[0.72–0.73, 0.72–0.74]` once a promising region was found (several near-duplicate queries in that cluster)
- No formal repeat-query noise test was run, but three near-identical inputs (`[0.727272, 0.727272]` queried three times) returned near-identical outputs (~2.71e-14 each time), suggesting the function is close to deterministic/low-noise
- Extremely peaked and non-smooth: outputs span from ~1e-124 up to ~3e-6, i.e. over 100 orders of magnitude, so the landscape was treated as effectively a single sharp spike against a near-zero background — output values were log-sign-transformed (`sign(y) * log1p(1e80 * |y|)`) before fitting the GP so it could resolve any structure at all

### Your optimisation strategy
- Bayesian optimisation with a GP surrogate, both UCB and EI acquisition functions, plus an auxiliary SVM/logistic-regression/neural-network layer for candidate sanity-checking
- An RBF kernel was chosen for its smoothness, appropriate once the log-transform had turned the raw spike into a smoother log-scale landscape; the sharp original landscape motivated a tight `length_scale` (0.05) and 40 optimiser restarts to avoid missing the peak
- Very high exploration parameters were used (UCB `κ=100`, EI `ξ=100`) to counteract the risk of the GP collapsing onto the flat near-zero background; heatmap/3D-surface plots were used each week to visually decide where to explore next - however this is only relevant during the "exploration" phase - as the exploration parameter was adjusted accordingly during exploitation phases, in order of magnitudes
- Our iterations pivoted between exploration and exploitation, and naturally, as we learned more about the universe, we focused our efforts accordingly in terms of exploring new regions or exploiting high performance regions

### Data handling and preprocessing
- Inputs were already in `[0, 1]`, so no rescaling was needed; outputs were log-sign-transformed given their huge dynamic range 
- A GP surrogate model was fitted
- The kernel required a tight `length_scale_bounds=(1e-3, 0.2)` to avoid over-smoothing the sharp peak; a `ConvergenceWarning` showed the constant-kernel amplitude landing near its upper bound, suggesting that bound could be widened further
- No explicit outlier removal; the near-duplicate cluster around `[0.727, 0.727]` was kept as-is and used to sanity-check output stability rather than being treated as noise to discard

### Weekly iteration and learning
- Early broad samples showed the function was essentially zero everywhere except a narrow band; later points refined the exact location of that band
- The GP repeatedly suggested points clustered tightly in one region, a sign of converging on a local optimum; this was cross-checked visually via the heatmap/3D plots rather than a formal multimodality test
- The near-duplicate points in the `[0.72–0.73]` cluster were the most informative, confirming the peak's exact location and its (very low) sensitivity to small input changes
- With hindsight, during exploitation, I would have removed poor data points and simply focused on the high performance ones for more efficient "exploitation" that is less impacted by visual outliers nor noise

### Performance and results
- Best raw output found: **≈ 3.061 × 10⁻⁶**.
- At input **[0.686868, 0.696969]**.
- Low-to-moderate confidence: the extreme peakiness of this function (100+ orders of magnitude of dynamic range) means a much sharper peak could exist just outside the sampled cluster; the tight clustering of final queries covers only a small fraction of the 2D domain
- The observed behaviour matches the expected "sharp, near-zero-everywhere" contamination-detection landscape described in the problem brief

### Ethical, practical and general considerations
- Mirrors real sensor-based source-localisation problems (e.g. radiation or gas-leak detection), where most readings are uninformative and only proximity to the source matters
- As a synthetic function it has none of the real measurement noise, sensor drift, or physical obstruction effects a real detector would face - i.e. our pipeline is not immediately applicable to real-world cases 
- The exploration heuristics used (manual visual inspection of heatmaps) would not scale to a genuinely expensive real-world sensor sweep without more automation and rigorous mathematical pre-processing such as numerical scale and outlier elimination
- A future user should be aware that outputs span many orders of magnitude — any downstream use must reproduce the log-transform, or risk the GP being dominated by near-zero noise

---

## Function 2: Mystery Model Log-Likelihood

### Function overview
- A black-box "mystery" ML model that takes two numbers as input and returns a log-likelihood score
- Two dimensions
- 10 initial points, growing to ~23 after weekly additions
- A log-likelihood score (unbounded real number, higher is better)

### Nature of the data
- Initial `X` shape `(10, 2)`, `y` shape `(10, 1)`
- One point appended per week; early queries were spread out, then concentrated around two apparent peaks once early results suggested a bimodal landscape
- No evidence of noisy
- Appears **bimodal**: an explicit attempt at lowering `kappa` (5→0.5) and the kernel `length_scale` (0.1→0.05) specifically "to investigate whether there is a connected ridge between the two peaks" — i.e. two separate high-likelihood regions were suspected from the heatmap

### Your optimisation strategy
- GP-based Bayesian optimisation with both UCB and EI, plus an auxiliary NN classifier for saliency.
- A Matérn (ν=1.5) kernel was chosen as a middle ground between smooth and rugged, appropriate for a 2D function that looked like it had two distinct peaks rather than one smooth mode
- UCB (`κ=10`, later reduced) and EI (`ξ=0.001`) were both tried; UCB pushed toward a domain corner (`[0,0]`) and EI toward the opposite corner (`[1,1]`), disagreeing sharply — a sign the surrogate was still uncertain about which peak was better - hence, more data points needed
- Exploration was dialled down (`κ` 5→0.5, shorter length scale) once the two-peak hypothesis formed, to probe the ridge between them rather than keep exploring blindly

### Data handling and preprocessing
- No rescaling needed (already `[0, 1]`); no output transform applied, the function's output range is modest
- A GP model was fitted
- `length_scale=0.1` with bounds `(0.01, 1.0)`; no automatic optimiser restarts were configured, so kernel fitting relied on a single optimisation run each week - the heatmap and 3D surface plots sufficed in manually confining the bounds
- No outliers removed; all weekly points were kept and used directly - intentionally kept to provide more data points for the underlying surrogate model to improve variance and uncertainity modelling

### Weekly iteration and learning
- Early points suggested a single broad region of high likelihood; later points revealed what looked like two separate peaks with a possible connecting ridge
- The UCB/EI disagreement (corner `[0,0]` vs `[1,1]`) was the main signal of a multimodal landscape / competing local optima.
- The point near `[0.703, 0.927]` (yielding the best result, 0.6112) and the boundary-seeking suggestions were most informative, the former for exploitation and the latter for revealing the surrogate's residual uncertainty
- Configuring `n_restarts_optimizer` on the GP from the start would likely have produced a more stable kernel fit and reduced the corner-seeking behaviour of the acquisition functions.

### Performance and results
- Best output found: **0.6112052157614438**.
- At input **[0.702637, 0.926564]**.
- Moderate confidence — the UCB/EI disagreement suggests the surrogate had not fully converged on a single global mode by the end of the observed queries 
- Our observations are broadly consistent with expectations for a "mystery model" log-likelihood surface — non-trivial, with more than one region of interest

### Ethical, practical and general considerations
- Mirrors real hyperparameter/likelihood-search problems where the true model internals are unknown and only a score is observed
- As a synthetic function, it lacks the computational cost and variance a real model's log-likelihood evaluation would carry
- The manual kappa/length-scale retuning approach would need to be automated (e.g. via a scheduled decay) to scale to a longer or more expensive campaign
- A future user should note the acquisition functions actively disagreed at points during this run — treating any single suggested point as definitively "best" without checking both UCB and EI would be risky

---

## Function 3: Drug Discovery (3 Compounds)

### Function overview
- A drug-discovery task testing combinations of three compounds to create a new medicine
- Three dimensions
- 15 initial points, growing to ~26 after weekly additions
- Output represents a (negatively-scaled) yield/efficacy score — values are negative throughout, with values closer to zero indicating a better outcome

### Nature of the data
- Initial `X` shape `(15, 3)`, `y` shape `(15,1)`
- One point per week; queries converged strongly on a specific region (`[0.851, 0.612, 0.505]`), which was suggested and re-queried multiple times
- No explicit repeated-query experiment, but the same input `[0.851221, 0.611593, 0.504938]` was queried three times with near-identical outputs (~-0.0047 to -0.014), suggesting low observation noise around that point
- Multimodal — with UCB and EI both initially struggling and repeatedly suggesting unexplored corners before the search was manually restricted to bounds around the known peaks

### Your optimisation strategy
- GP-based Bayesian optimisation using both EI and UCB plus an auxiliary NN for saliency
- Matérn (ν=1.5) plus a `WhiteKernel` for noise was chosen as "a good balance between smooth and rugged" and to explicitly separate observation noise from landscape shape, appropriate given the noted multimodality
- EI used an unusually large `ξ=10` and UCB used `κ=10`, later raised to `κ=100` "to find peaks" — both biased heavily toward exploration; a 40-random-restart `L-BFGS-B` search was used to avoid missing distant peaks in 3D - the exploration parametersd were reduced by orders of magnitude during exploitation
- After both acquisition functions kept proposing unexplored corners of the domain, the team switched to (nominally) localising the search within the known high performing regions

### Data handling and preprocessing
- No rescaling (inputs already `[0, 1]`); no output transform, since the output range here is modest (roughly `[-0.5, 0]`) and doesn't need log-scaling
- A GP surrogate model was fitted
- `length_scale=[1,1,1]` with bounds `(1e-2, 1e2)`; a `ConvergenceWarning` showed one dimension's length scale landing near its upper bound, suggesting the bound could be widened
- The repeated exact-duplicate query `[0.851221, 0.611593, 0.504938]` was kept rather than treated as an error — it was useful confirmation that the surrogate's best-known point was stable and ruling out noise/non-determinism

### Weekly iteration and learning
- Early points were spread across the cube; later points converged strongly on one peak once it was identified as the best of the three known regions
- The repeated corner-seeking by UCB/EI early on was the main sign of multiple local optima competing for the acquisition function's attention
- The point `[0.851221, 0.611593, 0.504938]`, queried three times, was the most informative for confirming stability of the best-found region - but equally a "red-flag" worth investigating why it was repeatedly proposed by the model - or human error
- With hindsight, actually coding the "localised bounds" mentioned in the iteration-3 comment (rather than leaving them at the full domain) would likely have produced a more efficient search in later weeks - or perhaps, leveraging ARD thereby enabling automation for boundary confinement

### Performance and results
- Best output found: **-0.004686647016489443** (closest to zero of all observed values)
- At input **[0.851221, 0.611593, 0.504938]**
- Reasonably confident locally — the point was re-queried three times with consistent results — but only one of the three known peaks was ever fully exploited, so the global optimum across all three peaks is not confirmed
- Consistent with expectations: a drug-combination search typically has several locally good combinations (the "three peaks"), and the team successfully found and confirmed one of them

### Ethical, practical and general considerations
- Directly mirrors real combinatorial drug-discovery screening, where each combination test is expensive and only a handful can be run per cycle
- The synthetic function has none of the real biological variability, toxicity risk, or assay noise a genuine drug trial would carry
- The approach (GP + EI/UCB with periodic manual bound restriction) would scale reasonably to more expensive real assays, provided the "localise near known peaks" step is actually implemented in code rather than left as a comment - with equal input from domain expertise
- A future user should note that only one of the three identified peaks was ever exploited in depth — the other two remain largely unexplored and might contain a better combination

---

## Function 4: Warehouse Placement

### Function overview
- Optimally placing products across warehouses for a business with high online sales, where accurate placement calculations are costly and only feasible biweekly
- Four dimensions
- 30 initial points, growing to 43 after weekly additions
- Output represents a placement-efficiency/cost score (mixed performance — mostly negative "cost" with a smaller number of positive "efficiency" results)

### Nature of the data
- Initial `X` shape `(30, 4)`, `y` shape `(30,1)`1
- One point per week; queries moved from wide exploration (`κ=50` UCB over a semi-restricted `[0.2, 0.5]` box) to a tightly localised EI search (±0.2 box) once a positive-yield region near `[0.41, 0.41, 0.35, 0.42]` was found - the exploration parameters were reduced by order of magnitudes during exploitation
- No explicit repeat-query test; several near-duplicate points (e.g. `[0.413140,...]`, `[0.413152,...]`, `[0.425258,...]`) around the best region returned consistently positive but slightly varying results (0.66, 0.55, 0.49), consistent with mild observation noise or genuine local sensitivity
- Mostly smooth and unimodal within the explored region: once the positive-yield pocket around `[0.41, 0.41, 0.35, 0.42]` was found, nearby points stayed positive, while points elsewhere in the domain were strongly negative (down to -32.6), suggesting one good region surrounded by a much larger poorer region

### Your optimisation strategy
- GP-based Bayesian optimisation, UCB and EI, plus an auxiliary NN for saliency
- Matérn (ν=2.5), noted in-notebook as "robust for high-D optimisation," was chosen given the 4D input space, more flexible than a smoother kernel for capturing the sharp transition between the good pocket and the poor surrounding region
- UCB (`κ=50`) was used first over a wide `[0.2, 0.5]⁴` box to explore; once a good region was found, the team switched explicitly to EI with a tight ±0.2 local box around the current best, favouring exploitation - with orders of magnitude smaller exploration parameter
- The strategy explicitly shifted from UCB to EI - to narrow in on the current peak once enough of the landscape had been explored

### Data handling and preprocessing
- No rescaling (inputs already `[0, 1]`); no output transform (range is moderate, roughly `[-33, 1]`)
- A GP surrogate model was fitted
- `length_scale=[0.1]*4` with bounds `(0.5, 2.0)`; a `ConvergenceWarning` showed one dimension's length scale near its upper bound
- No formal outlier handling

### Weekly iteration and learning
- Early wide exploration revealed the domain was mostly poor (negative), sharpening the search toward the one promising pocket found early on
- A rank table of all observed values (`np.argsort(y)`) was used explicitly to identify and track the best/worst performing points each week
- The cluster of near-duplicate points around `[0.41, 0.41, 0.35, 0.42]` was most informative, confirming that region as a genuine local optimum rather than a lucky single sample
- With hindsight, introduce genetic cross-over earlier in the iterations and gradully move towards the gradients of highest descent

### Performance and results
- Best output found: **0.6576375369770244**
- At input **[0.413140, 0.406530, 0.351979, 0.424546]**
- Reasonably confident within the explored pocket (multiple nearby points confirm it), but the wide, mostly-negative rest of the 4D domain was only sparsely sampled, so a better pocket elsewhere cannot be ruled out
- Consistent with the described "costly, infrequent placement calculation" domain — most placements are inefficient, and the search successfully zeroed in on one efficient configuration

### Ethical, practical and general considerations
- Mirrors real warehouse/inventory placement optimisation, where each placement evaluation is a genuinely expensive operational-research computation
- As a synthetic function it lacks real supply-chain constraints (capacity limits, geography, lead times) that would shape a genuine warehouse-placement search
- The exploration-then-exploitation switching strategy used here would scale reasonably well, but the manual data-entry step (typing new results into a Python list) is a real risk at scale and should be replaced with a validated data pipeline
- Attempt dimensionality reduction for 2D visualisation might help confine the high performing region

---

## Function 5: Chemical Process Yield

### Function overview
- Optimising a four-variable black-box function representing the yield of a chemical process in a factory
- Four dimensions
- 20 initial points, growing to ~33 after weekly additions
- Output represents a chemical yield (large positive number, unbounded above)

### Nature of the data
- Initial `X` shape `(20, 4)`, `y` shape `(20,1)`
- One point per week; queries increasingly favoured the high corner of the domain (e.g. `[0.99, 0.99, 0.99, 0.99]`, `[1, 0, 1, 1]`) as yields kept increasing there
- No formal repeat-query test, but the exact same input `[0.000001, 0.000001, 0.999999, 0.999999]` was queried twice and returned an identical yield (1616.63) both times, indicating low or zero observation noise
- Highly **skewed/monotonic-looking** toward one corner of the domain: yields span from ~0.1 to several thousand (over four orders of magnitude), and nearly every high-yield point has most or all coordinates near 1 — consistent with a landscape that increases toward a domain corner rather than having an interior peak

### Your optimisation strategy
- GP-based Bayesian optimisation with EI and UCB, plus an auxiliary NN for saliency
- Matérn (ν=2.5) with per-dimension length scales was chosen for its flexibility in 4D; the output's huge dynamic range motivated a `log1p` transform plus `normalize_y=True` before fitting, so the GP wasn't dominated by the largest few yields
- EI used an unusually large `ξ=101` (in log-space) and UCB used `κ` ranging from 2.576 (99% CI) up to 5.0 "to find peaks" — both strongly exploration-biased, consistent with the still-rising yields near the domain boundary - the exploration parameters were reduced by order of magnitudes when focusing on exploitation of high performance regions
- Pivoted from a single-start EI search to random-restart UCB search (10 starts) to help guide exploration of other promising regions since the underlying surrogate model greedly suggested a single direction of exploration

### Data handling and preprocessing
- No input rescaling (already `[0, 1]`); outputs were `log1p`-transformed given the multi-order-of-magnitude range, and further standardised via `normalize_y=True` in the GP
- A GP model was fitted
- `length_scale=[0.1]*4` initial guess with bounds `(0.01, 10.0)`; after fitting, two of the four length scales hit the upper bound (10.0), triggering a `ConvergenceWarning` and suggesting the bound needed widening further
- No points were removed; the repeated identical query was kept as a useful noise sanity-check

### Weekly iteration and learning
- Each new high-value query near the domain corner reinforced the hypothesis that yield increases with all four chemical parameters, shifting the strategy from broad exploration to corner-probing
- No clear separate local optimum was found — the landscape looked closer to monotonic-toward-a-corner than genuinely multimodal, so "local optima" in the usual sense were not really encountered
- Boundary/corner points (`[0.99]*4`, `[1,0,1,1]`) were the most informative, each confirming or refining the direction of increasing yield.
4. Use dimensionality reduction and leverage visualisation to confirm priori understanding of unimodal behaviour, and a deeper understanding of the sensitive dimensions for stricter confinement of search boundaries

### Performance and results
- Best output value in the final dataset: **8662.405001248297** 
- At input **[0.960914, 0.990000, 0.990000, 0.990000]**
- Low-to-moderate confidence that this is the global maximum: the landscape appears to still be increasing toward the `[1,1,1,1]` corner, which was never directly queried, so a higher yield may exist right at or beyond the domain boundary
- Consistent with expectations for a chemical yield function where more of each reagent generally helps, up to some real-world limit not captured by the `[0,1]` bounds - but ideally guided by domain expert

### Ethical, practical and general considerations
- Mirrors real chemical-process yield optimisation, where each run is costly (time, reagents, safety) and only a few experiments can be run per cycle
- The synthetic function's yields are unbounded within `[0,1]⁴`, whereas a real chemical process would have physical/safety limits well before reaching "all reagents at maximum."
- This strategy (log-transform + GP + high-exploration EI/UCB) would scale reasonably to a genuinely expensive process, but the corner-seeking behaviour is a real risk in practice — a real process pushed to its parameter extremes could be unsafe or infeasible, unlike this synthetic domain
- The apparent optimum sits right at the edge of the sampled domain - so it may not be a true interior optimum but a mathematical anomaly, hence, domain expertise is required

---

## Function 6: Cake Recipe

### Function overview
- Optimising a cake recipe using a black-box function with five ingredient inputs (e.g. flour, sugar, eggs, butter, milk)
- Five dimensions
- 20 initial points, growing to ~33 after weekly additions
- Output represents a recipe quality score; values in this dataset are consistently negative, with more negative implying better taste!

### Nature of the data
- Initial `X` shape `(20, 5)`, `y` shape `(20,1)`
- One point per week; two near-duplicate points (`[0.484049,…]` and `[0.484318,…]`) were queried close together and returned very similar scores (-1.79 and -1.79)
- No formal repeat test, but the two near-duplicate points above returned consistent results, suggesting low observation noise
- Appears **rugged/canyon-like** rather than smooth: 3 of 5 fitted length scales saturated at their upper bound (10.0), and it is likely two length scales are responsible for the steep descents

### Your optimisation strategy
- GP-based Bayesian optimisation with EI and UCB, plus an auxiliary NN for saliency
- Matérn (ν=2.5) was chosen for its flexibility in a 5D, apparently rugged landscape
- EI (`ξ=10`) and UCB (`κ=10`) were both used; EI's acquisition surface was searched via 50,000 random samples (seeded) refined with `L-BFGS-B`, a heavier search than other functions, reflecting the higher dimensionality and rugged canyon shape - the exploration parameters were reduced by orders of magnitude when pivoting into exploitation mode
- A mid-project bug fix, **sign-convention bug** — "we were previously maximising, i.e. making the negative less negative as opposed to making the negative more negative", the acquisition logic had been pushing the score in the wrong direction relative to the intended (maximisation) objective

### Data handling and preprocessing
- No input rescaling; no output transform (unlike Functions 1/5, the range here, roughly `[-3.2, -0.5]`, doesn't need log-scaling)
- A GP model was fitted
- `length_scale=[0.5]*5` with bounds `(0.1, 10)`; 3 of 5 dimensions hit the upper bound, explicitly flagged via a custom warning print recommending wider bounds — these warnings were not acted upon because input was limited to [0,1]
- No points removed; the near-duplicate/repeated points across weeks were kept as-is to assert confidence in the sensitive features

### Weekly iteration and learning
- New points confirmed the "canyon" shape — most of the 5D space scored similarly poorly, with a narrower band of relatively better recipes
- Yes, indirectly: the acquisition function repeatedly suggesting all-ingredients-required combinations, and more varied data-point are needed to rule out other valleys
- The point `[0.009710, 0.979677, 0.029675, 0.011367, 0.857560]` — the best (least-bad) score under the notebook's own `min(y)` framing — was the most informative single point, repeatedly used as a start for subsequent local searches
- If restarted, the sign-convention check would be done first, before any acquisition tuning, to avoid an entire round of the project potentially optimising in the wrong direction 

### Performance and results
- Best output value: **-3.1365278755680133** 
- Best (maximisation) input: **[0.009710, 0.979677, 0.029675, 0.011367, 0.857560]**
- Low confidence either way: given the documented sign-convention confusion, and 3 of 5 length scales sitting at their bound (suggesting the kernel may still be over-smoothing a genuinely rugged landscape)
- Partially — the "canyon" shape and the comment "all ingredients are required" are broadly consistent with a recipe-style function where extreme (near-zero or near-one) ingredient amounts tend to fail, but the objective-direction confusion means the specific numeric results should be treated cautiously

### Ethical, practical and general considerations
- Mirrors real recipe/formulation optimisation, where the "cost" of each trial is baking and evaluating an actual cake
- As a synthetic function, it has none of the sensory subjectivity or real ingredient-interaction chemistry a genuine recipe-optimisation task would involve
- This strategy would need a resolved, tested acquisition-direction sign convention before scaling to a more expensive real process — the bug found here is exactly the kind of silent error that would be very costly in a real, expensive experiment
- A future user must double-check whether this notebook is minimising or maximising `y` before trusting any of its "best point" output — the notebook itself documents catching this ambiguity

---

## Function 7: ML Hyperparameter Tuning

### Function overview
- Optimising an ML model by tuning six hyperparameters (e.g. learning rate, regularisation strength, number of hidden layers)
- Six dimensions
- 30 initial points, growing to ~43 after weekly additions
- Output represents a model performance score (higher is better)

### Nature of the data
- Initial `X` shape `(30, 6)`, `y` shape `(30,1)`
- One point per week; the saved chart (`function_7_chart_9th_iteration.png`) shows performance bouncing between roughly 0 and 2 up to 9th week of submissions without a clear monotonic trend
- No formal repeat-query test; no exact-duplicate inputs were queried, so noise could not be directly assessed from this data
- Appears **noisy/multimodal**: most sampled points score near zero, with a small number of much higher-scoring points (up to ~2.0) scattered non-contiguously — typical of a hyperparameter-tuning landscape where most configurations are mediocre and a few "get lucky"

### Your optimisation strategy
- GP-based Bayesian optimisation with both EI and UCB, plus an auxiliary NN for saliency
- Matérn (ν=2.5) plus a `WhiteKernel` was chosen to explicitly separate real landscape structure from the noisy, spiky pattern of scores typical of hyperparameter search
- EI (`ξ=0.1`) was optimised via a **Sobol-sequence** multi-start (32 starts, chosen as a power of 2 for Sobol efficiency) plus `L-BFGS-B` refinement, needed because a dense grid search is infeasible in 6D; UCB (`κ=10`) was also tried, from a single start at the current best point - in general, the exploration parameters were adjusted by orders of magnitude depending on optimisation focus, exploitation or exploration
4. After a series of consecutive exploitation we pivoted back into exploration since returns were deemed as diminished, the change in focus did not result in tangible gains

### Data handling and preprocessing
- No input rescaling (hyperparameters already normalised to `[0,1]`); no output transform (range is modest, roughly `[0, 2]`)
- A GP model was fitted
- `length_scale=[0.5]*6` with bounds `(0.01, 10.0)`; after fitting, one dimension's length scale hit the upper bound, flagged via `ConvergenceWarning`
- No explicit outlier handling, the landscape appears quite rugged between low and moderately performing regions

### Weekly iteration and learning
- Early broad sampling revealed most of the 6D space scores poorly; later Sobol-seeded EI search located a smaller region scoring consistently higher (~1.3–2.0)
- The scattered, non-contiguous pattern of high-scoring points (visible in the saved iteration chart) is consistent with several competing local optima rather than one smooth peak
- The higher-scoring points (indices with values 1.3–2.0) were most informative for narrowing the search; low/moderate points mainly served to rule out large parts of the domain
- If restarted, running the Sobol multi-start search from the very first iteration (rather than starting with denser grid-style sampling) would likely have found the productive region faster

### Performance and results
- Best output found: **1.9993628649406798**
- At input **[0.000001, 0.237979, 0.404631, 0.121878, 0.334399, 0.759561]**
- Moderate confidence: the Sobol multi-start (32 starts) gives reasonable coverage of the 6D acquisition surface, but the scattered nature of high-scoring points suggests other undiscovered good regions may remain
- Consistent with expectations for hyperparameter tuning — most configurations underperform, with a few standout combinations, matching the observed distribution

### Ethical, practical and general considerations
- Mirrors real (and increasingly common) automated hyperparameter tuning, where each "evaluation" is a full model training run
- The synthetic function skips the real computational cost, training instability, and stochasticity a genuine model-training evaluation would have
- This strategy (Sobol multi-start + GP-EI/UCB) is a standard, scalable approach and would transfer well to genuinely expensive hyperparameter searches, more so than the ad hoc manual approaches used on lower-dimensional functions in this project
- A future user should note the scattered, non-contiguous pattern of good results — assuming a single smooth "peak" exists here (as might be reasonable for Functions 1–4) would be misleading

---

## Function 8: Unknown 8D Function

### Function overview
- An eight-dimensional black-box function where each of the eight input parameters affects the output, but the internal mechanics are unknown
- Eight dimensions
- 40 initial points, growing to ~53 after weekly additions
- Output represents an unlabelled performance score (higher is better), observed to range roughly between 5.6 and 9.9

### Nature of the data
- Initial `X` shape `(40, 8)`, `y` shape `(40,1)`.
- One point per week; queries stayed within a moderate output range throughout (no extreme outliers), suggesting the initial sampling had already covered a reasonably representative part of the landscape
- No formal repeat-query test; no exact-duplicate inputs were found in the appended data, so noise could not be directly assessed
- Given the relatively tight output range (5.6–9.9, less than one order of magnitude) compared to Functions 1 and 5, this landscape looks **comparatively smooth**, though genuinely visualising an 8D surface isn't possible — this assessment relies on the GP's ARD length scales (see below) and the auxiliary NN's saliency scores rather than a direct plot

### Your optimisation strategy
- GP-based Bayesian optimisation with EI and UCB, plus an auxiliary NN for saliency
- Matérn (ν=2.5) with independent (ARD) length scales per dimension was chosen specifically "to learn which inputs are actually driving the 9.90 output" — i.e. to use the kernel fit itself as a feature-importance tool in a dimensionality too high to visualise directly
- EI used a deliberately high `ξ=1.0` ("high exploration parameter for 8D... forces the model to look for points that could beat 9.90 by a significant margin"), optimised via **Sobol-sequence multi-start** with 64 starts (doubled from Function 7's 32, "increased for 8D complexity") plus `L-BFGS-B`; UCB (`κ=10`) was also tried from a single start - the exploration parameters were adjusted accordingly for exploration or exploitation
- After the first kernel fit showed one length scale saturating at its upper bound (10.0), the team explicitly retightened the bounds (`length_scale_bounds` from `(0.01, 10)` to `(0.01, 1.0)`, noise bounds tightened to `(1e-6, 1e-4)`) to force the model to acknowledge there is 'detail' in the landscape

### Data handling and preprocessing
- No input rescaling; no output transform (the range, roughly `[5.6, 9.9]`, doesn't need log-scaling unlike Functions 1/5)
- A GP model was fitted
- Beyond the retightened length-scale and noise bounds, the fitted ARD length scales themselves (ranging 2.38–10.0 across the 8 dimensions) were inspected explicitly to identify which of the 8 inputs mattered most
- No outliers removed; all weekly points were kept

### Weekly iteration and learning
- The tight output range across all sampled points suggested no single input dominates the outcome as strongly as in Functions 1 or 5; the ARD length-scale check helped narrow down which of the 8 inputs mattered most
- No clear separate local optima were identified — the relatively narrow output range and the decision to retighten kernel bounds meant we focused on landscape detail to find promising peaks
- Points near the eventual best value (9.90+) were most informative; the retightened kernel bounds were specifically motivated by wanting the model to distinguish between these closely-scored top points
- If restarted, using the ARD ("which inputs matter") analysis from the very first iteration — rather than after the initial default-bounds fit already suggested it — would have focused the Sobol search on the important dimensions sooner, and incorporating dimensionality reduction to help assert nature of performance such as multi/uni modal etc

### Performance and results
- Best output value in the final dataset: **9.9122133391516**
- At input **[0.000001, 0.000001, 0.628520, 0.000001, 0.999999, 0.999999, 0.000001, 0.000001]**
- Low confidence: 64-start Sobol search gives decent coverage for 8D but 8D is a large space to cover with ~50 total observations
- The observation is broadly consistent with a generic "all 8 inputs matter, no obvious single dominant dimension" black-box function

### Ethical, practical and general considerations
- Mirrors real high-dimensional black-box optimisation problems (e.g. multi-parameter engineering or process tuning) where no single variable's effect is obvious in advance
- As a synthetic function, it lacks real 8D physical constraints or interaction effects that would make a genuine high-dimensional process much harder to reason about than this dataset suggests
- The Sobol-multi-start + ARD-kernel approach used here is a genuinely scalable pattern and is the most "production-ready" strategy in this project — it would extend reasonably well to more expensive, higher-dimensional real problems
- A future user should treat the exact "best input" for the top value with caution, since 8D coverage with ~50 points is inherently sparse
