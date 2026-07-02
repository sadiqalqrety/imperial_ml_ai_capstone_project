# Datasheet 
General project-wide context (data source, licence, weekly-query constraint) is in [here](archive/DATASHEET.md); this file focuses on the per-function optimisation reasoning.

## Function 1: Contamination Detection

### Function overview
1. **Function 1.**
2. Detecting likely contamination sources in a 2D area (e.g. a radiation field), where only proximity to the source yields a non-zero reading.
3. 2D.
4. 10 initial points (`initial_inputs.npy` shape `(10, 2)`), growing to ~22 after weekly additions.
5. The output represents a proximity/detection reading — effectively zero almost everywhere, with a sharp non-zero signal only very close to the source.

### Nature of the data
1. Initial `X` shape `(10, 2)`, `y` shape `(10,)`, values in `[0, 1]`.
2. One point was appended per week; early points explored broadly, later points converged tightly around `[0.72–0.73, 0.72–0.74]` once a promising region was found (several near-duplicate queries in that cluster).
3. No formal repeat-query noise test was run, but three near-identical inputs (`[0.727272, 0.727272]` queried three times) returned near-identical outputs (~2.71e-14 each time), suggesting the function is close to deterministic/low-noise.
4. Extremely peaked and non-smooth: outputs span from ~1e-124 up to ~3e-6, i.e. over 100 orders of magnitude, so the landscape was treated as effectively a single sharp spike against a near-zero background — output values were log-sign-transformed (`sign(y) * log1p(1e80 * |y|)`) before fitting the GP so it could resolve any structure at all.

### Your optimisation strategy
1. Bayesian optimisation with a GP surrogate, both UCB and EI acquisition functions, plus an auxiliary SVM/logistic-regression/neural-network layer for candidate sanity-checking.
2. An RBF kernel was chosen for its smoothness, appropriate once the log-transform had turned the raw spike into a smoother log-scale landscape; the sharp original landscape motivated a tight `length_scale` (0.05) and 40 optimiser restarts to avoid missing the peak.
3. Very high exploration parameters were used (UCB `κ=100`, EI `ξ=100`) to counteract the risk of the GP collapsing onto the flat near-zero background; heatmap/3D-surface plots were used each week to visually decide where to explore next.
4. Yes — iteration 4's comment explicitly notes continuing to "explore more of the universe inspired by heatmap and 3D plot," i.e. exploration was guided by visualisation rather than a fixed schedule.

### Data handling and preprocessing
1. Inputs were already in `[0, 1]`, so no rescaling was needed; outputs were log-sign-transformed given their huge dynamic range (see above).
2. Yes — `GaussianProcessRegressor` with `Constant × RBF`, `alpha=0.1`, `n_restarts_optimizer=40`.
3. The kernel required a tight `length_scale_bounds=(1e-3, 0.2)` to avoid over-smoothing the sharp peak; a `ConvergenceWarning` showed the constant-kernel amplitude landing near its upper bound, suggesting that bound could be widened further.
4. No explicit outlier removal; the near-duplicate cluster around `[0.727, 0.727]` was kept as-is and used to sanity-check output stability rather than being treated as noise to discard.

### Weekly iteration and learning
1. Early broad samples showed the function was essentially zero everywhere except a narrow band; later points refined the exact location of that band.
2. The GP repeatedly suggested points clustered tightly in one region, a sign of converging on a local optimum; this was cross-checked visually via the heatmap/3D plots rather than a formal multimodality test.
3. The near-duplicate points in the `[0.72–0.73]` cluster were the most informative, confirming the peak's exact location and its (very low) sensitivity to small input changes.
4. With hindsight, the log-transform should have been applied from iteration 1 rather than partway through, since the untransformed GP fit was likely uninformative during the earliest exploration rounds.

### Performance and results
1. Best raw output found: **≈ 3.061 × 10⁻⁶**.
2. At input **[0.686868, 0.696969]**.
3. Low-to-moderate confidence: the extreme peakiness of this function (100+ orders of magnitude of dynamic range) means a much sharper peak could exist just outside the sampled cluster; the tight clustering of final queries covers only a small fraction of the 2D domain.
4. Yes — this matches the expected "sharp, near-zero-everywhere" contamination-detection landscape described in the problem brief.

### Ethical, practical and general considerations
1. Mirrors real sensor-based source-localisation problems (e.g. radiation or gas-leak detection), where most readings are uninformative and only proximity to the source matters.
2. As a synthetic function it has none of the real measurement noise, sensor drift, or physical obstruction effects a real detector would face.
3. The exploration heuristics used (manual visual inspection of heatmaps) would not scale to a genuinely expensive real-world sensor sweep without more automation.
4. A future user should be aware that outputs span many orders of magnitude — any downstream use must reproduce the log-transform, or risk the GP being dominated by near-zero noise.

---

## Function 2: Mystery Model Log-Likelihood

### Function overview
1. **Function 2.**
2. A black-box "mystery" ML model that takes two numbers as input and returns a log-likelihood score.
3. 2D.
4. 10 initial points, growing to ~22–23 after weekly additions.
5. A log-likelihood score (unbounded real number, higher is better).

### Nature of the data
1. Initial `X` shape `(10, 2)`, `y` shape `(10,)`.
2. One point appended per week; early queries were spread out, then concentrated around two apparent peaks once early results suggested a bimodal landscape.
3. No explicit repeat-query test; the function's outputs are smooth relative to Function 1, and no evidence of noisy/inconsistent repeats was found.
4. Appears **bimodal**: an explicit iteration-3 note records lowering `kappa` (5→0.5) and the kernel `length_scale` (0.1→0.05) specifically "to investigate whether there is a connected ridge between the two peaks" — i.e. two separate high-likelihood regions were suspected from the heatmap.

### Your optimisation strategy
1. GP-based Bayesian optimisation with both UCB and EI, plus an auxiliary NN classifier for saliency.
2. A Matérn (ν=1.5) kernel was chosen as a middle ground between smooth and rugged, appropriate for a 2D function that looked like it had two distinct peaks rather than one smooth mode.
3. UCB (`κ=10`, later reduced) and EI (`ξ=0.001`) were both tried; UCB pushed toward a domain corner (`[0,0]`) and EI toward the opposite corner (`[1,1]`), disagreeing sharply — a sign the surrogate was still uncertain about which peak was better.
4. Yes — exploration was dialled down (`κ` 5→0.5, shorter length scale) once the two-peak hypothesis formed, to probe the ridge between them rather than keep exploring blindly.

### Data handling and preprocessing
1. No rescaling needed (already `[0, 1]`); no output transform applied (the function's output range is modest, unlike Function 1).
2. Yes — `GaussianProcessRegressor`, `Constant × Matérn(ν=1.5)`, `alpha=0.1`.
3. `length_scale=0.1` with bounds `(0.01, 1.0)`; no automatic optimiser restarts were configured, so kernel fitting relied on a single optimisation run each week.
4. No outliers removed; all weekly points were kept and used directly.

### Weekly iteration and learning
1. Early points suggested a single broad region of high likelihood; later points revealed what looked like two separate peaks with a possible connecting ridge.
2. Yes — the UCB/EI disagreement (corner `[0,0]` vs `[1,1]`) was the main signal of a multimodal landscape / competing local optima.
3. The point near `[0.703, 0.927]` (yielding the best result, 0.6112) and the boundary-seeking suggestions were most informative, the former for exploitation and the latter for revealing the surrogate's residual uncertainty.
4. Configuring `n_restarts_optimizer` on the GP from the start would likely have produced a more stable kernel fit and reduced the corner-seeking behaviour of the acquisition functions.

### Performance and results
1. Best output found: **0.6112052157614438**.
2. At input **[0.702637, 0.926564]**.
3. Moderate confidence — the UCB/EI disagreement suggests the surrogate had not fully converged on a single global mode by the end of the observed queries.
4. Broadly consistent with expectations for a "mystery model" log-likelihood surface — non-trivial, with more than one region of interest.

### Ethical, practical and general considerations
1. Mirrors real hyperparameter/likelihood-search problems where the true model internals are unknown and only a score is observed.
2. As a synthetic function, it lacks the computational cost and variance a real model's log-likelihood evaluation would carry.
3. The manual kappa/length-scale retuning approach would need to be automated (e.g. via a scheduled decay) to scale to a longer or more expensive campaign.
4. A future user should note the acquisition functions actively disagreed at points during this run — treating any single suggested point as definitively "best" without checking both UCB and EI would be risky.

---

## Function 3: Drug Discovery (3 Compounds)

### Function overview
1. **Function 3.**
2. A drug-discovery task testing combinations of three compounds to create a new medicine.
3. 3D.
4. 15 initial points, growing to ~26 after weekly additions.
5. Output represents a (negatively-scaled) yield/efficacy score — values are negative throughout, with values closer to zero indicating a better outcome.

### Nature of the data
1. Initial `X` shape `(15, 3)`, `y` shape `(15,)`.
2. One point per week; queries converged strongly on a specific region (`[0.851, 0.612, 0.505]`), which was suggested and re-queried multiple times.
3. No explicit repeated-query experiment, but the same input `[0.851221, 0.611593, 0.504938]` was queried three times with near-identical outputs (~-0.0047 to -0.014), suggesting low observation noise around that point.
4. Comments describe **"three known peaks"** in the landscape — multimodal — with UCB and EI both initially struggling and repeatedly suggesting unexplored corners before the search was manually restricted to bounds around the known peaks.

### Your optimisation strategy
1. GP-based Bayesian optimisation using both EI and UCB (with a partially implemented, unused Probability-of-Improvement function), plus an auxiliary NN for saliency.
2. Matérn (ν=1.5) plus a `WhiteKernel` for noise was chosen as "a good balance between smooth and rugged" and to explicitly separate observation noise from landscape shape, appropriate given the noted multimodality.
3. EI used an unusually large `ξ=10` and UCB used `κ=10`, later raised to `κ=100` "to find peaks" — both biased heavily toward exploration; a 40-random-restart `L-BFGS-B` search was used to avoid missing distant peaks in 3D.
4. Yes — after both acquisition functions kept proposing unexplored corners of the domain, the team switched to (nominally) localising the search "within the three known peaks," though the coded bounds were left at the full `[0,1]³` box rather than a genuinely narrowed region.

### Data handling and preprocessing
1. No rescaling (inputs already `[0, 1]`); no output transform, since the output range here is modest (roughly `[-0.5, 0]`) and doesn't need log-scaling like Function 1.
2. Yes — `GaussianProcessRegressor`, `Matérn(ν=1.5) + WhiteKernel(noise_level=1e-5)`, `alpha=0.1`, `n_restarts_optimizer=20`.
3. `length_scale=[1,1,1]` with bounds `(1e-2, 1e2)`; a `ConvergenceWarning` showed one dimension's length scale landing near its upper bound, suggesting the bound could be widened.
4. The repeated exact-duplicate query `[0.851221, 0.611593, 0.504938]` was kept rather than treated as an error — it was useful confirmation that the surrogate's best-known point was stable, not an artefact.

### Weekly iteration and learning
1. Early points were spread across the cube; later points converged strongly on one peak once it was identified as the best of the three known regions.
2. Yes — the repeated corner-seeking by UCB/EI early on was the main sign of multiple local optima competing for the acquisition function's attention.
3. The point `[0.851221, 0.611593, 0.504938]`, queried three times, was the most informative for confirming stability of the best-found region.
4. With hindsight, actually coding the "localised bounds" mentioned in the iteration-3 comment (rather than leaving them at the full domain) would likely have produced a more efficient search in later weeks.

### Performance and results
1. Best output found: **-0.004686647016489443** (closest to zero of all observed values).
2. At input **[0.851221, 0.611593, 0.504938]**.
3. Reasonably confident locally — the point was re-queried three times with consistent results — but only one of the three known peaks was ever fully exploited, so the global optimum across all three peaks is not confirmed.
4. Consistent with expectations: a drug-combination search typically has several locally good combinations (the "three peaks"), and the team successfully found and confirmed one of them.

### Ethical, practical and general considerations
1. Directly mirrors real combinatorial drug-discovery screening, where each combination test is expensive and only a handful can be run per cycle.
2. The synthetic function has none of the real biological variability, toxicity risk, or assay noise a genuine drug trial would carry.
3. The approach (GP + EI/UCB with periodic manual bound restriction) would scale reasonably to more expensive real assays, provided the "localise near known peaks" step is actually implemented in code rather than left as a comment.
4. A future user should note that only one of the three identified peaks was ever exploited in depth — the other two remain largely unexplored and might contain a better combination.

---

## Function 4: Warehouse Placement

### Function overview
1. **Function 4.**
2. Optimally placing products across warehouses for a business with high online sales, where accurate placement calculations are costly and only feasible biweekly.
3. 4D.
4. 30 initial points, growing to 40 after weekly additions.
5. Output represents a placement-efficiency/cost score (mixed sign — mostly negative "cost," with a smaller number of positive "efficiency" results).

### Nature of the data
1. Initial `X` shape `(30, 4)`, `y` shape `(30,)`.
2. One point per week; queries moved from wide exploration (`κ=50` UCB over a semi-restricted `[0.2, 0.5]` box) to a tightly localised EI search (±0.2 box) once a positive-yield region near `[0.41, 0.41, 0.35, 0.42]` was found.
3. No explicit repeat-query test; several near-duplicate points (e.g. `[0.413140,...]`, `[0.413152,...]`, `[0.425258,...]`) around the best region returned consistently positive but slightly varying results (0.66, 0.55, 0.49), consistent with mild observation noise or genuine local sensitivity.
4. Mostly smooth and unimodal within the explored region: once the positive-yield pocket around `[0.41, 0.41, 0.35, 0.42]` was found, nearby points stayed positive, while points elsewhere in the domain were strongly negative (down to -32.6), suggesting one good region surrounded by a much larger poor region.

### Your optimisation strategy
1. GP-based Bayesian optimisation, first UCB then EI, plus an auxiliary NN for saliency.
2. Matérn (ν=2.5), noted in-notebook as "robust for high-D optimisation," was chosen given the 4D input space, more flexible than a smoother kernel for capturing the sharp transition between the good pocket and the poor surrounding region.
3. UCB (`κ=50`) was used first over a wide `[0.2, 0.5]⁴` box to explore; once a good region was found, the team switched explicitly to EI with a tight ±0.2 local box around the current best, favouring exploitation.
4. Yes — an explicit comment ("shifted from UCB to EI - to narrow in on the current peak") documents the exploration-to-exploitation switch once the positive region was located.

### Data handling and preprocessing
1. No rescaling (inputs already `[0, 1]`); no output transform (range is moderate, roughly `[-33, 1]`).
2. Yes — `GaussianProcessRegressor`, `Constant × Matérn(ν=2.5)`, `alpha=0.1`, `n_restarts_optimizer=20`.
3. `length_scale=[0.1]*4` with bounds `(0.5, 2.0)`; a `ConvergenceWarning` showed one dimension's length scale near its upper bound.
4. No formal outlier handling; one weekly data-entry issue was found on review — a missing comma between two appended output values causes Python to silently subtract them rather than treat them as two separate entries, corrupting that single data point (it does not affect the reported best value, which comes from an earlier entry).

### Weekly iteration and learning
1. Early wide exploration revealed the domain was mostly poor (negative), sharpening the search toward the one promising pocket found early on.
2. A rank table of all observed values (`np.argsort(y)`) was used explicitly to identify and track the best/worst performing points each week.
3. The cluster of near-duplicate points around `[0.41, 0.41, 0.35, 0.42]` was most informative, confirming that region as a genuine local optimum rather than a lucky single sample.
4. With hindsight, double-checking each appended output row for syntax errors (like the missing comma found here) before fitting the GP would guard against silently corrupting a data point.

### Performance and results
1. Best output found: **0.6576375369770244**.
2. At input **[0.413140, 0.406530, 0.351979, 0.424546]**.
3. Reasonably confident within the explored pocket (multiple nearby points confirm it), but the wide, mostly-negative rest of the 4D domain was only sparsely sampled, so a better pocket elsewhere cannot be ruled out.
4. Consistent with the described "costly, infrequent placement calculation" domain — most placements are inefficient, and the search successfully zeroed in on one efficient configuration.

### Ethical, practical and general considerations
1. Mirrors real warehouse/inventory placement optimisation, where each placement evaluation is a genuinely expensive operational-research computation.
2. As a synthetic function it lacks real supply-chain constraints (capacity limits, geography, lead times) that would shape a genuine warehouse-placement search.
3. The exploration-then-exploitation switching strategy used here would scale reasonably well, but the manual data-entry step (typing new results into a Python list) is a real risk at scale and should be replaced with a validated data pipeline.
4. A future user should be aware of the missing-comma data-entry issue found in this notebook — a reminder that hand-typed weekly results are a real source of silent corruption.

---

## Function 5: Chemical Process Yield

### Function overview
1. **Function 5.**
2. Optimising a four-variable black-box function representing the yield of a chemical process in a factory.
3. 4D.
4. 20 initial points, growing to ~32 after weekly additions.
5. Output represents a chemical yield (large positive number, unbounded above).

### Nature of the data
1. Initial `X` shape `(20, 4)`, `y` shape `(20,)`.
2. One point per week; queries increasingly favoured the high corner of the domain (e.g. `[0.99, 0.99, 0.99, 0.99]`, `[1, 0, 1, 1]`) as yields kept increasing there.
3. No formal repeat-query test, but the exact same input `[0.000001, 0.000001, 0.999999, 0.999999]` was queried twice and returned an identical yield (1616.63) both times, indicating low or zero observation noise.
4. Highly **skewed/monotonic-looking** toward one corner of the domain: yields span from ~0.1 to several thousand (over four orders of magnitude), and nearly every high-yield point has most or all coordinates near 1 — consistent with a landscape that increases toward a domain corner rather than having an interior peak.

### Your optimisation strategy
1. GP-based Bayesian optimisation with EI and UCB, plus an auxiliary NN for saliency.
2. Matérn (ν=2.5) with per-dimension length scales was chosen for its flexibility in 4D; the output's huge dynamic range motivated a `log1p` transform plus `normalize_y=True` before fitting, so the GP wasn't dominated by the largest few yields.
3. EI used an unusually large `ξ=101` (in log-space) and UCB used `κ` ranging from 2.576 (99% CI) up to 5.0 "to find peaks" — both strongly exploration-biased, consistent with the still-rising yields near the domain boundary.
4. Yes — later suggestions moved from a single-start EI search to random-restart UCB search (10 starts) as the team tried to confirm whether yield kept rising all the way to the domain corner or would eventually turn over.

### Data handling and preprocessing
1. No input rescaling (already `[0, 1]`); outputs were `log1p`-transformed given the multi-order-of-magnitude range, and further standardised via `normalize_y=True` in the GP.
2. Yes — `GaussianProcessRegressor`, `Constant × Matérn(ν=2.5)`, `alpha=1`, `n_restarts_optimizer=40`, `normalize_y=True`.
3. `length_scale=[0.1]*4` initial guess with bounds `(0.01, 10.0)`; after fitting, two of the four length scales hit the upper bound (10.0), triggering a `ConvergenceWarning` and suggesting the bound needed widening further.
4. No points were removed; the repeated identical query (see above) was kept as a useful noise sanity-check rather than deduplicated.

### Weekly iteration and learning
1. Each new high-value query near the domain corner reinforced the hypothesis that yield increases with all four chemical parameters, shifting the strategy from broad exploration to corner-probing.
2. No clear separate local optimum was found — the landscape looked closer to monotonic-toward-a-corner than genuinely multimodal, so "local optima" in the usual sense were not really encountered.
3. Boundary/corner points (`[0.99]*4`, `[1,0,1,1]`) were the most informative, each confirming or refining the direction of increasing yield.
4. **Note on reproducibility**: the notebook's own printed "current best" (7915.74, from `max(y)` run mid-session) does not match the true maximum in the final appended data (8662.41), because the cell was not re-run after a later data point was added — a good example of why notebook outputs should be verified by re-running top-to-bottom rather than trusted at face value.

### Performance and results
1. Best output value in the final dataset: **8662.405001248297** (the in-notebook printed "current best" of 7915.74 is stale — see note above).
2. At input **[0.960914, 0.990000, 0.990000, 0.990000]**.
3. Low-to-moderate confidence that this is the global maximum: the landscape appears to still be increasing toward the `[1,1,1,1]` corner, which was never directly queried, so a higher yield may exist right at or beyond the domain boundary.
4. Consistent with expectations for a chemical yield function where more of each reagent generally helps, up to some real-world limit not captured by the `[0,1]` bounds.

### Ethical, practical and general considerations
1. Mirrors real chemical-process yield optimisation, where each run is costly (time, reagents, safety) and only a few experiments can be run per cycle.
2. The synthetic function's yields are unbounded within `[0,1]⁴`, whereas a real chemical process would have physical/safety limits well before reaching "all reagents at maximum."
3. This strategy (log-transform + GP + high-exploration EI/UCB) would scale reasonably to a genuinely expensive process, but the corner-seeking behaviour is a real risk in practice — a real process pushed to its parameter extremes could be unsafe or infeasible, unlike this synthetic domain.
4. A future user should be aware that (a) the notebook's printed "best value" was stale relative to the actual data at one point, and (b) the apparent optimum sits right at the edge of the sampled domain, so it may not be a true interior optimum.

---

## Function 6: Cake Recipe

### Function overview
1. **Function 6.**
2. Optimising a cake recipe using a black-box function with five ingredient inputs (e.g. flour, sugar, eggs, butter, milk).
3. 5D.
4. 20 initial points, growing to ~30–33 after weekly additions.
5. Output represents a recipe quality score; values in this dataset are consistently negative, with values closer to zero taken as better.

### Nature of the data
1. Initial `X` shape `(20, 5)`, `y` shape `(20,)`.
2. One point per week; two near-duplicate points (`[0.484049,…]` and `[0.484318,…]`) were queried close together and returned very similar scores (-1.79 and -1.79), and the identical point `[0.000001, 0.000001, 0.999999, 0.999999]` recurs verbatim in another function's data list, suggesting some copy-paste reuse of candidate points across weeks.
3. No formal repeat test, but the two near-duplicate points above returned consistent results, suggesting low observation noise.
4. Appears **rugged/canyon-like** rather than smooth: 3 of 5 fitted length scales saturated at their upper bound (10.0), and an in-notebook comment explicitly describes "exploring the canyon" — i.e. a narrow, elongated region of relatively better (less negative) values rather than a single smooth peak.

### Your optimisation strategy
1. GP-based Bayesian optimisation with EI and UCB, plus an auxiliary NN for saliency.
2. Matérn (ν=2.5) was chosen for its flexibility in a 5D, apparently rugged landscape.
3. EI (`ξ=10`) and UCB (`κ=10`) were both used; EI's acquisition surface was searched via 50,000 random samples (seeded) refined with `L-BFGS-B`, a heavier search than other functions, reflecting the higher dimensionality and rugged canyon shape.
4. Yes, and notably: the notebook documents catching a **sign-convention bug** mid-project — a comment records "we were previously maximising, i.e. making the negative less negative as opposed to making the negative more negative," i.e. the acquisition logic had been pushing the score in the wrong direction relative to the intended (maximisation) objective. A fix (lower exploration, tighter length scales 0.2–0.8) was proposed in the following cell but its actual code application is not clearly visible in the notebook.

### Data handling and preprocessing
1. No input rescaling; no output transform (unlike Functions 1/5, the range here, roughly `[-3.2, -0.5]`, doesn't need log-scaling).
2. Yes — `GaussianProcessRegressor`, `Constant × Matérn(ν=2.5)`, `alpha=1`, `n_restarts_optimizer=20`.
3. `length_scale=[0.5]*5` with bounds `(0.1, 10)`; 3 of 5 dimensions hit the upper bound, explicitly flagged via a custom warning print recommending wider bounds — a recommendation not clearly acted on afterward.
4. No points removed; the near-duplicate/repeated points across weeks were kept as-is.

### Weekly iteration and learning
1. New points confirmed the "canyon" shape — most of the 5D space scored similarly poorly, with a narrower band of relatively better recipes.
2. Yes, indirectly: the acquisition function repeatedly suggesting all-ingredients-required combinations, combined with the sign-convention confusion, made it hard to tell true local optima from an artefact of the (temporarily) inverted objective.
3. The point `[0.009710, 0.979677, 0.029675, 0.011367, 0.857560]` — the best (least-bad) score under the notebook's own `min(y)` framing — was the most informative single point, repeatedly used as a start for subsequent local searches.
4. If restarted, the sign-convention check would be done first, before any acquisition tuning, to avoid an entire round of the project potentially optimising in the wrong direction.

### Performance and results
1. Best output value: **-0.4954099363279621** under the maximisation objective stated in the project brief (closest to zero); the notebook's own working "current best" was **-3.1365278755680133** under a `min(y)` framing that a later comment flags as likely a logic error.
2. Best (maximisation) input: **[0.381380, 0.098554, 0.582360, 0.694000, 0.050001]**. The notebook's `min(y)`-framed "best" input was **[0.009710, 0.979677, 0.029675, 0.011367, 0.857560]**.
3. Low confidence either way: given the documented sign-convention confusion, and 3 of 5 length scales sitting at their bound (suggesting the kernel may still be over-smoothing a genuinely rugged landscape), neither reported "best" should be treated as close to a confirmed global optimum.
4. Partially — the "canyon" shape and the comment "all ingredients are required" are broadly consistent with a recipe-style function where extreme (near-zero or near-one) ingredient amounts tend to fail, but the objective-direction confusion means the specific numeric results should be treated cautiously.

### Ethical, practical and general considerations
1. Mirrors real recipe/formulation optimisation, where the "cost" of each trial is baking and evaluating an actual cake.
2. As a synthetic function, it has none of the sensory subjectivity or real ingredient-interaction chemistry a genuine recipe-optimisation task would involve.
3. This strategy would need a resolved, tested acquisition-direction sign convention before scaling to a more expensive real process — the bug found here is exactly the kind of silent error that would be very costly in a real, expensive experiment.
4. A future user must double-check whether this notebook is minimising or maximising `y` before trusting any of its "best point" output — the notebook itself documents catching this ambiguity but does not conclusively resolve it in code.

---

## Function 7: ML Hyperparameter Tuning

### Function overview
1. **Function 7.**
2. Optimising an ML model by tuning six hyperparameters (e.g. learning rate, regularisation strength, number of hidden layers).
3. 6D.
4. 30 initial points, growing to ~40 after weekly additions.
5. Output represents a model performance score (higher is better).

### Nature of the data
1. Initial `X` shape `(30, 6)`, `y` shape `(30,)`.
2. One point per week; the saved chart (`function_7_chart_9th_iteration.png`) shows performance bouncing between roughly 0 and 2 across the 9 weekly iterations shown, without a clear monotonic trend.
3. No formal repeat-query test; no exact-duplicate inputs were queried, so noise could not be directly assessed from this data.
4. Appears **noisy/multimodal**: most sampled points score near zero, with a small number of much higher-scoring points (up to ~2.0) scattered non-contiguously — typical of a hyperparameter-tuning landscape where most configurations are mediocre and a few "get lucky."

### Your optimisation strategy
1. GP-based Bayesian optimisation with both EI and UCB, plus an auxiliary NN for saliency.
2. Matérn (ν=2.5) plus a `WhiteKernel` was chosen to explicitly separate real landscape structure from the noisy, spiky pattern of scores typical of hyperparameter search.
3. EI (`ξ=0.1`) was optimised via a **Sobol-sequence** multi-start (32 starts, chosen as a power of 2 for Sobol efficiency) plus `L-BFGS-B` refinement, needed because a dense grid search is infeasible in 6D; UCB (`κ=10`) was also tried, from a single start at the current best point.
4. Yes — an explicit comment ("continuing to exploit with moderate exploration") marks a deliberate shift toward exploitation once a strong-scoring region (~1.9–2.0) was found.

### Data handling and preprocessing
1. No input rescaling (hyperparameters already normalised to `[0,1]`); no output transform (range is modest, roughly `[0, 2]`).
2. Yes — `GaussianProcessRegressor`, `Constant × Matérn(ν=2.5) + WhiteKernel(noise_level=1)`, `n_restarts_optimizer=25`.
3. `length_scale=[0.5]*6` with bounds `(0.01, 10.0)`; after fitting, one dimension's length scale hit the upper bound, flagged via `ConvergenceWarning`.
4. No explicit outlier handling; the many near-zero scores were kept as genuine (poor-performing) observations rather than treated as noise.

### Weekly iteration and learning
1. Early broad sampling revealed most of the 6D space scores poorly; later Sobol-seeded EI search located a smaller region scoring consistently higher (~1.3–2.0).
2. Yes — the scattered, non-contiguous pattern of high-scoring points (visible in the saved iteration chart) is consistent with several competing local optima rather than one smooth peak.
3. The higher-scoring points (indices with values 1.3–2.0) were most informative for narrowing the search; near-zero points mainly served to rule out large parts of the domain.
4. If restarted, running the Sobol multi-start search from the very first iteration (rather than starting with denser grid-style sampling) would likely have found the productive region faster.

### Performance and results
1. Best output found: **1.9993628649406798**.
2. At input **[0.000001, 0.237979, 0.404631, 0.121878, 0.334399, 0.759561]**.
3. Moderate confidence: the Sobol multi-start (32 starts) gives reasonable coverage of the 6D acquisition surface, but the scattered nature of high-scoring points suggests other undiscovered good regions may remain.
4. Consistent with expectations for hyperparameter tuning — most configurations underperform, with a few standout combinations, matching the observed distribution.

### Ethical, practical and general considerations
1. Mirrors real (and increasingly common) automated hyperparameter tuning, where each "evaluation" is a full model training run.
2. The synthetic function skips the real computational cost, training instability, and stochasticity a genuine model-training evaluation would have.
3. This strategy (Sobol multi-start + GP-EI/UCB) is a standard, scalable approach and would transfer well to genuinely expensive hyperparameter searches, more so than the ad hoc manual approaches used on lower-dimensional functions in this project.
4. A future user should note the scattered, non-contiguous pattern of good results — assuming a single smooth "peak" exists here (as might be reasonable for Functions 1–4) would be misleading.

---

## Function 8: Unknown 8D Function

### Function overview
1. **Function 8.**
2. An eight-dimensional black-box function where each of the eight input parameters affects the output, but the internal mechanics are unknown.
3. 8D.
4. 40 initial points, growing to ~50–53 after weekly additions.
5. Output represents an unlabelled performance score (higher is better), observed to range roughly between 5.6 and 9.9.

### Nature of the data
1. Initial `X` shape `(40, 8)`, `y` shape `(40,)`.
2. One point per week; queries stayed within a moderate output range throughout (no extreme outliers), suggesting the initial sampling had already covered a reasonably representative part of the landscape.
3. No formal repeat-query test; no exact-duplicate inputs were found in the appended data, so noise could not be directly assessed.
4. Given the relatively tight output range (5.6–9.9, less than one order of magnitude) compared to Functions 1 and 5, this landscape looks **comparatively smooth**, though genuinely visualising an 8D surface isn't possible — this assessment relies on the GP's ARD length scales (see below) and the auxiliary NN's saliency scores rather than a direct plot.

### Your optimisation strategy
1. GP-based Bayesian optimisation with EI and UCB, plus an auxiliary NN for saliency.
2. Matérn (ν=2.5) with independent (ARD) length scales per dimension was chosen specifically "to learn which inputs are actually driving the 9.90 output" — i.e. to use the kernel fit itself as a feature-importance tool in a dimensionality too high to visualise directly.
3. EI used a deliberately high `ξ=1.0` ("high exploration parameter for 8D... forces the model to look for points that could beat 9.90 by a significant margin"), optimised via **Sobol-sequence multi-start** with 64 starts (doubled from Function 7's 32, "increased for 8D complexity") plus `L-BFGS-B`; UCB (`κ=10`) was also tried from a single start.
4. Yes — after the first kernel fit showed one length scale saturating at its upper bound (10.0), the team explicitly retightened the bounds (`length_scale_bounds` from `(0.01, 10)` to `(0.01, 1.0)`, noise bounds tightened to `(1e-6, 1e-4)`) "to force the model to acknowledge there is 'detail' in the landscape."

### Data handling and preprocessing
1. No input rescaling; no output transform (the range, roughly `[5.6, 9.9]`, doesn't need log-scaling unlike Functions 1/5).
2. Yes — `GaussianProcessRegressor`, `Constant × Matérn(ν=2.5) + WhiteKernel(noise_level=1)`, `n_restarts_optimizer=25`, later retuned to tighter bounds (see above).
3. Beyond the retightened length-scale and noise bounds, the fitted ARD length scales themselves (ranging 2.38–10.0 across the 8 dimensions) were inspected explicitly to identify which of the 8 inputs mattered most.
4. No outliers removed; all weekly points were kept.

### Weekly iteration and learning
1. The tight output range across all sampled points suggested no single input dominates the outcome as strongly as in Functions 1 or 5; the ARD length-scale check helped narrow down which of the 8 inputs mattered most.
2. No clear separate local optima were identified — the relatively narrow output range and the decision to retighten kernel bounds (rather than switch acquisition strategy) suggests the team treated this as more of a "find the fine detail" problem than a multimodal-optimum-hunting one.
3. Points near the eventual best value (9.90+) were most informative; the retightened kernel bounds were specifically motivated by wanting the model to distinguish between these closely-scored top points.
4. If restarted, using the ARD ("which inputs matter") analysis from the very first iteration — rather than after the initial default-bounds fit already suggested it — would have focused the Sobol search on the important dimensions sooner.

### Performance and results
1. Best output value in the final dataset: **9.9122133391516**. (As with Function 5, the notebook's own printed "current best," 9.9036, is a mid-session snapshot rather than the true final maximum — again a reminder to re-verify by re-running top-to-bottom.)
2. Input not explicitly logged for the true maximum in the notebook; the mid-session "current best" of 9.9036 corresponds to input `[0.043396, 0.053307, 0.137597, 0.000025, 0.776263, 0.493814, 0.043575, 0.567457]` — treat this pairing as indicative rather than confirmed.
3. Low-to-moderate confidence: 64-start Sobol search gives decent coverage for 8D, but 8D is a large space to cover with ~50 total observations, and the exact best-input pairing above is not fully confirmed in-notebook.
4. Broadly consistent with a generic "all 8 inputs matter, no obvious single dominant dimension" black-box function, matching the problem description.

### Ethical, practical and general considerations
1. Mirrors real high-dimensional black-box optimisation problems (e.g. multi-parameter engineering or process tuning) where no single variable's effect is obvious in advance.
2. As a synthetic function, it lacks real 8D physical constraints or interaction effects that would make a genuine high-dimensional process much harder to reason about than this dataset suggests.
3. The Sobol-multi-start + ARD-kernel approach used here is a genuinely scalable pattern and is the most "production-ready" strategy in this project — it would extend reasonably well to more expensive, higher-dimensional real problems.
4. A future user should treat the exact "best input" for the top value with caution, since (as with Function 5) it isn't unambiguously logged in the notebook, and 8D coverage with ~50 points is inherently sparse.
