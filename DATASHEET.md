# Motivation
The data set has been created to support the BBO hyperparameter optimisation. The data set provides respective function outputs for given inputs (hyperparameters). The data set was provided by Imperial Executive Education as part of the educational course Professional Certificate in Machine Learning and Artificial Intelligence as learning materials for students. The data set has been refined to include more data as part of the learning objective from Module 12 Bayesian Optimisation. The additional data was provided incrementally as students engaged with the weekly function submissions. There are ten additional data points that were obtained over ten weeks, one data output per week, per function.

Funding is solely by the course provider, Imperial Executive Education.

# Composition
| File                  | Content | Format   |
| :---                  |    :----:   |          ---: |
| initial_input.npy     | Original data provided by the course provider. The input contains the respective features per function of interest in varying dimensions, from two to eight. The inputs are floating point numbers precise to six decimal places ranging from zero to one. The number of rows vary from 10 to 30 as per function of interest. The functions with more dimension have more than 10 rows.     | NumPy binary file format   |
| initial_output.npy    | Original output provided by the course provider. The output contains a single column, floating point numbers. The number of rows vary from 10 to 30 as per function of interest. The functions with more dimension have more than 10 rows.   | NumPy binary file format   |
| incremental_input.npy     | The weekly function submissions. The number of columns match the number of dimensions for the respective function, and the data is floating point numbers precise to six decimal places between zero and one. There are 10 rows.   | NumPy binary file format   |
| incremental_output.npy    | The weekly function outputs as per weekly function input submissions. The output contains a single column, floating point numbers. There are 10 rows.| NumPy binary file format   |

# Collection process

The datasheet explains how queries were generated, the strategy used and the time frame

The "incremental" input and output data was collected on a weekly basis, one row at a time. The input queries were generated using statistical methods such as Gaussian Process with various parameters, see individual function Jupyter Notebooks for configuration. In total, the data was compiled over a period of ten consecutive weeks, from 28/03/2026 to 02/06/2026. The output data is the function output. The function operates as a black-box and our only insight into the internals are the function inputs.

# Preprocessing/cleaning/labelling
Whilst no data processing has been applied to the actual data in the files we do recommend pre-processing as part of the actual notebook workflow for numerical stability. For example, where data ranges are massive we recommend applying log to reduce the number range for numerical stability.

# Uses
The data should be used as part of ongoing hyperparameter optimisation for the respective functions. Hopefully, serving as initial warm-start data-points for priming the underlying approaches for optimal exploration and/or exploitation.

The data for a given function may not correlate accordingly for other functions, i.e. input and output are strictly for the function at hand.

# Distribution
The data sets (raw NumPy binary files) are available as part of the repository, under the individual function directories inside the directory "data". 

The data is available under MIT license, and may be used by aspiring students for inspiration or continuing work.

# Maintenance
The data set is no longer maintained nor supported from July, 2nd, 2026 onwards.