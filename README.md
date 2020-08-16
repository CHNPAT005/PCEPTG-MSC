# High-frequency correlation dynamics: Is the Epps effect a bias?

Patrick Chang's Masters dissertation supervised by Tim Gebbie and Etienne Pienaar working on building a suit of fast integrated and instantaneous volatility/co-volatility estimators. Using these tools we investigate the correlation dynamics at ultra-high frequency to determine if correlations are an emerging property.

## Authors:
- Patrick Chang
- Etienne Pienaar
- Tim Gebbie

## Link to resources:
Papers:
- Malliavin-Mancino estimators implemented with non-uniform fast Fourier transforms: https://arxiv.org/abs/2003.02842
- Fourier instantaneous estimators and the Epps effect: https://arxiv.org/abs/2007.03453
- Using the Epps effect to detect discrete data generating processes: https://arxiv.org/abs/2005.10568

Datasets used in the dissertation:
[Link1](https://zivahub.uct.ac.za/articles/Malliavin-Mancino_estimators_implemented_with_the_non-uniform_fast_Fourier_transform_Dataset/11903442) and [Link2](https://zivahub.uct.ac.za/articles/dataset/Using_the_Epps_effect_to_detect_discrete_data_generating_processes_Dataset/12315092)

## Steps for Replication:

First, the code must be downloaded/cloned. Next, download the datasets from ZivaHub and place the CSV files into the folder `/Real Data`. Each of the results in the dissertation have the script file associated that produces the result. Before running the script ensure that the directory is changed. Currently the directories are set as: `cd("/Users/patrickchang1/PCEPTG-MSC")`. Change this to where you have stored the file `PCEPTG-MSC`. Now it is a simple matter of running the script file to reproduce the results. Note that for the results that have a long compute time, I have stored the results in `/Computed Data` so that the figures and tables can be reproduced without the long run time.

## Using the functions for other purposes:
### NUFFT

The functions include the 1-Dimensional Type 1 non-uniform fast Fourier transforms using three types of kernels. These functions can be found under the folder [\Functions\NUFFT](https://github.com/CHNPAT005/PCEPTG-MSC/tree/master/Functions/NUFFT).

The functions require four input variables:
- cj: vector of source strength,
- xj: vector of source points,
- M: the number of Fourier coefficients you want returned (integer), and
- tol: the precision requested

##### Example

```julia

include("Functions/NUFFT/NUFFT-FGG.jl")
include("Functions/NUFFT/NUFFT-KB.jl")
include("Functions/NUFFT/NUFFT-ES.jl")

# Simulate some non-uniform data
nj = 10
x = (collect(0:nj-1) + 0.5 .* rand(nj))
xj = (x .- minimum(x)) .* (2*pi / (maximum(x) - minimum(x))) 	# Re-scale s.t. xj \in [0, 2\pi]
cj = rand(nj) + 0im*rand(nj)

# Parameter settings
M = 11 # Output size
tol = 10^-12 # Tolerance

# Output 
fk_G = NUFFTFGG(cj, xj, M, tol)   # Gaussian kernel
fk_KB = NUFFTKB(cj, xj, M, tol)   # Kaiser-Bessel kernel
fk_ES = NUFFTES(cj, xj, M, tol)   # Exponential of semi-circle kernel

```

### Integrated estimators
#### Malliavin-Mancino integrated estimators using NUFFTs

The implementation is performed using two representations: the Dirichlet and the Fejér. Both representations require two input variables:

- p: (nxD) matrix of prices, with non-trade times represented as NaNs,
- t: (nxD) corresponding matrix of trade times, with non-trade times represented as NaNs,

and two optional input variables:

- N: (optional input) for the number of Fourier coefficients used in the convolution of the Malliavin-Mancino estimator (integer) - defaults to the Nyquist frequency,
- tol: tolerance requested - defaults to 10^-12.


##### Example

```julia

include("Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")
include("Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FGG.jl")
include("Functions/SDEs/GBM.jl")

# Create some data
mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

P = GBM(10000, mu, sigma, seed = 10)
t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

# Parameter settings
N = 500
tol = 10^-12

# Obtain results
output1 = NUFFTcorrDKFGG(P, t, N = N, tol = tol)    # Dirichlet
output2 = NUFFTcorrFKFGG(P, t, N = N, tol = tol)    # Fejer

# Extract results
cor1 = output1[1]   # correlation matrix
cov1 = output1[2]   # integrated covariance

cor2 = output2[1]   # correlation matrix
cov2 = output2[2]   # integrated covariance

```

#### Hayashi-Yoshida estimator

The Hayashi-Yoshida implementation is performed using the Kanatani weight matrix. The function only computes the integrated covariance for two assets at a time. The function requires four input variables:

- p1: vector of observed prices for the first asset,
- p2: vector of observed prices for the second asset,
- t1: vector of observed trading times for the first asset, and
- t2: vector of observed trading times for the second asset.

##### Example

```julia

include("Functions/Correlation Estimators/HY/HYcorr.jl")
include("Functions/SDEs/GBM.jl")

# Create some data
mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

P = GBM(10000, mu, sigma, seed = 10)
t = collect(1:1:10000.0)

# Obtain results
output = HYcorr(p1 = P[:,1], p2 = P[:,2], t1 = t, t2 = t)

# Extract results
cor = output[1]   # correlation matrix
cov = output[2]   # integrated covariance

```

### Instantaneous estimators
#### Malliavin-Mancino instantaneous estimators using NUFFTs

The function requires three input variables:
- p: (nx2) matrix of prices, with non-trade times represented as NaNs,
- t: (nx2) corresponding matrix of trade times, with non-trade times represented as NaNs,
- outlength: the number of synchronous grid points to reconstruct the spot estimates.

and three optional input variables:

- N: (optional input) for the number of Fourier coefficients of the price process used in the convolution of the Malliavin-Mancino estimator (integer), controls the level of averaging and directly affects the time-scale investigated - defaults to the Nyquist frequency,
- M: (optional input) for the number of Fourier coefficients of the volatility process using in the reconstruction of the spot estimates - defaults to <img src="https://render.githubusercontent.com/render/math?math=M = \frac{1}{8} \frac{1}{2\pi} \sqrt{n} \log n">, and
- tol: tolerance requested - defaults to 10^-12.

##### Example

```julia

include("Functions/SDEs/Heston.jl")
include("Functions/Instantaneous Estimators/MM-Inst.jl")

# Simulate some price observations from the Heston model.

nsim = 28800
P_Heston = Heston_CT(nsim, seed = 1, dt = nsim)
	# First variable in P_Heston is the price matrix, 
	# Second to fourth variable are the true volatility and co-volatility.
t = collect(1:1:nsim)

# Parameter settings
outlength = 1000	# length of output vector
M = 100	# Cutting freq.
tol = 10^-12

# Output 
MM_Heston = MM_inst(P_Heston[1], [t t], outlength, M = M, tol = tol)
	# First variable is the volatility estimates of asset 1.
	# Second variable is the volatility estimates of asset 2.
	# Third variable is the co-volatility estimates of asset 1 and 2.
	
# Extract results
vol11 = MM_Heston[1]    # Vector of spot volatility estimates for the first asset
vol22 = MM_Heston[2]    # Vector of spot volatility estimates for the second asset
vol12 = MM_Heston[3]    # Vector of spot co-volatility estimates

```

#### Cuchiero-Teichmann

The Cuchiero-Teichmann instantaneous estimator uses the specification of g(x) = cos(x). The estimator requires the data to be strictly synchronous, therefore the asynchronous data needs to be synchronised beforehand using the previous tick interpolation.

The function requires three input variables:

- p: (n x 2) double float matrix of price observations,
- N: the cutting frequency (integer) used in the reconstruction of the spot estimates, the dissertation uses the notation M, and
- outlength: the number of synchronous grid points to reconstruct the spot estimates.

##### Example

```julia

include("Functions/SDEs/Heston.jl")
include("Functions/Instantaneous Estimators/MM-JR.jl")

# Simulate some price observations from the Heston model.

nsim = 28800
P_Heston = Heston_CT(nsim, seed = 1, dt = nsim)
	# First variable in P_Heston is the price matrix, 
	# Second to fourth variable are the true volatility and co-volatility.
t = collect(1:1:nsim)

# Parameter settings
outlength = 1000	# length of output vector
M = 100	# Cutting freq.

# Output 
JR_Heston = MM_JR(P_Heston[1], M, outlength)
	# First variable is the volatility estimates of asset 1.
	# Second variable is the volatility estimates of asset 2.
	# Third variable is the co-volatility estimates of asset 1 and 2.
	
# Extract results
vol11 = JR_Heston[1]    # Vector of spot volatility estimates for the first asset
vol22 = JR_Heston[2]    # Vector of spot volatility estimates for the second asset
vol12 = JR_Heston[3]    # Vector of spot co-volatility estimates

```

### Hawkes

The functions include a variety of functions for the simulation and calibration of a M-variate Hawkes process. The Hawkes process uses a single exponential kernel.

#### Simulation

The function to simulate the Hawkes process requires four input variables:

- lambda0: the vector of constant base-line intensity,
- alpha: MxM matrix of alphas in the exponential kernel,
- beta: MxM matrix of betas in the exponential kernel, and
- T: the time horizon of the simulation.

##### Example

```julia

include("Functions/Hawkes/Hawkes.jl")

# Setting the parameters for a 2-variate Hawkes process
lambda0 = [0.016 0.016]
alpha = [0 0.023; 0.023 0]
beta = [0 0.11; 0.11 0]
T = 3600

# Simulate the process
t = simulateHawkes(lambda0, alpha, beta, T)

# Extract the simulation results
t1 = t[1] # vector of arrival times for the first count process
t2 = t[2] # vector of arrival times for the second count process

```

#### Calibration

The calibration requires the user to decide on how many parameters to estimate and write a small function to initialise the input matrix of lambda0, alpha and beta and invoke the log-likelihood.

The calibration uses the Optim package in Julia.

##### Example

```julia
using Optim
include("Functions/Hawkes/Hawkes")

# Function to be used in optimization for the above simulation
# i.e. creating a function that takes in a vector of observations
# to create the Hawkes specification
function calibrateHawkes(param)
    lambda0 = [param[1] param[1]]
    alpha = [0 param[2]; param[2] 0]
    beta = [0 param[3]; param[3] 0]
    return -loglikeHawkes(t, lambda0, alpha, beta, T)   # t is the vector of vector of observations; 
    # returns the negative because Optim minimises
end

# Optimize the parameters using Optim
res = optimize(calibrateHawkes, [0.01, 0.015, 0.15]) # function to minimise
    # and vector of initial parameters
par = Optim.minimizer(res)  # MLE estimates of the parameter

```



