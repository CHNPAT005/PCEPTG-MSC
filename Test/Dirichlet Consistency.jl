## Author: Patrick Chang
# Script file to test the various Dirichlet implementations
# to ensure they are all self consistent

cd("/Users/patrickchang1/PCEPTG-MSC")

using StatsBase; using Random
include("../Functions/Correlation Estimators/Dirichlet/CFTcorrDK.jl")
include("../Functions/Correlation Estimators/Dirichlet/FFTcorrDK.jl")
include("../Functions/Correlation Estimators/Dirichlet/FFTZPcorrDK.jl")
include("../Functions/Correlation Estimators/Dirichlet/MScorrDK.jl")
include("../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FINUFFT.jl")
include("../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")
include("../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-KB.jl")
include("../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-ES.jl")
include("../Functions/SDEs/GBM.jl")

#---------------------------------------------------------------------------
## Synchronous Case

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

P = GBM(10000, mu, sigma, seed = 10)
t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

#--------------------

s1 = CFTcorrDK(P, t)
s2 = FFTcorrDK(P)
s3 = FFTZPcorrDK(P, t)
s4 = MScorrDK(P, t)
s5 = NUFFTcorrDKFINUFFT(P, t)
s6 = NUFFTcorrDKFGG(P, t)
s7 = NUFFTcorrDKKB(P, t)
s8 = NUFFTcorrDKES(P, t)

#---------------------------------------------------------------------------
## Asynchronous Case (Down-sampled 40%)

rm1 = sample(2:9999, 4000, replace = false)
rm2 = sample(2:9999, 4000, replace = false)

P[rm1, 1] .= NaN
t[rm1, 1] .= NaN
P[rm2, 2] .= NaN
t[rm2, 2] .= NaN

#--------------------
# Can't include FFTcorrDK

as1 = CFTcorrDK(P, t)
# as2 = FFTcorrDK(P)
as3 = FFTZPcorrDK(P, t)
as4 = MScorrDK(P, t)
as5 = NUFFTcorrDKFINUFFT(P, t)
as6 = NUFFTcorrDKFGG(P, t)
as7 = NUFFTcorrDKKB(P, t)
as8 = NUFFTcorrDKES(P, t)
