## Author: Patrick Chang
# Script file to test the various Fejer implementations
# to ensure they are all self consistent

cd("/Users/patrickchang1/PCEPTG-MSC")

using StatsBase
include("../Functions/Correlation Estimators/Fejer/CFTcorrFK.jl")
include("../Functions/Correlation Estimators/Fejer/FFTcorrFK.jl")
include("../Functions/Correlation Estimators/Fejer/FFTZPcorrFK.jl")
include("../Functions/Correlation Estimators/Fejer/MScorrFK.jl")
include("../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FINUFFT.jl")
include("../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FGG.jl")
include("../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-KB.jl")
include("../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-ES.jl")
include("../Functions/SDEs/GBM.jl")

#---------------------------------------------------------------------------
## Synchronous Case

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

P = GBM(10000, mu, sigma, seed = 10)
t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

#--------------------

s1 = CFTcorrFK(P, t)
s2 = FFTcorrFK(P)
s3 = FFTZPcorrFK(P, t)
s4 = MScorrFK(P, t)
s5 = NUFFTcorrFKFINUFFT(P, t)
s6 = NUFFTcorrFKFGG(P, t)
s7 = NUFFTcorrFKKB(P, t)
s8 = NUFFTcorrFKES(P, t)

#---------------------------------------------------------------------------
## Asynchronous Case (Down-sampled 40%)

rm1 = sample(2:9999, 4000, replace = false)
rm2 = sample(2:9999, 4000, replace = false)

P[rm1, 1] .= NaN
t[rm1, 1] .= NaN
P[rm2, 2] .= NaN
t[rm2, 2] .= NaN

#--------------------
# Can't include FFTcorrFK

as1 = CFTcorrFK(P, t)
#s2 = FFTcorrFK(P)
as3 = FFTZPcorrFK(P, t)
as4 = MScorrFK(P, t)
as5 = NUFFTcorrFKFINUFFT(P, t)
as6 = NUFFTcorrFKFGG(P, t)
as7 = NUFFTcorrFKKB(P, t)
as8 = NUFFTcorrFKES(P, t)
