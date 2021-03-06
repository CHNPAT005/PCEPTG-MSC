## Author: Patrick Chang
# Script file to project eigenvectors for various time scales
# to see if there is any change in underlying correlation/covariation
# structure depending on the time scale
# Done using 10 assets with varying liquidity

using LinearAlgebra; using LaTeXStrings; using StatsBase; using Random;
using Statistics; using Distributions; using ProgressMeter; using JLD

cd("/Users/patrickchang1/PCEPTG-MM-NUFFT")

include("../Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")

include("../Correlation Estimators/Fejer/NUFFTcorrFK-FGG.jl")

include("../SDEs/GBM.jl")
include("../Misc/RandCovMat.jl")

#---------------------------------------------------------------------------
# Supporting functions
#---------------------------------------------------------------------------

function rexp(n, mean)
    t = -mean .* log.(rand(n))
end

#---------------------------------------------------------------------------

mu = repeat([0.01/86400], 10)
sig = repeat([sqrt(0.1/86400)], 10)
Random.seed!(2020)
sigma = gencovmatrix(10, sig)

P = GBM(86400*5, mu, sigma, seed = 2020)

#---------------------------------------------------------------------------
## Synchronous case

t = reshape(repeat(collect(1:1:86400*5), size(P)[2]), 86400*5 , size(P)[2])

N1H = 60
N1H = N
test = NUFFTcorrFKFGG(P, t, N = N1H)

N5M = 720
N5M = floor(N/2)
test2 = NUFFTcorrFKFGG(P, t, N = N5M)

N1M = 3600
N1M = floor(N/4)

N1M = floor(N/30)
test3 = NUFFTcorrFKFGG(P, t, N = N1M)

E1 = eigvecs(test[1])#[:, 1:3]
E2 = eigvecs(test2[1])#[:, 1:3]
E3 = eigvecs(test3[1])#[:, 1:3]

aa = E1' * E2

plot(E1' * E2, st=:heatmap, clim=(-1,1), color=cgrad([:blue, :white,:red, :yellow]), colorbar_title="y", flip = true)

plot(E2' * E3, st=:heatmap, clim=(-1,1), color=cgrad([:blue, :white,:red, :yellow]), colorbar_title="y", flip = true)

plot(E1' * E3, st=:heatmap, clim=(-1,1), color=cgrad([:blue, :white,:red, :yellow]), colorbar_title="y", flip = true)

#---------------------------------------------------------------------------
## Random Exponential


lam = [2 3 15 15 20 20 25 25 60 70]

Random.seed!(2020)
t1 = [1; rexp(86400*5, lam[1])]
t1 = cumsum(t1)
t1 = filter((x) -> x < 86400*5, t1)

D = length(t1)
p = zeros(D, 10)
p[:,1] = P[Int.(floor.(t1)), 1]
t = zeros(D, 10)
t[:,1] = t1

for i in 2:length(lam)
    Random.seed!(2020 + i)
    tstore = [1; rexp(86400*5, lam[i])]
    tstore = cumsum(tstore)
    tstore = filter((x) -> x < 86400*5, tstore)
    pstore = P[Int.(floor.(tstore)), i]


    Dstore = D - length(tstore)
    tstore = [tstore; repeat([NaN], Dstore)]
    pstore = [pstore; repeat([NaN], Dstore)]

    t[:,i] = tstore
    p[:,i] = pstore
end


tau = scale(t)
# Computing minimum time change
# minumum step size to avoid smoothing
dtau = diff(filter(!isnan, tau))
taumin = minimum(filter((x) -> x>0, dtau))
taumax = 2*pi
# Sampling Freq.
N0 = taumax/taumin
N = floor(N0/2)
