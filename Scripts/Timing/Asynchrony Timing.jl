## Author: Patrick Chang
# Script file to benchmark performance between various algorithms
# under asynchronous conditions.
# We compare the Mancino-Sanfelici code to Legacy code and the FFT and NUFFT

#---------------------------------------------------------------------------

using LinearAlgebra; using LaTeXStrings; using StatsBase; using Random;
using Statistics; using Distributions; using ProgressMeter; using JLD; using DataTables

#---------------------------------------------------------------------------

cd("/Users/patrickchang1/PCEPTG-MSC")

include("../../Functions/Correlation Estimators/Dirichlet/MScorrDK.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")


include("../../Functions/Correlation Estimators/Fejer/MScorrFK.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FGG.jl")

include("../../Functions/SDEs/GBM.jl")
include("../../Functions/SDEs/RandCovMat.jl")

#---------------------------------------------------------------------------
# Supporting Functions

function rexp(n, mean)
    t = -mean .* log.(rand(n))
end

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

#---------------------------------------------------------------------------
# Timing Functions

## Dirichlet

function timeDK(n, lambdarange, reps)
    resultFGG = zeros(reps, length(lambdarange))
    resultMRS = zeros(reps, length(lambdarange))
    n1 = zeros(length(lambdarange), 1)
    n2 = zeros(length(lambdarange), 1)
    N = zeros(length(lambdarange), 1)

    @showprogress "Computing..." for i in 1:length(lambdarange)
        P = GBM(n, mu, sigma, seed = i)

        Random.seed!(reps)
        t1 = [1; rexp(n, lambdarange[1])]
        t1 = cumsum(t1)
        t1 = filter((x) -> x < n, t1)

        Random.seed!(i+reps)
        t2 = [1; rexp(n, lambdarange[i])]
        t2 = cumsum(t2)
        t2 = filter((x) -> x < n, t2)

        n1[i] = length(t1)
        n2[i] = length(t2)

        p1 = P[Int.(floor.(t1)), 1]
        p2 = P[Int.(floor.(t2)), 2]

        D = maximum([length(t1); length(t2)]) - minimum([length(t1); length(t2)])
        if length(t1) < length(t2)
            t1 = [t1; repeat([NaN], D)]
            p1 = [p1; repeat([NaN], D)]
        else
            t2 = [t2; repeat([NaN], D)]
            p2 = [p2; repeat([NaN], D)]
        end

        P = [p1 p2]
        t = [t1 t2]

        tau = scale(t)
        # Computing minimum time change
        dtau = zeros(2,1)
        for i in 1:2
            dtau[i] = minimum(diff(filter(!isnan, tau[:,i])))
        end
        # maximum of minumum step size to avoid aliasing
        taumin = maximum(dtau)
        taumax = 2*pi
        # Sampling Freq.
        N0 = taumax/taumin

        N[i] = floor(N0/2)

        for j in 1:reps
            resultFGG[j, i] = @elapsed NUFFTcorrDKFGG(P, t)
            resultMRS[j, i] = @elapsed MScorrDK(P, t)
        end
        GC.gc()
    end
    return resultFGG, resultMRS, n1, n2, N
end

## Fejer

function timeFK(n, lambdarange, reps)
    resultFGG = zeros(reps, length(lambdarange))
    resultMRS = zeros(reps, length(lambdarange))
    n1 = zeros(length(lambdarange), 1)
    n2 = zeros(length(lambdarange), 1)
    N = zeros(length(lambdarange), 1)

    @showprogress "Computing..." for i in 1:length(lambdarange)
        P = GBM(n, mu, sigma, seed = i)

        Random.seed!(reps)
        t1 = [1; rexp(n, lambdarange[1])]
        t1 = cumsum(t1)
        t1 = filter((x) -> x < n, t1)

        Random.seed!(i+reps)
        t2 = [1; rexp(n, lambdarange[i])]
        t2 = cumsum(t2)
        t2 = filter((x) -> x < n, t2)

        n1[i] = length(t1)
        n2[i] = length(t2)

        p1 = P[Int.(floor.(t1)), 1]
        p2 = P[Int.(floor.(t2)), 2]

        D = maximum([length(t1); length(t2)]) - minimum([length(t1); length(t2)])
        if length(t1) < length(t2)
            t1 = [t1; repeat([NaN], D)]
            p1 = [p1; repeat([NaN], D)]
        else
            t2 = [t2; repeat([NaN], D)]
            p2 = [p2; repeat([NaN], D)]
        end

        P = [p1 p2]
        t = [t1 t2]

        tau = scale(t)
        # Computing minimum time change
        dtau = zeros(2,1)
        for i in 1:2
            dtau[i] = minimum(diff(filter(!isnan, tau[:,i])))
        end
        # maximum of minumum step size to avoid aliasing
        taumin = maximum(dtau)
        taumax = 2*pi
        # Sampling Freq.
        N0 = taumax/taumin

        N[i] = floor(N0/2)

        for j in 1:reps
            resultFGG[j, i] = @elapsed NUFFTcorrFKFGG(P, t)
            resultMRS[j, i] = @elapsed MScorrFK(P, t)
        end
        GC.gc()
    end
    return resultFGG, resultMRS, n1, n2, N
end

#---------------------------------------------------------------------------
# Compute results

lambdarange = collect(30:10:100)
reps = 10

# Compute
Dirichlet = timeDK(10000, lambdarange, reps)
Fejer = timeFK(10000, lambdarange, reps)

# Save
save("Computed Data/Asynchrony times/DirAsyn.jld", "Dirichlet", Dirichlet)
save("Computed Data/Asynchrony times/FejAsyn.jld", "Fejer", Fejer)

# Load
Dirichlet = load("Computed Data/Asynchrony times/DirAsyn.jld")
Dirichlet = Dirichlet["Dirichlet"]
Fejer = load("Computed Data/Asynchrony times/FejAsyn.jld")
Fejer = Fejer["Fejer"]

# Make table
tab = DataTable(n1 = Dirichlet[3], n2 = Dirichlet[4], N = Dirichlet[5], lambda = lambdarange,
                DFGG = minimum(Dirichlet[1], dims=1), DMRS = minimum(Dirichlet[2], dims=1),
                FFGG = minimum(Fejer[1], dims=1), FMRS = minimum(Fejer[2], dims=1))

dt = DataTable(Fish = ["Suzy", "Amir"], Mass = [1.5, 2])
