## Author: Patrick Chang
# Script file to test if the desired level of error
# is achieved using our implementaion of the NUFFT methods

# Error is measured with respect to the l2 error of the
# Fourier Coefficients

#---------------------------------------------------------------------------

using LinearAlgebra

#---------------------------------------------------------------------------

include("../Functions/NUFFT/NUFFT-FGG.jl")
include("../Functions/NUFFT/NUFFT-KB.jl")
include("../Functions/NUFFT/NUFFT-ES.jl")

n = collect(-3:-1:-14)
tol = 10.0.^n

function testErrors(n, M, tolrange, error)
    tol = tolrange

    FGGerror = zeros(length(tol), 1)
    KBerror = zeros(length(tol), 1)
    ESerror = zeros(length(tol), 1)

    nj = n
    x = (collect(0:nj-1) + 0.5 .* rand(nj))
    xj = (x .- minimum(x)) .* (2*pi / (maximum(x) - minimum(x)))
    cj = rand(nj) + 0im*rand(nj)
    xjj = xj ./ (2*pi)

    M = M
    Mr = 2*M
    freq = fftfreq(Mr, 1) .* Mr
    pos = findall(abs.(freq) .<= M/2)
    k = freq[pos]

    ref = (cj' * exp.(1im .* xj * k'))'

    for i in 1:length(tol)
        FGG = NUFFTFGG(cj, xj, M, tol[i])
        KB = NUFFTKB(cj, xjj, M, tol[i])
        ES = NUFFTES(cj, xj, M, tol[i])
        relerrFGG = norm(FGG-ref, error) / norm(ref, error)
        relerrKB = norm(KB-ref, error) / norm(ref, error)
        relerrES = norm(ES-ref, error) / norm(ref, error)

        if relerrFGG <= tol[i]
            FGGerror[i] = 1
        else
            FGGerror[i] = 0
        end

        if relerrKB <= tol[i]
            KBerror[i] = 1
        else
            KBerror[i] = 0
        end

        if relerrES <= tol[i]
            ESerror[i] = 1
        else
            ESerror[i] = 0
        end
    end

    if Int(sum(FGGerror))==length(tol) && Int(sum(KBerror))==length(tol) && Int(sum(ESerror))==length(tol)
        print("All passed")
    elseif Int(sum(FGGerror))<length(tol)
        print("FGG failed")
    elseif Int(sum(KBerror))<length(tol)
        print("KB failed")
    elseif Int(sum(ESerror))<length(tol)
        print("ES failed")
    end

end

# l_2 error
testErrors(100, 51, tol, 2)

# l_∞ error
testErrors(100, 51, tol, Inf)

# When M_{sp} for KB = Int(floor((ceil(log(10, 1/tol)) + 2)/2)),
# it passes both error tests.

# When M_{sp} for KB = Int(floor((ceil(log(10, 1/tol)) + 1)/2)),
# it doesnt pass l_2 error, but nearly passes l_∞ error.
