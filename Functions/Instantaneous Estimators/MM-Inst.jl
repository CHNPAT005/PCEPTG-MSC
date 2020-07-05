## Author: Patrick Chang
# Script file for the MM NUFFT
# Supporting Algorithms are at the start of the script
#  Include:
#           - Scale function to re-scale time to [0, 2 \pi]
#           - Fast Gaussian Gridding [GL] (Own implementaion)
# Number of Fourier Coefficients used is length of data

## Implementation uses Fast Gaussian Gridding

#---------------------------------------------------------------------------

### Data Format:
## p = [n x m] matrix of prices, log returns are computed in the function
# non-trading times are indicated by NaNs
## t = [n x m] matrix of trading times, non-trading times are indicated by NaNs
# dimensions of p and t must match.
## N = Optional input for cutoff frequency
## tol = error tolerance for NUFFT - determines how much spreading, default = 10^-12

#---------------------------------------------------------------------------

using ArgCheck, LinearAlgebra

#---------------------------------------------------------------------------
### Supporting functions

# cd("/Users/patrickchang1/PCEPTG-MM-NUFFT")
include("../NUFFT/NUFFT-FGG")

function scale(t)
    maxt = maximum(filter(!isnan, t))
    mint = minimum(filter(!isnan, t))

    tau = (2*pi) .* (t .- mint) ./ (maxt - mint)
    return tau
end

#---------------------------------------------------------------------------

# Non-uniform Fast Fourier Transform implementaion of the Dirichlet Kernel

function MM_inst(p, t, outlength; kwargs...)
    ## Pre-allocate arrays and check Data
    np = size(p)[1]
    mp = size(p)[2]
    nt = size(t)[1]

    # @argcheck size(p) == size(t) DimensionMismatch

    # Re-scale trading times
    tau = scale(t)
    # Computing minimum time change
    dtau = zeros(mp,1)
    for i in 1:mp
        dtau[i] = minimum(diff(filter(!isnan, tau[:,i])))
    end
    # maximum of minumum step size to avoid aliasing
    taumin = maximum(dtau)
    taumax = 2*pi
    # Sampling Freq.
    N0 = taumax/taumin

    #------------------------------------------------------

    # Optional Cutoff - if not specified we use Nyquist Cutoff
    kwargs = Dict(kwargs)

    if haskey(kwargs, :N)
        N = kwargs[:N]
    else
        N = Int(floor(N0/2))
    end

    if haskey(kwargs, :T)
        T = kwargs[:T]
    else
        T = 1
    end

    if haskey(kwargs, :M)
        M = kwargs[:M]
    else
        n = np-1
        M = Int(floor(sqrt(n)*log(n)/(8*2*pi)))
    end

    if haskey(kwargs, :tol)
        tol = kwargs[:tol]
    else
        tol = 10^-12
    end

    # if input is 1D return 1D solutions
    # if mp == 1
    #     t = (t .- minimum(t)) .* (1 / (maximum(t) - minimum(t)))
    #     return MM_inst1D(p, t, outlength, N, M, T)
    # end

    #------------------------------------------------------

    ks = collect(-N-M:1:N+M)
    Den = length(ks)

    #------------------------------------------------------
    c_ks = zeros(ComplexF64, mp, Den)

    for i in 1:mp
        psii = findall(!isnan, p[:,i])
        P = p[psii, i]
        Time = tau[psii, i]
        DiffP = complex(diff(log.(P)))
        Time = Time[1:(end-1)]

        C = NUFFTFGG(DiffP, Time, Den, tol)
        # C = conj(C) ./ T

        c_ks[i,:] = [C[Int(ceil(Den/2))+1:end]; C[1:Int(ceil(Den/2))]]
    end

    #------------------------------------------------------
    α_11 = zeros(ComplexF64, 1, 2*M+1)
    α_12 = zeros(ComplexF64, 1, 2*M+1)
    α_22 = zeros(ComplexF64, 1, 2*M+1)

    # cent = Int(ceil(Den/2))
    cent = N + M + 1

    for k in -M:M
        α_11[k+M+1] = 0.0
        α_12[k+M+1] = 0.0
        α_22[k+M+1] = 0.0
        for l in -N:N
            α_11[k+M+1] = α_11[k+M+1] + (c_ks[1,l+cent] * c_ks[1,k-l+cent]) * (T / (2*N+1))
            α_12[k+M+1] = α_12[k+M+1] + (c_ks[1,l+cent] * c_ks[2,k-l+cent]) * (T / (2*N+1))
            α_22[k+M+1] = α_22[k+M+1] + (c_ks[2,l+cent] * c_ks[2,k-l+cent]) * (T / (2*N+1))
        end
    end

    #------------------------------------------------------

    k = collect(-M:1:M)
    tt = collect(0:1/outlength:1-(1/outlength))
    con = 2*pi/T
    σ_11 = zeros(ComplexF64, outlength, 1)
    σ_12 = zeros(ComplexF64, outlength, 1)
    σ_22 = zeros(ComplexF64, outlength, 1)

    for i in 1:outlength
        σ_11[i] = 0.0
        for k in -M:M
            σ_11[i] += (1-abs(k)/M)*α_11[k+M+1]*exp(1im*tt[i]*con*k)
            σ_12[i] += (1-abs(k)/M)*α_12[k+M+1]*exp(1im*tt[i]*con*k)
            σ_22[i] += (1-abs(k)/M)*α_22[k+M+1]*exp(1im*tt[i]*con*k)
        end
    end

    return real(σ_11), real(σ_22), real(σ_12)
end

function MM_inst1D(p, t, outlength, N, M, T=1)
    n = length(p); con = 2*pi/T;
    c_pp = zeros(ComplexF64, N+M,1); c_p = zeros(ComplexF64, 2*N+2*M+1,1)
    r = diff(log.(p), dims = 1); c_s = zeros(ComplexF64, 2*M+1,1)
    c_0 = sum(r)

    for k in 1:N+M
        c_pp[k] = sum(exp.((-1im*con*k) .* t[1:end-1]) .* r)
    end
    for j in 1:N+M
        c_p[j] = conj(c_pp[N+M+1-j]) / T
    end
    c_p[N+M+1]=c_0/T
    for j in 1:N+M
        c_p[N+M+1+j] = c_pp[j] / T
    end
    fact = T / (2*N+1)
    nshift = N + M + 1
    for k in -M:M
        c_s[k+M+1] = 0.0
        for l in -N:N
            c_s[k+M+1] += fact*(c_p[l+nshift]*c_p[k-l+nshift])
        end
    end
    tt = collect(0:1/outlength:T-(1/outlength))
    σ_11 = zeros(ComplexF64, outlength, 1)
    for i in 1:outlength
        σ_11[i] = 0.0
        for k in -M:M
            σ_11[i] += (1-abs(k)/M)*c_s[k+M+1]*exp(1im*tt[i]*con*k)
        end
    end
    return real(σ_11)
end
