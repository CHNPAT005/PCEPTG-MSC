## Author: Patrick Chang
# Script file for the MM Jump Robust estimator
# by Cuchiero and Teichmann 2015
# Supporting Algorithms are at the start of the script
#  Include:
#           - Fejer computation
#           - hat_ρ

#---------------------------------------------------------------------------

### Data Format:
## p = [n x 2] matrix of prices, prices MUST be synchronous.
## N = Cutoff Frequency

#---------------------------------------------------------------------------

using ArgCheck, LinearAlgebra

#---------------------------------------------------------------------------

function Fejer(x, N)
    num = sin((N+1)*x/2)^2
    den = sin(x/2)^2
    if x == 0
        return N
    else
        return (num/den) * (1/(N+1))
    end
end

# function hat_ρ(dj, N, n, T=1)
#     M = 2*N+1
#     res = zeros(M, 1)
#     gj = cos.(sqrt(n).*dj)
#     t = collect(0:1/n:T-(1/n)) #./ T
#     tt = collect(0:1/M:T-(1/M)) #./ T
#
#     for i in 1:M
#         res[i] = sum(Fejer.((2*pi/T).*(tt[i] .- t), N) .* gj) / (n*T)
#     end
#     return res
# end

function hat_ρ(dj, N, n, outlength, T=1)
    res = zeros(outlength, 1)
    gj = cos.(sqrt(n).*dj)
    t = collect(0:1/n:T-(1/n)) #./ T
    tt = collect(0:1/outlength:T-(1/outlength)) #./ T

    for i in 1:length(tt)
        res[i] = sum(Fejer.((2*pi/T).*(tt[i] .- t), N) .* gj) / (n*T)
    end
    return res
end

function MM_JR(p, N, outlength, T=1)
    n = size(p)[1] - 1
    y1 = log.(p[:,1])
    y2 = log.(p[:,2])

    dj_1 = diff(y1, dims = 1)
    dj_2 = diff(y2, dims = 1)
    dj_12 = diff(y1+y2, dims = 1)

    rho_1 = hat_ρ(dj_1, N, n, outlength, T)
    rho_2 = hat_ρ(dj_2, N, n, outlength, T)
    rho_12 = hat_ρ(dj_12, N, n, outlength, T)

    sighat_1 = -2 .* log.(real(rho_1))
    sighat_2 = -2 .* log.(real(rho_2))
    sighat_12 = 0.5 .* ( (-2 .* log.(real(rho_12))) - sighat_1 - sighat_2 )

    return sighat_1, sighat_2, sighat_12
end
