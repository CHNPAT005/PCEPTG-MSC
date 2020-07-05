## Author: Patrick Chang
# Script file for the local realized variance estimator
# Supporting Algorithms are at the start of the script
#  Include:
#           - Fejer computation
#           - hat_ρ

#---------------------------------------------------------------------------

### Data Format:
## p = [n x 2] matrix of prices, prices MUST be synchronous.
## N = Cutoff Frequency

#---------------------------------------------------------------------------

using ArgCheck, LinearAlgebra, Intervals

#---------------------------------------------------------------------------

function ϵ_t(t, j, N, T)
    i = Interval((j-1)*T/N, j*T/N, true, false)
    if in(t, i)
        return 1/sqrt(T/N)
    else
        return 0
    end
end

function hat_X(dj, N, n, outlength, T=1)
    res = zeros(outlength, 1)
    t = collect(0:1/n:T-(1/n))
    tt = collect(0:1/outlength:T-(1/outlength))

    # gj = dj.^2

    for i in 1:outlength
        ϵ = 0.0
        for j in 1:N
            # inner = sum(gj .* ϵ_t.(t, j, N, 1))
            inner = 0.0
            for m in 1:n*T
                inner += dj[m]^2 * ϵ_t(t[m], j, N, T)
            end
            ϵ += inner * ϵ_t(tt[i], j, N, T)
        end
        res[i] = ϵ
    end
    return res
end

function LRV(p, N, outlength, T=1)
    n = size(p)[1] - 1
    y1 = log.(p[:,1])
    y2 = log.(p[:,2])

    dj_1 = diff(y1, dims = 1)
    dj_2 = diff(y2, dims = 1)
    dj_12 = diff(y1+y2, dims = 1)

    sighat_1 = hat_X(dj_1, N, n, outlength, T)
    sighat_2 = hat_X(dj_2, N, n, outlength, T)
    sighat_12 = 0.5 .* ( hat_X(dj_12, N, n, outlength, T) - sighat_1 - sighat_2 )

    return sighat_1, sighat_2, sighat_12
end
