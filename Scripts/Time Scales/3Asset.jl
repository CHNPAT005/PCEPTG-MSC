## Author: Patrick Chang
# Script file to compare the averaging scale of the MM estimator
# using pathological correlation choices.

using JLD, LaTeXStrings, Plots, Statistics, Distributions

#---------------------------------------------------------------------------

cd("/Users/patrickchang1/PCEPTG-MSC")

include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")

include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FGG.jl")

include("../../Functions/SDEs/GBM.jl")

#---------------------------------------------------------------------------

function rexp(n, mean)
    t = -mean .* log.(rand(n))
end

dt = collect(1:1:100)
reps = 100

#---------------------------------------------------------------------------

function MMZmodel2(T, dt, lam, reps)
    N = Int.(floor.(((T ./dt).-1.0) ./ 2))
    # N = Int.(floor.(86400 ./ ((1) .*dt)))
    lam2 = 1/lam

    DKres12 = zeros(reps, length(N))
    FKres12 = zeros(reps, length(N))
    DKres13 = zeros(reps, length(N))
    FKres13 = zeros(reps, length(N))
    DKres23 = zeros(reps, length(N))
    FKres23 = zeros(reps, length(N))

    for j in 1:reps
        P = GBM(T, mu, sigma, seed = j)
        t = [collect(1:1:T) collect(1:1:T) collect(1:1:T)]

        Random.seed!(j)
        t1 = [1; rexp(T, lam)]
        t1 = cumsum(t1)
        t1 = filter((x) -> x < T, t1)

        Random.seed!(j+reps)
        t2 = [1; rexp(T, lam)]
        t2 = cumsum(t2)
        t2 = filter((x) -> x < T, t2)

        Random.seed!(j+reps+reps)
        t3 = [1; rexp(T, lam)]
        t3 = cumsum(t3)
        t3 = filter((x) -> x < T, t3)

        p1 = P[Int.(floor.(t1)), 1]
        p2 = P[Int.(floor.(t2)), 2]
        p3 = P[Int.(floor.(t3)), 3]

        D = maximum([length(t1); length(t2); length(t3)])
        P = fill(NaN, (D, 3))
        t = fill(NaN, (D, 3))

        P[1:length(p1), 1] = p1
        P[1:length(p2), 2] = p2
        P[1:length(p3), 3] = p3

        t[1:length(t1), 1] = t1
        t[1:length(t2), 2] = t2
        t[1:length(t3), 3] = t3

        for i in 1:length(N)
            DKres = NUFFTcorrDKFGG(P, t, N = N[i])[1]
            FKres = NUFFTcorrFKFGG(P, t, N = N[i])[1]

            DKres12[j, i] = DKres[1,2]
            FKres12[j, i] = FKres[1,2]
            DKres13[j, i] = DKres[1,3]
            FKres13[j, i] = FKres[1,3]
            DKres23[j, i] = DKres[2,3]
            FKres23[j, i] = FKres[2,3]
        end
    end
    # model = 0.35 .* (1 .+ (exp.(-lam2 .* dt) .- 1) ./ (lam2 .* dt))
    return DKres12, FKres12, DKres13, FKres13, DKres23, FKres23
end

#---------------------------------------------------------------------------

mu = repeat([0.01/86400], 3)
sig = repeat([sqrt(0.1/86400)], 3)
rho = [1 -0.5 0.7;
        -0.5 1 0.01;
        0.7 0.01 1]
sigma = zeros(3,3)
for i in 1:3, j in 1:3
    sigma[i,j] = rho[i,j] * sig[i] * sig[j]
end

res = MMZmodel2(60*60*8, dt, 5, reps)

save("Computed Data/TimeScale/3Asset1.jld", "res", res)

res = load("Computed Data/TimeScale/3Asset1.jld")
res = res["res"]


p1 = plot(dt, mean(res[1], dims = 1)', ribbon = std(res[1], dims = 1)' * quantile.(TDist(reps-1), [0.84]),
        fillalpha=.1, label = L"\rho = -0.5", legend = :bottomright, legendtitle="Induced",
        dpi = 300, ylims = (-0.7, 0.8), marker=([:x :d],3,1,stroke(2,:blue)), color = :blue)
plot!(p1, dt, mean(res[3], dims = 1)', ribbon=std(res[3], dims = 1)' * quantile.(TDist(reps-1), [0.84]),
        fillalpha=.1, label = L"\rho = 0.7", marker=([:circ :d],3,1,stroke(2,:red)), color = :red)
plot!(p1, dt, mean(res[5], dims = 1)', ribbon=std(res[5], dims = 1)' * quantile.(TDist(reps-1), [0.84]),
        fillalpha=.1, label = L"\rho = 0.01", marker=([:+ :d],3,1,stroke(2,:green)), color = :green)
ylabel!(p1, L"\rho_{\Delta t}^{ij}")
xlabel!(p1, L"\Delta t \textrm{ [sec]}")

savefig(p1, "Plots/MM-NUFFT/3Asset1DK.svg")


#---------------------------------------------------------------------------

sig = repeat([sqrt(0.1/86400)], 3)
rho = [1 0.5 0.7;
        0.5 1 0.01;
        0.7 0.01 1]
sigma = zeros(3,3)
for i in 1:3, j in 1:3
    sigma[i,j] = rho[i,j] * sig[i] * sig[j]
end

res2 = MMZmodel2(60*60*8, dt, 5, reps)

save("Computed Data/TimeScale/3Asset2.jld", "res2", res2)

res = load("Computed Data/TimeScale/3Asset2.jld")
res = res["res2"]

p1 = plot(dt, mean(res[1], dims = 1)', ribbon = std(res[1], dims = 1)' * quantile.(TDist(reps-1), [0.84]),
        fillalpha=.1, label = L"\rho = -0.5", legend = :bottomright, legendtitle="Induced",
        dpi = 300, ylims = (-0.7, 0.8), marker=([:x :d],3,1,stroke(2,:blue)), color = :blue)
plot!(p1, dt, mean(res[3], dims = 1)', ribbon=std(res[3], dims = 1)' * quantile.(TDist(reps-1), [0.84]),
        fillalpha=.1, label = L"\rho = 0.7", marker=([:circ :d],3,1,stroke(2,:red)), color = :red)
plot!(p1, dt, mean(res[5], dims = 1)', ribbon=std(res[5], dims = 1)' * quantile.(TDist(reps-1), [0.84]),
        fillalpha=.1, label = L"\rho = 0.01", marker=([:+ :d],3,1,stroke(2,:green)), color = :green)
ylabel!(p1, L"\rho_{\Delta t}^{ij}")
xlabel!(p1, L"\Delta t \textrm{ [sec]}")

savefig(p1, "Plots/MM-NUFFT/3Asset2DK.svg")
