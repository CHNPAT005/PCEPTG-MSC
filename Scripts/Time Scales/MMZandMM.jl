## Author: Patrick Chang
# Script file to compare the averaging scale of the MM estimator
# against the model MMZ used to model the Epps effect.

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

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

#---------------------------------------------------------------------------

dt = collect(1:1:100)
# N = Int.(floor.(86400 ./ ((2) .*dt)))

function MMZmodel(T, dt, lam, reps)
    N = Int.(floor.(((T ./dt).-1.0) ./ 2))
    # N = Int.(floor.(86400 ./ ((1) .*dt)))
    lam2 = 1/lam

    DKres = zeros(reps, length(N))
    FKres = zeros(reps, length(N))

    for j in 1:reps
        P = GBM(T, mu, sigma, seed = j)
        t = reshape([collect(1:1:T); collect(1:1:T)], T, 2)

        Random.seed!(j)
        t1 = [1; rexp(T, lam)]
        t1 = cumsum(t1)
        t1 = filter((x) -> x < T, t1)

        Random.seed!(j+reps)
        t2 = [1; rexp(T, lam)]
        t2 = cumsum(t2)
        t2 = filter((x) -> x < T, t2)

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

        for i in 1:length(N)
            DKres[j, i] = NUFFTcorrDKFGG(P, t, N = N[i])[1][1,2]
            FKres[j, i] = NUFFTcorrFKFGG(P, t, N = N[i])[1][1,2]
        end
    end
    model = 0.35 .* (1 .+ (exp.(-lam2 .* dt) .- 1) ./ (lam2 .* dt))
    return DKres, FKres, model
end

reps = 100

#---------------------------------------------------------------------------
## 1 Hour of data
# lam = 5
T1h = 60*60
lam1h5 = MMZmodel(T1h, dt, 5, reps)

save("Computed Data/TimeScale/lam1h5.jld", "lam1h5", lam1h5)

lam1h5 = load("Computed Data/TimeScale/lam1h5.jld")
lam1h5 = lam1h5["lam1h5"]

q = quantile.(TDist(reps-1), [0.84])
erD = q .* std(lam1h5[1], dims = 1)
erF = q .* std(lam1h5[2], dims = 1)

p1 = plot(dt, mean(lam1h5[1], dims = 1)', ribbon=erD', fillalpha=.1, label = L"\textrm{MM Dirichlet}", legend = :bottomright, legendtitle="Method",
    color = :blue, dpi = 300, ylims = (-0.1, 0.7), marker=([:x :d],3,1,stroke(2,:blue)))
plot!(p1, dt, mean(lam1h5[2], dims = 1)', ribbon=erF', fillalpha=.1, label = L"\textrm{MM Fej\'{e}r}", color = :red, marker=([:+ :d],3,1,stroke(2,:red)))
plot!(p1, dt, lam1h5[3], label = L"\textrm{Theoretical}", line = (2, :solid), color = :black)
ylabel!(p1, L"\rho(\Delta t)")
xlabel!(p1, L"\Delta t \textrm{ [sec]}")

savefig(p1, "Plots/MM-NUFFT/AveScaleLam5_1h.svg")


# lam = 20
lam1h20 = MMZmodel(T1h, dt, 20, reps)

save("Computed Data/TimeScale/lam1h20.jld", "lam1h20", lam1h20)

lam1h20 = load("Computed Data/TimeScale/lam1h20.jld")
lam1h20 = lam1h20["lam1h20"]

q = quantile.(TDist(reps-1), [0.84])
erD = q .* std(lam1h20[1], dims = 1)
erF = q .* std(lam1h20[2], dims = 1)

p1 = plot(dt, mean(lam1h20[1], dims = 1)', ribbon=erD', fillalpha=.1, label = L"\textrm{MM Dirichlet}", legend = :bottomright, legendtitle="Method",
    color = :blue, dpi = 300, ylims = (-0.1, 0.7), marker=([:x :d],3,1,stroke(2,:blue)))
plot!(p1, dt, mean(lam1h20[2], dims = 1)', ribbon=erF', fillalpha=.1, label = L"\textrm{MM Fej\'{e}r}", color = :red, marker=([:+ :d],3,1,stroke(2,:red)))
plot!(p1, dt, lam1h20[3], label = L"\textrm{Theoretical}", line = (2, :solid), color = :black)
ylabel!(p1, L"\rho(\Delta t)")
xlabel!(p1, L"\Delta t \textrm{ [sec]}")

savefig(p1, "Plots/MM-NUFFT/AveScaleLam20_1h.svg")


#---------------------------------------------------------------------------
## 1 Trading day of data
# lam = 5
T1d = 60*60*8
lam1d5 = MMZmodel(T1d, dt, 5, reps)

save("Computed Data/TimeScale/lam1d5.jld", "lam1d5", lam1d5)

lam1d5 = load("Computed Data/TimeScale/lam1d5.jld")
lam1d5 = lam1d5["lam1d5"]

q = quantile.(TDist(reps-1), [0.84])
erD = q .* std(lam1d5[1], dims = 1)
erF = q .* std(lam1d5[2], dims = 1)

p1 = plot(dt, mean(lam1d5[1], dims = 1)', ribbon=erD', fillalpha=.1, label = L"\textrm{MM Dirichlet}", legend = :bottomright, legendtitle="Method",
    color = :blue, dpi = 300, ylims = (-0.1, 0.7), marker=([:x :d],3,1,stroke(2,:blue)))
plot!(p1, dt, mean(lam1d5[2], dims = 1)', ribbon=erF', fillalpha=.1, label = L"\textrm{MM Fej\'{e}r}", color = :red, marker=([:+ :d],3,1,stroke(2,:red)))
plot!(p1, dt, lam1d5[3], label = L"\textrm{Theoretical}", line = (2, :solid), color = :black)
ylabel!(p1, L"\rho(\Delta t)")
xlabel!(p1, L"\Delta t \textrm{ [sec]}")

savefig(p1, "Plots/MM-NUFFT/AveScaleLam5_1d.svg")


# lam = 20
lam1d20 = MMZmodel(T1d, dt, 20, reps)

save("Computed Data/TimeScale/lam1d20.jld", "lam1d20", lam1d20)

lam1d20 = load("Computed Data/TimeScale/lam1d20.jld")
lam1d20 = lam1d20["lam1d20"]

q = quantile.(TDist(reps-1), [0.84])
erD = q .* std(lam1d20[1], dims = 1)
erF = q .* std(lam1d20[2], dims = 1)

p1 = plot(dt, mean(lam1d20[1], dims = 1)', ribbon=erD', fillalpha=.1, label = L"\textrm{MM Dirichlet}", legend = :bottomright, legendtitle="Method",
    color = :blue, dpi = 300, ylims = (-0.1, 0.7), marker=([:x :d],3,1,stroke(2,:blue)))
plot!(p1, dt, mean(lam1d20[2], dims = 1)', ribbon=erF', fillalpha=.1, label = L"\textrm{MM Fej\'{e}r}", color = :red, marker=([:+ :d],3,1,stroke(2,:red)))
plot!(p1, dt, lam1d20[3], label = L"\textrm{Theoretical}", line = (2, :solid), color = :black)
ylabel!(p1, L"\rho(\Delta t)")
xlabel!(p1, L"\Delta t \textrm{ [sec]}")

savefig(p1, "Plots/MM-NUFFT/AveScaleLam20_1d.svg")


#---------------------------------------------------------------------------
## 1 Trading week of data
# lam = 5
T1w = 60*60*8*5
lam1w5 = MMZmodel(T1w, dt, 5, reps)

save("Computed Data/TimeScale/lam1w5.jld", "lam1w5", lam1w5)

lam1w5 = load("Computed Data/TimeScale/lam1w5.jld")
lam1w5 = lam1w5["lam1w5"]

q = quantile.(TDist(reps-1), [0.84])
erD = q .* std(lam1w5[1], dims = 1)
erF = q .* std(lam1w5[2], dims = 1)

p1 = plot(dt, mean(lam1w5[1], dims = 1)', ribbon=erD', fillalpha=.1, label = L"\textrm{MM Dirichlet}", legend = :bottomright, legendtitle="Method",
    color = :blue, dpi = 300, ylims = (-0.1, 0.7), marker=([:x :d],3,1,stroke(2,:blue)))
plot!(p1, dt, mean(lam1w5[2], dims = 1)', ribbon=erF', fillalpha=.1, label = L"\textrm{MM Fej\'{e}r}", color = :red, marker=([:+ :d],3,1,stroke(2,:red)))
plot!(p1, dt, lam1w5[3], label = L"\textrm{Theoretical}", line = (2, :solid), color = :black)
ylabel!(p1, L"\rho(\Delta t)")
xlabel!(p1, L"\Delta t \textrm{ [sec]}")

savefig(p1, "Plots/MM-NUFFT/AveScaleLam5_1w.svg")

# lam = 20
lam1w20 = MMZmodel(T1w, dt, 20, reps)

save("Computed Data/TimeScale/lam1w20.jld", "lam1w20", lam1w20)

lam1w20 = load("Computed Data/TimeScale/lam1w20.jld")
lam1w20 = lam1w20["lam1w20"]

q = quantile.(TDist(reps-1), [0.84])
erD = q .* std(lam1w20[1], dims = 1)
erF = q .* std(lam1w20[2], dims = 1)

p1 = plot(dt, mean(lam1w20[1], dims = 1)', ribbon=erD', fillalpha=.1, label = L"\textrm{MM Dirichlet}", legend = :bottomright, legendtitle="Method",
    color = :blue, dpi = 300, ylims = (-0.1, 0.7), marker=([:x :d],3,1,stroke(2,:blue)))
plot!(p1, dt, mean(lam1w20[2], dims = 1)', ribbon=erF', fillalpha=.1, label = L"\textrm{MM Fej\'{e}r}", color = :red, marker=([:+ :d],3,1,stroke(2,:red)))
plot!(p1, dt, lam1w20[3], label = L"\textrm{Theoretical}", line = (2, :solid), color = :black)
ylabel!(p1, L"\rho(\Delta t)")
xlabel!(p1, L"\Delta t \textrm{ [sec]}")

savefig(p1, "Plots/MM-NUFFT/AveScaleLam20_1w.svg")


#---------------------------------------------------------------------------
# test plot of all paths
p1 = plot(dt,lam1h20[1]', legend = false)
plot!(p1, dt, lam1h20[3], color = :black, line = (2, :dash))
title!(p1, L"\textrm{\sffamily (f) Corr and sampling interval (} \lambda=1/20 \textrm{\sffamily , T = 144000, Dir)}")
ylabel!(p1, L"\textrm{Correlation } \rho")
xlabel!(p1, L"\textrm{Sampling interval } \Delta t \textrm{ [seconds]}")

savefig(p1, "Plots/AveScaleLam20_1w_full.pdf")
