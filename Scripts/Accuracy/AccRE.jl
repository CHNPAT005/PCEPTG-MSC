## Author: Patrick Chang
# Script file to test the impact of ϵ on the correlation estimate
# for various averaging kernels using the Vectorized code as a baseline.
# We investigate the arrival-time representation case.

#---------------------------------------------------------------------------

using LinearAlgebra, LaTeXStrings, StatsBase, Random, Statistics, Distributions
using ProgressMeter, JLD, Plots

#---------------------------------------------------------------------------

cd("/Users/patrickchang1/PCEPTG-MSC")

include("../../Functions/Correlation Estimators/Dirichlet/CFTcorrDK.jl")
include("../../Functions/Correlation Estimators/Dirichlet/FFTZPcorrDK.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FINUFFT.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-KB.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-ES.jl")

include("../../Functions/Correlation Estimators/Fejer/CFTcorrFK.jl")
include("../../Functions/Correlation Estimators/Fejer/FFTZPcorrFK.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FINUFFT.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FGG.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-KB.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-ES.jl")

include("../../Functions/SDEs/GBM.jl")

#---------------------------------------------------------------------------
## Asynchronous Case - Random Exponential
#---------------------------------------------------------------------------

function rexp(n, mean)
    t = -mean .* log.(rand(n))
end

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

#---------------------------------------------------------------------------

## Nyquist - varying with each replication
n1 = collect(-1:-1:-14)
tol = 10.0.^n1
reps = 100

# Dirichlet
CFTaccDKRENyq = zeros(1, reps)
ZFFTaccDKRENyq = zeros(1, reps)

FGGaccDKRENyq = zeros(length(n1), reps)
KBaccDKRENyq = zeros(length(n1), reps)
FINUFFTaccDKRENyq = zeros(length(n1), reps)
ESaccDKRENyq = zeros(length(n1), reps)

# Fejer
CFTaccFKRENyq = zeros(1, reps)
ZFFTaccFKRENyq = zeros(1, reps)

FGGaccFKRENyq = zeros(length(n1), reps)
KBaccFKRENyq = zeros(length(n1), reps)
FINUFFTaccFKRENyq = zeros(length(n1), reps)
ESaccFKRENyq = zeros(length(n1), reps)

# Compute
@showprogress "Computing..." for j in 1:reps
    P = GBM(10000, mu, sigma, seed = j)
    t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

    Random.seed!(j)
    t1 = [1; rexp(10000, 30)]
    t1 = cumsum(t1)
    t1 = filter((x) -> x < 10000, t1)

    Random.seed!(j+reps)
    t2 = [1; rexp(10000, 45)]
    t2 = cumsum(t2)
    t2 = filter((x) -> x < 10000, t2)

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

    CFTaccDKRENyq[j] = CFTcorrDK(P, t)[1][1,2]
    ZFFTaccDKRENyq[j] = FFTZPcorrDK(P, t)[1][1,2]

    CFTaccFKRENyq[j] = CFTcorrFK(P, t)[1][1,2]
    ZFFTaccFKRENyq[j] = FFTZPcorrFK(P, t)[1][1,2]

    for i in 1:length(n1)
        FGGaccDKRENyq[i, j] = NUFFTcorrDKFGG(P, t, tol = tol[i])[1][1,2]
        KBaccDKRENyq[i, j] = NUFFTcorrDKKB(P, t, tol = tol[i])[1][1,2]
        FINUFFTaccDKRENyq[i, j] = NUFFTcorrDKFINUFFT(P, t, tol = tol[i])[1][1,2]
        ESaccDKRENyq[i, j] = NUFFTcorrDKES(P, t, tol = tol[i])[1][1,2]

        FGGaccFKRENyq[i, j] = NUFFTcorrFKFGG(P, t, tol = tol[i])[1][1,2]
        KBaccFKRENyq[i, j] = NUFFTcorrFKKB(P, t, tol = tol[i])[1][1,2]
        FINUFFTaccFKRENyq[i, j] = NUFFTcorrFKFINUFFT(P, t, tol = tol[i])[1][1,2]
        ESaccFKRENyq[i, j] = NUFFTcorrFKES(P, t, tol = tol[i])[1][1,2]
    end
end

# Save
save("Computed Data/Accuracy/CFTaccDKRENyq.jld", "CFTaccDKRENyq", CFTaccDKRENyq)
save("Computed Data/Accuracy/ZFFTaccDKRENyq.jld", "ZFFTaccDKRENyq", ZFFTaccDKRENyq)
save("Computed Data/Accuracy/FGGaccDKRENyq.jld", "FGGaccDKRENyq", FGGaccDKRENyq)
save("Computed Data/Accuracy/KBaccDKRENyq.jld", "KBaccDKRENyq", KBaccDKRENyq)
save("Computed Data/Accuracy/FINUFFTaccDKRENyq.jld", "FINUFFTaccDKRENyq", FINUFFTaccDKRENyq)
save("Computed Data/Accuracy/ESaccDKRENyq.jld", "ESaccDKRENyq", ESaccDKRENyq)

save("Computed Data/Accuracy/CFTaccFKRENyq.jld", "CFTaccFKRENyq", CFTaccFKRENyq)
save("Computed Data/Accuracy/ZFFTaccFKRENyq.jld", "ZFFTaccFKRENyq", ZFFTaccFKRENyq)
save("Computed Data/Accuracy/FGGaccFKRENyq.jld", "FGGaccFKRENyq", FGGaccFKRENyq)
save("Computed Data/Accuracy/KBaccFKRENyq.jld", "KBaccFKRENyq", KBaccFKRENyq)
save("Computed Data/Accuracy/FINUFFTaccFKRENyq.jld", "FINUFFTaccFKRENyq", FINUFFTaccFKRENyq)
save("Computed Data/Accuracy/ESaccFKRENyq.jld", "ESaccFKRENyq", ESaccFKRENyq)

# Load
CFTaccDKRENyq = load("Computed Data/Accuracy/CFTaccDKRENyq.jld")
CFTaccDKRENyq = CFTaccDKRENyq["CFTaccDKRENyq"]
ZFFTaccDKRENyq = load("Computed Data/Accuracy/ZFFTaccDKRENyq.jld")
ZFFTaccDKRENyq = ZFFTaccDKRENyq["ZFFTaccDKRENyq"]
FGGaccDKRENyq = load("Computed Data/Accuracy/FGGaccDKRENyq.jld")
FGGaccDKRENyq = FGGaccDKRENyq["FGGaccDKRENyq"]
KBaccDKRENyq = load("Computed Data/Accuracy/KBaccDKRENyq.jld")
KBaccDKRENyq = KBaccDKRENyq["KBaccDKRENyq"]
FINUFFTaccDKRENyq = load("Computed Data/Accuracy/FINUFFTaccDKRENyq.jld")
FINUFFTaccDKRENyq = FINUFFTaccDKRENyq["FINUFFTaccDKRENyq"]
ESaccDKRENyq = load("Computed Data/Accuracy/ESaccDKRENyq.jld")
ESaccDKRENyq = ESaccDKRENyq["ESaccDKRENyq"]

CFTaccFKRENyq = load("Computed Data/Accuracy/CFTaccFKRENyq.jld")
CFTaccFKRENyq = CFTaccFKRENyq["CFTaccFKRENyq"]
ZFFTaccFKRENyq = load("Computed Data/Accuracy/ZFFTaccFKRENyq.jld")
ZFFTaccFKRENyq = ZFFTaccFKRENyq["ZFFTaccFKRENyq"]
FGGaccFKRENyq = load("Computed Data/Accuracy/FGGaccFKRENyq.jld")
FGGaccFKRENyq = FGGaccFKRENyq["FGGaccFKRENyq"]
KBaccFKRENyq = load("Computed Data/Accuracy/KBaccFKRENyq.jld")
KBaccFKRENyq = KBaccFKRENyq["KBaccFKRENyq"]
FINUFFTaccFKRENyq = load("Computed Data/Accuracy/FINUFFTaccFKRENyq.jld")
FINUFFTaccFKRENyq = FINUFFTaccFKRENyq["FINUFFTaccFKRENyq"]
ESaccFKRENyq = load("Computed Data/Accuracy/ESaccFKRENyq.jld")
ESaccFKRENyq = ESaccFKRENyq["ESaccFKRENyq"]



# Plot - Dirichlet
n1 = collect(-1:-1:-14)
xticklabel = [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

p1 = plot(n1, mean(FGGaccDKRENyq .- CFTaccDKRENyq, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p1, n1, mean(KBaccDKRENyq .- CFTaccDKRENyq, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p1, n1, mean(ESaccDKRENyq .- CFTaccDKRENyq, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p1, n1, mean(FINUFFTaccDKRENyq .- CFTaccDKRENyq, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
hline!(p1, [mean(ZFFTaccDKRENyq .- CFTaccDKRENyq)], label = "ZFFT", line=(2, [:dashdot]), color = :gray)
plot(p1, annotations=(-13, -0.00004, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccDKRENyq), digits = 4))\$"), :left)))
ylabel!(p1, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p1, L"\textrm{Tolerance } \epsilon ")

savefig(p1, "Plots/MM-NUFFT/AccREDKNyq.svg")



# Fejer
p2 = plot(n1, mean(FGGaccFKRENyq .- CFTaccFKRENyq, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p2, n1, mean(KBaccFKRENyq .- CFTaccFKRENyq, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p2, n1, mean(ESaccFKRENyq .- CFTaccFKRENyq, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p2, n1, mean(FINUFFTaccFKRENyq .- CFTaccFKRENyq, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
hline!(p2, [mean(ZFFTaccFKRENyq .- CFTaccFKRENyq)], label = "ZFFT", line=(2, [:dashdot]), color = :gray)
plot(p2, annotations=(-13, -0.00002, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccDKRENyq), digits = 4))\$"), :left)))
ylabel!(p2, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p2, L"\textrm{Tolerance } \epsilon ")

savefig(p2, "Plots/MM-NUFFT/AccREFKNyq.svg")


#---------------------------------------------------------------------------
## Minimum Nyquist from the replications - fixed for each replication
# No option for kwarg N for ZFFT
NDist = zeros(reps, 1)

@showprogress "Computing..." for j in 1:reps
    Random.seed!(j)
    t1 = [1; rexp(10000, 30)]
    t1 = cumsum(t1)
    t1 = filter((x) -> x < 10000, t1)

    Random.seed!(j+reps)
    t2 = [1; rexp(10000, 45)]
    t2 = cumsum(t2)
    t2 = filter((x) -> x < 10000, t2)

    D = maximum([length(t1); length(t2)]) - minimum([length(t1); length(t2)])
    if length(t1) < length(t2)
        t1 = [t1; repeat([NaN], D)]
    else
        t2 = [t2; repeat([NaN], D)]
    end

    t = [t1 t2]

    tau = scale(t)
    # Computing minimum time change
    # minumum step size to avoid smoothing
    dtau = diff(filter(!isnan, tau))
    taumin = minimum(filter((x) -> x>0, dtau))
    taumax = 2*pi
    # Sampling Freq.
    N0 = taumax/taumin
    NDist[j] = floor(N0/2)
end
# Obtain the minimum Nyquist from the various replications.
NH = Int(minimum(NDist))


n1 = collect(-1:-1:-14)
tol = 10.0.^n1
reps = 100

# Dirichlet
CFTaccDKRENH = zeros(1, reps)
# ZFFTaccDKRENH = zeros(1, reps)

FGGaccDKRENH = zeros(length(n1), reps)
KBaccDKRENH = zeros(length(n1), reps)
FINUFFTaccDKRENH = zeros(length(n1), reps)
ESaccDKRENH = zeros(length(n1), reps)

# Fejer
CFTaccFKRENH = zeros(1, reps)
# ZFFTaccFKRENH = zeros(1, reps)

FGGaccFKRENH = zeros(length(n1), reps)
KBaccFKRENH = zeros(length(n1), reps)
FINUFFTaccFKRENH = zeros(length(n1), reps)
ESaccFKRENH = zeros(length(n1), reps)

# Compute
@showprogress "Computing..." for j in 1:reps
    P = GBM(10000, mu, sigma, seed = j)
    t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

    Random.seed!(j)
    t1 = [1; rexp(10000, 30)]
    t1 = cumsum(t1)
    t1 = filter((x) -> x < 10000, t1)

    Random.seed!(j+reps)
    t2 = [1; rexp(10000, 45)]
    t2 = cumsum(t2)
    t2 = filter((x) -> x < 10000, t2)

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

    CFTaccDKRENH[j] = CFTcorrDK(P, t, N = NH)[1][1,2]

    CFTaccFKRENH[j] = CFTcorrFK(P, t, N = NH)[1][1,2]

    for i in 1:length(n1)
        FGGaccDKRENH[i, j] = NUFFTcorrDKFGG(P, t, tol = tol[i], N = NH)[1][1,2]
        KBaccDKRENH[i, j] = NUFFTcorrDKKB(P, t, tol = tol[i], N = NH)[1][1,2]
        FINUFFTaccDKRENH[i, j] = NUFFTcorrDKFINUFFT(P, t, tol = tol[i], N = NH)[1][1,2]
        ESaccDKRENH[i, j] = NUFFTcorrDKES(P, t, tol = tol[i], N = NH)[1][1,2]

        FGGaccFKRENH[i, j] = NUFFTcorrFKFGG(P, t, tol = tol[i], N = NH)[1][1,2]
        KBaccFKRENH[i, j] = NUFFTcorrFKKB(P, t, tol = tol[i], N = NH)[1][1,2]
        FINUFFTaccFKRENH[i, j] = NUFFTcorrFKFINUFFT(P, t, tol = tol[i], N = NH)[1][1,2]
        ESaccFKRENH[i, j] = NUFFTcorrFKES(P, t, tol = tol[i], N = NH)[1][1,2]
    end
end

# Save
save("Computed Data/Accuracy/CFTaccDKRENH.jld", "CFTaccDKRENH", CFTaccDKRENH)
# save("Computed Data/Accuracy/ZFFTaccDKRENyq.jld", "ZFFTaccDKRENyq", ZFFTaccDKRENyq)
save("Computed Data/Accuracy/FGGaccDKRENH.jld", "FGGaccDKRENH", FGGaccDKRENH)
save("Computed Data/Accuracy/KBaccDKRENH.jld", "KBaccDKRENH", KBaccDKRENH)
save("Computed Data/Accuracy/FINUFFTaccDKRENH.jld", "FINUFFTaccDKRENH", FINUFFTaccDKRENH)
save("Computed Data/Accuracy/ESaccDKRENH.jld", "ESaccDKRENH", ESaccDKRENH)

save("Computed Data/Accuracy/CFTaccFKRENH.jld", "CFTaccFKRENH", CFTaccFKRENH)
# save("Computed Data/Accuracy/ZFFTaccFKRENyq.jld", "ZFFTaccFKRENyq", ZFFTaccFKRENyq)
save("Computed Data/Accuracy/FGGaccFKRENH.jld", "FGGaccFKRENH", FGGaccFKRENH)
save("Computed Data/Accuracy/KBaccFKRENH.jld", "KBaccFKRENH", KBaccFKRENH)
save("Computed Data/Accuracy/FINUFFTaccFKRENH.jld", "FINUFFTaccFKRENH", FINUFFTaccFKRENH)
save("Computed Data/Accuracy/ESaccFKRENH.jld", "ESaccFKRENH", ESaccFKRENH)

# Load
CFTaccDKRENH = load("Computed Data/Accuracy/CFTaccDKRENH.jld")
CFTaccDKRENH = CFTaccDKRENH["CFTaccDKRENH"]
# ZFFTaccDKRENyq = load("Computed Data/Accuracy/ZFFTaccDKRENyq.jld")
# ZFFTaccDKRENyq = ZFFTaccDKRENyq["ZFFTaccDKRENyq"]
FGGaccDKRENH = load("Computed Data/Accuracy/FGGaccDKRENH.jld")
FGGaccDKRENH = FGGaccDKRENH["FGGaccDKRENH"]
KBaccDKRENH = load("Computed Data/Accuracy/KBaccDKRENH.jld")
KBaccDKRENH = KBaccDKRENH["KBaccDKRENH"]
FINUFFTaccDKRENH = load("Computed Data/Accuracy/FINUFFTaccDKRENH.jld")
FINUFFTaccDKRENH = FINUFFTaccDKRENH["FINUFFTaccDKRENH"]
ESaccDKRENH = load("Computed Data/Accuracy/ESaccDKRENH.jld")
ESaccDKRENH = ESaccDKRENH["ESaccDKRENH"]

CFTaccFKRENH = load("Computed Data/Accuracy/CFTaccFKRENH.jld")
CFTaccFKRENH = CFTaccFKRENH["CFTaccFKRENH"]
# ZFFTaccFKRENyq = load("Computed Data/Accuracy/ZFFTaccFKRENyq.jld")
# ZFFTaccFKRENyq = ZFFTaccFKRENyq["ZFFTaccFKRENyq"]
FGGaccFKRENH = load("Computed Data/Accuracy/FGGaccFKRENH.jld")
FGGaccFKRENH = FGGaccFKRENH["FGGaccFKRENH"]
KBaccFKRENH = load("Computed Data/Accuracy/KBaccFKRENH.jld")
KBaccFKRENH = KBaccFKRENH["KBaccFKRENH"]
FINUFFTaccFKRENH = load("Computed Data/Accuracy/FINUFFTaccFKRENH.jld")
FINUFFTaccFKRENH = FINUFFTaccFKRENH["FINUFFTaccFKRENH"]
ESaccFKRENH = load("Computed Data/Accuracy/ESaccFKRENH.jld")
ESaccFKRENH = ESaccFKRENH["ESaccFKRENH"]


# Plot - Dirichlet
n1 = collect(-1:-1:-14)
xticklabel = [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

p3 = plot(n1, mean(FGGaccDKRENH .- CFTaccDKRENH, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p3, n1, mean(KBaccDKRENH .- CFTaccDKRENH, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p3, n1, mean(ESaccDKRENH .- CFTaccDKRENH, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p3, n1, mean(FINUFFTaccDKRENH .- CFTaccDKRENH, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot(p3, annotations=(-13, -0.0000045, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccDKRENH), digits = 4))\$"), :left)))
ylabel!(p3, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p3, L"\textrm{Tolerance } \epsilon ")

savefig(p3, "Plots/MM-NUFFT/AccREDKNH.svg")


# Fejer
p4 = plot(n1, mean(FGGaccFKRENH .- CFTaccFKRENH, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p4, n1, mean(KBaccFKRENH .- CFTaccFKRENH, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p4, n1, mean(ESaccFKRENH .- CFTaccFKRENH, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p4, n1, mean(FINUFFTaccFKRENH .- CFTaccFKRENH, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot(p4, annotations=(-13, -0.00001, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccFKRENH), digits = 4))\$"), :left)))
ylabel!(p4, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p4, L"\textrm{Tolerance } \epsilon ")

savefig(p4, "Plots/MM-NUFFT/AccREFKNH.svg")



#---------------------------------------------------------------------------
## Average Nyquist from the sampling rate - fixed for each replication
# No option for kwarg N for ZFFT

# Obtain the minimum Nyquist from the various replications.
NA = Int(floor(10000/(30*2)))


n1 = collect(-1:-1:-14)
tol = 10.0.^n1
reps = 100

# Dirichlet
CFTaccDKRENA = zeros(1, reps)
# ZFFTaccDKRENH = zeros(1, reps)

FGGaccDKRENA = zeros(length(n1), reps)
KBaccDKRENA = zeros(length(n1), reps)
FINUFFTaccDKRENA = zeros(length(n1), reps)
ESaccDKRENA = zeros(length(n1), reps)

# Fejer
CFTaccFKRENA = zeros(1, reps)
# ZFFTaccFKRENH = zeros(1, reps)

FGGaccFKRENA = zeros(length(n1), reps)
KBaccFKRENA = zeros(length(n1), reps)
FINUFFTaccFKRENA = zeros(length(n1), reps)
ESaccFKRENA = zeros(length(n1), reps)

# Compute
@showprogress "Computing..." for j in 1:reps
    P = GBM(10000, mu, sigma, seed = j)
    t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

    Random.seed!(j)
    t1 = [1; rexp(10000, 30)]
    t1 = cumsum(t1)
    t1 = filter((x) -> x < 10000, t1)

    Random.seed!(j+reps)
    t2 = [1; rexp(10000, 45)]
    t2 = cumsum(t2)
    t2 = filter((x) -> x < 10000, t2)

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

    CFTaccDKRENA[j] = CFTcorrDK(P, t, N = NA)[1][1,2]

    CFTaccFKRENA[j] = CFTcorrFK(P, t, N = NA)[1][1,2]

    for i in 1:length(n1)
        FGGaccDKRENA[i, j] = NUFFTcorrDKFGG(P, t, tol = tol[i], N = NA)[1][1,2]
        KBaccDKRENA[i, j] = NUFFTcorrDKKB(P, t, tol = tol[i], N = NA)[1][1,2]
        FINUFFTaccDKRENA[i, j] = NUFFTcorrDKFINUFFT(P, t, tol = tol[i], N = NA)[1][1,2]
        ESaccDKRENA[i, j] = NUFFTcorrDKES(P, t, tol = tol[i], N = NA)[1][1,2]

        FGGaccFKRENA[i, j] = NUFFTcorrFKFGG(P, t, tol = tol[i], N = NA)[1][1,2]
        KBaccFKRENA[i, j] = NUFFTcorrFKKB(P, t, tol = tol[i], N = NA)[1][1,2]
        FINUFFTaccFKRENA[i, j] = NUFFTcorrFKFINUFFT(P, t, tol = tol[i], N = NA)[1][1,2]
        ESaccFKRENA[i, j] = NUFFTcorrFKES(P, t, tol = tol[i], N = NA)[1][1,2]
    end
end

# Save
save("Computed Data/Accuracy/CFTaccDKRENA.jld", "CFTaccDKRENA", CFTaccDKRENA)
# save("Computed Data/Accuracy/ZFFTaccDKRENyq.jld", "ZFFTaccDKRENyq", ZFFTaccDKRENyq)
save("Computed Data/Accuracy/FGGaccDKRENA.jld", "FGGaccDKRENA", FGGaccDKRENA)
save("Computed Data/Accuracy/KBaccDKRENA.jld", "KBaccDKRENA", KBaccDKRENA)
save("Computed Data/Accuracy/FINUFFTaccDKRENA.jld", "FINUFFTaccDKRENA", FINUFFTaccDKRENA)
save("Computed Data/Accuracy/ESaccDKRENA.jld", "ESaccDKRENA", ESaccDKRENA)

save("Computed Data/Accuracy/CFTaccFKRENA.jld", "CFTaccFKRENA", CFTaccFKRENA)
# save("Computed Data/Accuracy/ZFFTaccFKRENyq.jld", "ZFFTaccFKRENyq", ZFFTaccFKRENyq)
save("Computed Data/Accuracy/FGGaccFKRENA.jld", "FGGaccFKRENA", FGGaccFKRENA)
save("Computed Data/Accuracy/KBaccFKRENA.jld", "KBaccFKRENA", KBaccFKRENA)
save("Computed Data/Accuracy/FINUFFTaccFKRENA.jld", "FINUFFTaccFKRENA", FINUFFTaccFKRENA)
save("Computed Data/Accuracy/ESaccFKRENA.jld", "ESaccFKRENA", ESaccFKRENA)

# Load
CFTaccDKRENA = load("Computed Data/Accuracy/CFTaccDKRENA.jld")
CFTaccDKRENA = CFTaccDKRENA["CFTaccDKRENA"]
# ZFFTaccDKRENyq = load("Computed Data/Accuracy/ZFFTaccDKRENyq.jld")
# ZFFTaccDKRENyq = ZFFTaccDKRENyq["ZFFTaccDKRENyq"]
FGGaccDKRENA = load("Computed Data/Accuracy/FGGaccDKRENA.jld")
FGGaccDKRENA = FGGaccDKRENA["FGGaccDKRENA"]
KBaccDKRENA = load("Computed Data/Accuracy/KBaccDKRENA.jld")
KBaccDKRENA = KBaccDKRENA["KBaccDKRENA"]
FINUFFTaccDKRENA = load("Computed Data/Accuracy/FINUFFTaccDKRENA.jld")
FINUFFTaccDKRENA = FINUFFTaccDKRENA["FINUFFTaccDKRENA"]
ESaccDKRENA = load("Computed Data/Accuracy/ESaccDKRENA.jld")
ESaccDKRENA = ESaccDKRENA["ESaccDKRENA"]

CFTaccFKRENA = load("Computed Data/Accuracy/CFTaccFKRENA.jld")
CFTaccFKRENA = CFTaccFKRENA["CFTaccFKRENA"]
# ZFFTaccFKRENyq = load("Computed Data/Accuracy/ZFFTaccFKRENyq.jld")
# ZFFTaccFKRENyq = ZFFTaccFKRENyq["ZFFTaccFKRENyq"]
FGGaccFKRENA = load("Computed Data/Accuracy/FGGaccFKRENA.jld")
FGGaccFKRENA = FGGaccFKRENA["FGGaccFKRENA"]
KBaccFKRENA = load("Computed Data/Accuracy/KBaccFKRENA.jld")
KBaccFKRENA = KBaccFKRENA["KBaccFKRENA"]
FINUFFTaccFKRENA = load("Computed Data/Accuracy/FINUFFTaccFKRENA.jld")
FINUFFTaccFKRENA = FINUFFTaccFKRENA["FINUFFTaccFKRENA"]
ESaccFKRENA = load("Computed Data/Accuracy/ESaccFKRENA.jld")
ESaccFKRENA = ESaccFKRENA["ESaccFKRENA"]


# Plot - Dirichlet
n1 = collect(-1:-1:-14)
xticklabel = [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

p5 = plot(n1, mean(FGGaccDKRENA .- CFTaccDKRENA, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p5, n1, mean(KBaccDKRENA .- CFTaccDKRENA, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p5, n1, mean(ESaccDKRENA .- CFTaccDKRENA, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p5, n1, mean(FINUFFTaccDKRENA .- CFTaccDKRENA, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot(p5, annotations=(-13, -0.00125, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccDKRENA), digits = 4))\$"), :left)))
ylabel!(p5, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p5, L"\textrm{Tolerance } \epsilon ")

savefig(p5, "Plots/MM-NUFFT/AccREDKNA.svg")


# Fejer
p6 = plot(n1, mean(FGGaccFKRENA .- CFTaccFKRENA, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p6, n1, mean(KBaccFKRENA .- CFTaccFKRENA, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p6, n1, mean(ESaccFKRENA .- CFTaccFKRENA, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p6, n1, mean(FINUFFTaccFKRENA .- CFTaccFKRENA, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot(p6, annotations=(-13, -0.001, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccFKRENA), digits = 4))\$"), :left)))
ylabel!(p6, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p6, L"\textrm{Tolerance } \epsilon ")

savefig(p6, "Plots/MM-NUFFT/AccREFKNA.svg")


#---------------------------------------------------------------------------
## Small N - fixed for each replication
# No option for kwarg N for ZFFT

# Obtain the minimum Nyquist from the various replications.
NL = Int(15)


n1 = collect(-1:-1:-14)
tol = 10.0.^n1
reps = 100

# Dirichlet
CFTaccDKRENL = zeros(1, reps)
# ZFFTaccDKRENH = zeros(1, reps)

FGGaccDKRENL = zeros(length(n1), reps)
KBaccDKRENL = zeros(length(n1), reps)
FINUFFTaccDKRENL = zeros(length(n1), reps)
ESaccDKRENL = zeros(length(n1), reps)

# Fejer
CFTaccFKRENL = zeros(1, reps)
# ZFFTaccFKRENH = zeros(1, reps)

FGGaccFKRENL = zeros(length(n1), reps)
KBaccFKRENL = zeros(length(n1), reps)
FINUFFTaccFKRENL = zeros(length(n1), reps)
ESaccFKRENL = zeros(length(n1), reps)

# Compute
@showprogress "Computing..." for j in 1:reps
    P = GBM(10000, mu, sigma, seed = j)
    t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

    Random.seed!(j)
    t1 = [1; rexp(10000, 30)]
    t1 = cumsum(t1)
    t1 = filter((x) -> x < 10000, t1)

    Random.seed!(j+reps)
    t2 = [1; rexp(10000, 45)]
    t2 = cumsum(t2)
    t2 = filter((x) -> x < 10000, t2)

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

    CFTaccDKRENL[j] = CFTcorrDK(P, t, N = NL)[1][1,2]

    CFTaccFKRENL[j] = CFTcorrFK(P, t, N = NL)[1][1,2]

    for i in 1:length(n1)
        FGGaccDKRENL[i, j] = NUFFTcorrDKFGG(P, t, tol = tol[i], N = NL)[1][1,2]
        KBaccDKRENL[i, j] = NUFFTcorrDKKB(P, t, tol = tol[i], N = NL)[1][1,2]
        FINUFFTaccDKRENL[i, j] = NUFFTcorrDKFINUFFT(P, t, tol = tol[i], N = NL)[1][1,2]
        ESaccDKRENL[i, j] = NUFFTcorrDKES(P, t, tol = tol[i], N = NL)[1][1,2]

        FGGaccFKRENL[i, j] = NUFFTcorrFKFGG(P, t, tol = tol[i], N = NL)[1][1,2]
        KBaccFKRENL[i, j] = NUFFTcorrFKKB(P, t, tol = tol[i], N = NL)[1][1,2]
        FINUFFTaccFKRENL[i, j] = NUFFTcorrFKFINUFFT(P, t, tol = tol[i], N = NL)[1][1,2]
        ESaccFKRENL[i, j] = NUFFTcorrFKES(P, t, tol = tol[i], N = NL)[1][1,2]
    end
end

# Save
save("Computed Data/Accuracy/CFTaccDKRENL.jld", "CFTaccDKRENL", CFTaccDKRENL)
# save("Computed Data/Accuracy/ZFFTaccDKRENyq.jld", "ZFFTaccDKRENyq", ZFFTaccDKRENyq)
save("Computed Data/Accuracy/FGGaccDKRENL.jld", "FGGaccDKRENL", FGGaccDKRENL)
save("Computed Data/Accuracy/KBaccDKRENL.jld", "KBaccDKRENL", KBaccDKRENL)
save("Computed Data/Accuracy/FINUFFTaccDKRENL.jld", "FINUFFTaccDKRENL", FINUFFTaccDKRENL)
save("Computed Data/Accuracy/ESaccDKRENL.jld", "ESaccDKRENL", ESaccDKRENL)

save("Computed Data/Accuracy/CFTaccFKRENL.jld", "CFTaccFKRENL", CFTaccFKRENL)
# save("Computed Data/Accuracy/ZFFTaccFKRENyq.jld", "ZFFTaccFKRENyq", ZFFTaccFKRENyq)
save("Computed Data/Accuracy/FGGaccFKRENL.jld", "FGGaccFKRENL", FGGaccFKRENL)
save("Computed Data/Accuracy/KBaccFKRENL.jld", "KBaccFKRENL", KBaccFKRENL)
save("Computed Data/Accuracy/FINUFFTaccFKRENL.jld", "FINUFFTaccFKRENL", FINUFFTaccFKRENL)
save("Computed Data/Accuracy/ESaccFKRENL.jld", "ESaccFKRENL", ESaccFKRENL)

# Load
CFTaccDKRENL = load("Computed Data/Accuracy/CFTaccDKRENL.jld")
CFTaccDKRENL = CFTaccDKRENL["CFTaccDKRENL"]
# ZFFTaccDKRENyq = load("Computed Data/Accuracy/ZFFTaccDKRENyq.jld")
# ZFFTaccDKRENyq = ZFFTaccDKRENyq["ZFFTaccDKRENyq"]
FGGaccDKRENL = load("Computed Data/Accuracy/FGGaccDKRENL.jld")
FGGaccDKRENL = FGGaccDKRENL["FGGaccDKRENL"]
KBaccDKRENL = load("Computed Data/Accuracy/KBaccDKRENL.jld")
KBaccDKRENL = KBaccDKRENL["KBaccDKRENL"]
FINUFFTaccDKRENL = load("Computed Data/Accuracy/FINUFFTaccDKRENL.jld")
FINUFFTaccDKRENL = FINUFFTaccDKRENL["FINUFFTaccDKRENL"]
ESaccDKRENL = load("Computed Data/Accuracy/ESaccDKRENL.jld")
ESaccDKRENL = ESaccDKRENL["ESaccDKRENL"]

CFTaccFKRENL = load("Computed Data/Accuracy/CFTaccFKRENL.jld")
CFTaccFKRENL = CFTaccFKRENL["CFTaccFKRENL"]
# ZFFTaccFKRENyq = load("Computed Data/Accuracy/ZFFTaccFKRENyq.jld")
# ZFFTaccFKRENyq = ZFFTaccFKRENyq["ZFFTaccFKRENyq"]
FGGaccFKRENL = load("Computed Data/Accuracy/FGGaccFKRENL.jld")
FGGaccFKRENL = FGGaccFKRENL["FGGaccFKRENL"]
KBaccFKRENL = load("Computed Data/Accuracy/KBaccFKRENL.jld")
KBaccFKRENL = KBaccFKRENL["KBaccFKRENL"]
FINUFFTaccFKRENL = load("Computed Data/Accuracy/FINUFFTaccFKRENL.jld")
FINUFFTaccFKRENL = FINUFFTaccFKRENL["FINUFFTaccFKRENL"]
ESaccFKRENL = load("Computed Data/Accuracy/ESaccFKRENL.jld")
ESaccFKRENL = ESaccFKRENL["ESaccFKRENL"]


# Plot - Dirichlet
n1 = collect(-1:-1:-14)
xticklabel = [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

p7 = plot(n1, mean(FGGaccDKRENL .- CFTaccDKRENL, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p7, n1, mean(KBaccDKRENL .- CFTaccDKRENL, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p7, n1, mean(ESaccDKRENL .- CFTaccDKRENL, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p7, n1, mean(FINUFFTaccDKRENL .- CFTaccDKRENL, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot(p7, annotations=(-13, -0.0015, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccDKRENL), digits = 4))\$"), :left)))
ylabel!(p7, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p7, L"\textrm{Tolerance } \epsilon ")

savefig(p7, "Plots/MM-NUFFT/AccREDKNL.svg")


# Fejer
p8 = plot(n1, mean(FGGaccFKRENL .- CFTaccFKRENL, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p8, n1, mean(KBaccFKRENL .- CFTaccFKRENL, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p8, n1, mean(ESaccFKRENL .- CFTaccFKRENL, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p8, n1, mean(FINUFFTaccFKRENL .- CFTaccFKRENL, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot(p8, annotations=(-13, -0.00065, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccFKRENL), digits = 4))\$"), :left)))
ylabel!(p8, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p8, L"\textrm{Tolerance } \epsilon ")

savefig(p8, "Plots/MM-NUFFT/AccREFKNL.svg")
