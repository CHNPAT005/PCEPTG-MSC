## Author: Patrick Chang
# Script file to test the impact of ϵ on the correlation estimate
# for various averaging kernels using the Vectorized code as a baseline.
# We investigate the synchronous case and down-sampled case

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
## Dirichlet
#---------------------------------------------------------------------------

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

#--------------------

n1 = collect(-1:-1:-14)
tol = 10.0.^n1
reps = 100

## Synchronous Case
CFTaccDKSyn = zeros(1, reps)
ZFFTaccDKSyn = zeros(1, reps)

FGGaccDKSyn = zeros(length(n1), reps)
KBaccDKSyn = zeros(length(n1), reps)
FINUFFTaccDKSyn = zeros(length(n1), reps)
ESaccDKSyn = zeros(length(n1), reps)

# Compute
@showprogress "Computing..." for j in 1:reps
    P = GBM(10000, mu, sigma, seed = j)
    t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

    CFTaccDKSyn[j] = CFTcorrDK(P, t)[1][1,2]
    ZFFTaccDKSyn[j] = FFTZPcorrDK(P, t)[1][1,2]

    for i in 1:length(n1)
        FGGaccDKSyn[i, j] = NUFFTcorrDKFGG(P, t, tol = tol[i])[1][1,2]
        KBaccDKSyn[i, j] = NUFFTcorrDKKB(P, t, tol = tol[i])[1][1,2]
        FINUFFTaccDKSyn[i, j] = NUFFTcorrDKFINUFFT(P, t, tol = tol[i])[1][1,2]
        ESaccDKSyn[i, j] = NUFFTcorrDKES(P, t, tol = tol[i])[1][1,2]
    end
end

# Save
save("Computed Data/Accuracy/CFTaccDKSyn.jld", "CFTaccDKSyn", CFTaccDKSyn)
save("Computed Data/Accuracy/ZFFTaccDKSyn.jld", "ZFFTaccDKSyn", ZFFTaccDKSyn)
save("Computed Data/Accuracy/FGGaccDKSyn.jld", "FGGaccDKSyn", FGGaccDKSyn)
save("Computed Data/Accuracy/KBaccDKSyn.jld", "KBaccDKSyn", KBaccDKSyn)
save("Computed Data/Accuracy/FINUFFTaccDKSyn.jld", "FINUFFTaccDKSyn", FINUFFTaccDKSyn)
save("Computed Data/Accuracy/ESaccDKSyn.jld", "ESaccDKSyn", ESaccDKSyn)

# Load
CFTaccDKSyn = load("Computed Data/Accuracy/CFTaccDKSyn.jld")
CFTaccDKSyn = CFTaccDKSyn["CFTaccDKSyn"]

ZFFTaccDKSyn = load("Computed Data/Accuracy/ZFFTaccDKSyn.jld")
ZFFTaccDKSyn = ZFFTaccDKSyn["ZFFTaccDKSyn"]

FGGaccDKSyn = load("Computed Data/Accuracy/FGGaccDKSyn.jld")
FGGaccDKSyn = FGGaccDKSyn["FGGaccDKSyn"]

KBaccDKSyn = load("Computed Data/Accuracy/KBaccDKSyn.jld")
KBaccDKSyn = KBaccDKSyn["KBaccDKSyn"]

FINUFFTaccDKSyn = load("Computed Data/Accuracy/FINUFFTaccDKSyn.jld")
FINUFFTaccDKSyn = FINUFFTaccDKSyn["FINUFFTaccDKSyn"]

ESaccDKSyn = load("Computed Data/Accuracy/ESaccDKSyn.jld")
ESaccDKSyn = ESaccDKSyn["ESaccDKSyn"]

# Plot
n1 = collect(-1:-1:-14)
xticklabel = [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

p1 = plot(n1, mean(FGGaccDKSyn .- CFTaccDKSyn, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p1, n1, mean(KBaccDKSyn .- CFTaccDKSyn, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p1, n1, mean(ESaccDKSyn .- CFTaccDKSyn, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p1, n1, mean(FINUFFTaccDKSyn .- CFTaccDKSyn, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
hline!(p1, [mean(ZFFTaccDKSyn .- CFTaccDKSyn)], label = "ZFFT", line=(2, [:dashdot]), color = :gray)
plot(p1, annotations=(-13, 0.00003, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccDKSyn), digits = 4))\$"), :left)))
ylabel!(p1, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p1, L"\textrm{Tolerance } \epsilon ")

savefig(p1, "Plots/MM-NUFFT/AccSynDK.svg")



## Down-sampled Case
n1 = collect(-1:-1:-14)
tol = 10.0.^n1
reps = 100

CFTaccDKDS = zeros(1, reps)
ZFFTaccDKDS = zeros(1, reps)

FGGaccDKDS = zeros(length(n1), reps)
KBaccDKDS = zeros(length(n1), reps)
FINUFFTaccDKDS = zeros(length(n1), reps)
ESaccDKDS = zeros(length(n1), reps)

# Compute
@showprogress "Computing..." for j in 1:reps
    P = GBM(10000, mu, sigma, seed = j)
    t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

    Random.seed!(j)
    rm1 = sample(2:9999, 4000, replace = false)
    Random.seed!(j+reps)
    rm2 = sample(2:9999, 4000, replace = false)

    P[rm1, 1] .= NaN
    t[rm1, 1] .= NaN
    P[rm2, 2] .= NaN
    t[rm2, 2] .= NaN

    CFTaccDKDS[j] = CFTcorrDK(P, t)[1][1,2]
    ZFFTaccDKDS[j] = FFTZPcorrDK(P, t)[1][1,2]

    for i in 1:length(n1)
        FGGaccDKDS[i, j] = NUFFTcorrDKFGG(P, t, tol = tol[i])[1][1,2]
        KBaccDKDS[i, j] = NUFFTcorrDKKB(P, t, tol = tol[i])[1][1,2]
        FINUFFTaccDKDS[i, j] = NUFFTcorrDKFINUFFT(P, t, tol = tol[i])[1][1,2]
        ESaccDKDS[i, j] = NUFFTcorrDKES(P, t, tol = tol[i])[1][1,2]
    end
end

# Save
save("Computed Data/Accuracy/CFTaccDKDS.jld", "CFTaccDKDS", CFTaccDKDS)
save("Computed Data/Accuracy/ZFFTaccDKDS.jld", "ZFFTaccDKDS", ZFFTaccDKDS)
save("Computed Data/Accuracy/FGGaccDKDS.jld", "FGGaccDKDS", FGGaccDKDS)
save("Computed Data/Accuracy/KBaccDKDS.jld", "KBaccDKDS", KBaccDKDS)
save("Computed Data/Accuracy/FINUFFTaccDKDS.jld", "FINUFFTaccDKDS", FINUFFTaccDKDS)
save("Computed Data/Accuracy/ESaccDKDS.jld", "ESaccDKDS", ESaccDKDS)

# Load
CFTaccDKDS = load("Computed Data/Accuracy/CFTaccDKDS.jld")
CFTaccDKDS = CFTaccDKDS["CFTaccDKDS"]

ZFFTaccDKDS = load("Computed Data/Accuracy/ZFFTaccDKDS.jld")
ZFFTaccDKDS = ZFFTaccDKDS["ZFFTaccDKDS"]

FGGaccDKDS = load("Computed Data/Accuracy/FGGaccDKDS.jld")
FGGaccDKDS = FGGaccDKDS["FGGaccDKDS"]

KBaccDKDS = load("Computed Data/Accuracy/KBaccDKDS.jld")
KBaccDKDS = KBaccDKDS["KBaccDKDS"]

FINUFFTaccDKDS = load("Computed Data/Accuracy/FINUFFTaccDKDS.jld")
FINUFFTaccDKDS = FINUFFTaccDKDS["FINUFFTaccDKDS"]

ESaccDKDS = load("Computed Data/Accuracy/ESaccDKDS.jld")
ESaccDKDS = ESaccDKDS["ESaccDKDS"]

# Plot
n1 = collect(-1:-1:-14)
xticklabel = [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

p2 = plot(n1, mean(FGGaccDKDS .- CFTaccDKDS, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p2, n1, mean(KBaccDKDS .- CFTaccDKDS, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p2, n1, mean(ESaccDKDS .- CFTaccDKDS, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p2, n1, mean(FINUFFTaccDKDS .- CFTaccDKDS, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
hline!(p2, [mean(ZFFTaccDKDS .- CFTaccDKDS)], label = "ZFFT", line=(2, [:dashdot]), color = :gray)
plot(p2, annotations=(-13, -0.004, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccDKDS), digits = 4))\$"), :left)))
ylabel!(p2, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p2, L"\textrm{Tolerance } \epsilon ")

savefig(p2, "Plots/MM-NUFFT/AccDSDK.svg")



#---------------------------------------------------------------------------
## Fejer
#---------------------------------------------------------------------------

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

#--------------------

n1 = collect(-1:-1:-14)
tol = 10.0.^n1
reps = 100

## Synchronous Case
CFTaccFKSyn = zeros(1, reps)
ZFFTaccFKSyn = zeros(1, reps)

FGGaccFKSyn = zeros(length(n1), reps)
KBaccFKSyn = zeros(length(n1), reps)
FINUFFTaccFKSyn = zeros(length(n1), reps)
ESaccFKSyn = zeros(length(n1), reps)

# Compute
@showprogress "Computing..." for j in 1:reps
    P = GBM(10000, mu, sigma, seed = j)
    t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

    CFTaccFKSyn[j] = CFTcorrFK(P, t)[1][1,2]
    ZFFTaccFKSyn[j] = FFTZPcorrFK(P, t)[1][1,2]

    for i in 1:length(n1)
        FGGaccFKSyn[i, j] = NUFFTcorrFKFGG(P, t, tol = tol[i])[1][1,2]
        KBaccFKSyn[i, j] = NUFFTcorrFKKB(P, t, tol = tol[i])[1][1,2]
        FINUFFTaccFKSyn[i, j] = NUFFTcorrFKFINUFFT(P, t, tol = tol[i])[1][1,2]
        ESaccFKSyn[i, j] = NUFFTcorrFKES(P, t, tol = tol[i])[1][1,2]
    end
end

# Save
save("Computed Data/Accuracy/CFTaccFKSyn.jld", "CFTaccFKSyn", CFTaccFKSyn)
save("Computed Data/Accuracy/ZFFTaccFKSyn.jld", "ZFFTaccFKSyn", ZFFTaccFKSyn)
save("Computed Data/Accuracy/FGGaccFKSyn.jld", "FGGaccFKSyn", FGGaccFKSyn)
save("Computed Data/Accuracy/KBaccFKSyn.jld", "KBaccFKSyn", KBaccFKSyn)
save("Computed Data/Accuracy/FINUFFTaccFKSyn.jld", "FINUFFTaccFKSyn", FINUFFTaccFKSyn)
save("Computed Data/Accuracy/ESaccFKSyn.jld", "ESaccFKSyn", ESaccFKSyn)

# Load
CFTaccFKSyn = load("Computed Data/Accuracy/CFTaccFKSyn.jld")
CFTaccFKSyn = CFTaccFKSyn["CFTaccFKSyn"]

ZFFTaccFKSyn = load("Computed Data/Accuracy/ZFFTaccFKSyn.jld")
ZFFTaccFKSyn = ZFFTaccFKSyn["ZFFTaccFKSyn"]

FGGaccFKSyn = load("Computed Data/Accuracy/FGGaccFKSyn.jld")
FGGaccFKSyn = FGGaccFKSyn["FGGaccFKSyn"]

KBaccFKSyn = load("Computed Data/Accuracy/KBaccFKSyn.jld")
KBaccFKSyn = KBaccFKSyn["KBaccFKSyn"]

FINUFFTaccFKSyn = load("Computed Data/Accuracy/FINUFFTaccFKSyn.jld")
FINUFFTaccFKSyn = FINUFFTaccFKSyn["FINUFFTaccFKSyn"]

ESaccFKSyn = load("Computed Data/Accuracy/ESaccFKSyn.jld")
ESaccFKSyn = ESaccFKSyn["ESaccFKSyn"]

# Plot
n1 = collect(-1:-1:-14)
xticklabel = [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

p3 = plot(n1, mean(FGGaccFKSyn .- CFTaccFKSyn, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p3, n1, mean(KBaccFKSyn .- CFTaccFKSyn, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p3, n1, mean(ESaccFKSyn .- CFTaccFKSyn, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p3, n1, mean(FINUFFTaccFKSyn .- CFTaccFKSyn, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
hline!(p3, [mean(ZFFTaccFKSyn .- CFTaccFKSyn)], label = "ZFFT", line=(2, [:dashdot]), color = :gray)
plot(p3, annotations=(-13, -0.000005, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccFKSyn), digits = 4))\$"), :left)))
ylabel!(p3, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p3, L"\textrm{Tolerance } \epsilon ")

savefig(p3, "Plots/MM-NUFFT/AccSynFK.svg")


## Down-sampled Case
n1 = collect(-1:-1:-14)
tol = 10.0.^n1
reps = 100

CFTaccFKDS = zeros(1, reps)
ZFFTaccFKDS = zeros(1, reps)

FGGaccFKDS = zeros(length(n1), reps)
KBaccFKDS = zeros(length(n1), reps)
FINUFFTaccFKDS = zeros(length(n1), reps)
ESaccFKDS = zeros(length(n1), reps)

# Compute
@showprogress "Computing..." for j in 1:reps
    P = GBM(10000, mu, sigma, seed = j)
    t = reshape([collect(1:1:10000.0); collect(1:1:10000.0)], 10000, 2)

    Random.seed!(j)
    rm1 = sample(2:9999, 4000, replace = false)
    Random.seed!(j+reps)
    rm2 = sample(2:9999, 4000, replace = false)

    P[rm1, 1] .= NaN
    t[rm1, 1] .= NaN
    P[rm2, 2] .= NaN
    t[rm2, 2] .= NaN

    CFTaccFKDS[j] = CFTcorrFK(P, t)[1][1,2]
    ZFFTaccFKDS[j] = FFTZPcorrFK(P, t)[1][1,2]

    for i in 1:length(n1)
        FGGaccFKDS[i, j] = NUFFTcorrFKFGG(P, t, tol = tol[i])[1][1,2]
        KBaccFKDS[i, j] = NUFFTcorrFKKB(P, t, tol = tol[i])[1][1,2]
        FINUFFTaccFKDS[i, j] = NUFFTcorrFKFINUFFT(P, t, tol = tol[i])[1][1,2]
        ESaccFKDS[i, j] = NUFFTcorrFKES(P, t, tol = tol[i])[1][1,2]
    end
end

# Save
save("Computed Data/Accuracy/CFTaccFKDS.jld", "CFTaccFKDS", CFTaccFKDS)
save("Computed Data/Accuracy/ZFFTaccFKDS.jld", "ZFFTaccFKDS", ZFFTaccFKDS)
save("Computed Data/Accuracy/FGGaccFKDS.jld", "FGGaccFKDS", FGGaccFKDS)
save("Computed Data/Accuracy/KBaccFKDS.jld", "KBaccFKDS", KBaccFKDS)
save("Computed Data/Accuracy/FINUFFTaccFKDS.jld", "FINUFFTaccFKDS", FINUFFTaccFKDS)
save("Computed Data/Accuracy/ESaccFKDS.jld", "ESaccFKDS", ESaccFKDS)

# Load
CFTaccFKDS = load("Computed Data/Accuracy/CFTaccFKDS.jld")
CFTaccFKDS = CFTaccFKDS["CFTaccFKDS"]

ZFFTaccFKDS = load("Computed Data/Accuracy/ZFFTaccFKDS.jld")
ZFFTaccFKDS = ZFFTaccFKDS["ZFFTaccFKDS"]

FGGaccFKDS = load("Computed Data/Accuracy/FGGaccFKDS.jld")
FGGaccFKDS = FGGaccFKDS["FGGaccFKDS"]

KBaccFKDS = load("Computed Data/Accuracy/KBaccFKDS.jld")
KBaccFKDS = KBaccFKDS["KBaccFKDS"]

FINUFFTaccFKDS = load("Computed Data/Accuracy/FINUFFTaccFKDS.jld")
FINUFFTaccFKDS = FINUFFTaccFKDS["FINUFFTaccFKDS"]

ESaccFKDS = load("Computed Data/Accuracy/ESaccFKDS.jld")
ESaccFKDS = ESaccFKDS["ESaccFKDS"]

# Plot
n1 = collect(-1:-1:-14)
xticklabel = [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

p4 = plot(n1, mean(FGGaccFKDS .- CFTaccFKDS, dims = 2), xticks = (n1, xticklabel), legend=:topleft, legendtitle=L"\textrm{Method} (*)", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), formatter = :plain, dpi = 300)
plot!(p4, n1, mean(KBaccFKDS .- CFTaccFKDS, dims = 2), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p4, n1, mean(ESaccFKDS .- CFTaccFKDS, dims = 2), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p4, n1, mean(FINUFFTaccFKDS .- CFTaccFKDS, dims = 2), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
hline!(p4, [mean(ZFFTaccFKDS .- CFTaccFKDS)], label = "ZFFT", line=(2, [:dashdot]), color = :gray)
plot(p4, annotations=(-13, -0.003, Plots.text(latexstring("\$\\overline{\\rho_{v}} = $(round(mean(CFTaccFKDS), digits = 4))\$"), :left)))
ylabel!(p4, L"\textrm{Accuracy } \overline{\rho_{*} - \rho_{v}}")
xlabel!(p4, L"\textrm{Tolerance } \epsilon ")

savefig(p4, "Plots/MM-NUFFT/AccDSFK.svg")
