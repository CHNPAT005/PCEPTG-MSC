## Author: Patrick Chang
# Script file to investigate the sensitivity analysis to ensure
# the implementation methods recover the target, and not dependent on
# the parameters chosen.

#---------------------------------------------------------------------------

using LinearAlgebra; using LaTeXStrings; using StatsBase; using Random;
using Statistics; using Distributions; using ProgressMeter; using JLD

#---------------------------------------------------------------------------

cd("/Users/patrickchang1/PCEPTG-MSC")

include("../../Functions/Correlation Estimators/Dirichlet/CFTcorrDK.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FINUFFT.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-KB.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-ES.jl")

include("../../Functions/Correlation Estimators/Fejer/CFTcorrFK.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FINUFFT.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FGG.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-KB.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-ES.jl")

include("../../Functions/SDEs/GBM.jl")

#---------------------------------------------------------------------------
## Functions to steamline results

function sigma11Dir(n, range)
    CFT = zeros(length(range), 1)
    FGG = zeros(length(range), 1)
    KB = zeros(length(range), 1)
    ES = zeros(length(range), 1)
    FINUFFT = zeros(length(range), 1)

    mu = [0.01, 0.01]
    t = reshape([collect(1:1:n); collect(1:1:n)], n, 2)
    for i in 1:length(range)
        sigma = [range[i] sqrt(range[i])*0.35*sqrt(range[i]);
                        sqrt(range[i])*0.35*sqrt(range[i]) range[i]]
        P = GBM2(n, mu, sigma, seed = 1, dt = 1/n)

        CFT[i] = CFTcorrDK(P, t)[2][1,1]
        FGG[i] = NUFFTcorrDKFGG(P, t)[2][1,1]
        KB[i] = NUFFTcorrDKKB(P, t)[2][1,1]
        ES[i] = NUFFTcorrDKES(P, t)[2][1,1]
        FINUFFT[i] = NUFFTcorrDKFINUFFT(P, t)[2][1,1]
    end
    return CFT, FGG, KB, ES, FINUFFT
end

function sigma12Dir(n, ρ)
    CFT = zeros(length(ρ), 1)
    FGG = zeros(length(ρ), 1)
    KB = zeros(length(ρ), 1)
    ES = zeros(length(ρ), 1)
    FINUFFT = zeros(length(ρ), 1)

    mu = [0.01, 0.01]
    t = reshape([collect(1:1:n); collect(1:1:n)], n, 2)

    σ_12 = sqrt(0.1*0.1) .* ρ
    for i in 1:length(ρ)
        sigma = [0.1 σ_12[i]; σ_12[i] 0.1]
        P = GBM2(n, mu, sigma, seed = 1, dt = 1/n)

        CFT[i] = CFTcorrDK(P, t)[2][1,2]
        FGG[i] = NUFFTcorrDKFGG(P, t)[2][1,2]
        KB[i] = NUFFTcorrDKKB(P, t)[2][1,2]
        ES[i] = NUFFTcorrDKES(P, t)[2][1,2]
        FINUFFT[i] = NUFFTcorrDKFINUFFT(P, t)[2][1,2]
    end
    return CFT, FGG, KB, ES, FINUFFT
end

function sigma11Fej(n, range)
    CFT = zeros(length(range), 1)
    FGG = zeros(length(range), 1)
    KB = zeros(length(range), 1)
    ES = zeros(length(range), 1)
    FINUFFT = zeros(length(range), 1)

    mu = [0.01, 0.01]
    t = reshape([collect(1:1:n); collect(1:1:n)], n, 2)
    for i in 1:length(range)
        sigma = [range[i] sqrt(range[i])*0.35*sqrt(range[i]);
                        sqrt(range[i])*0.35*sqrt(range[i]) range[i]]
        P = GBM2(n, mu, sigma, seed = 1, dt = 1/n)

        CFT[i] = CFTcorrFK(P, t)[2][1,1]
        FGG[i] = NUFFTcorrFKFGG(P, t)[2][1,1]
        KB[i] = NUFFTcorrFKKB(P, t)[2][1,1]
        ES[i] = NUFFTcorrFKES(P, t)[2][1,1]
        FINUFFT[i] = NUFFTcorrFKFINUFFT(P, t)[2][1,1]
    end
    return CFT, FGG, KB, ES, FINUFFT
end

function sigma12Fej(n, ρ)
    CFT = zeros(length(ρ), 1)
    FGG = zeros(length(ρ), 1)
    KB = zeros(length(ρ), 1)
    ES = zeros(length(ρ), 1)
    FINUFFT = zeros(length(ρ), 1)

    mu = [0.01, 0.01]
    t = reshape([collect(1:1:n); collect(1:1:n)], n, 2)

    σ_12 = sqrt(0.1*0.1) .* ρ
    for i in 1:length(ρ)
        sigma = [0.1 σ_12[i]; σ_12[i] 0.1]
        P = GBM2(n, mu, sigma, seed = 1, dt = 1/n)

        CFT[i] = CFTcorrFK(P, t)[2][1,2]
        FGG[i] = NUFFTcorrFKFGG(P, t)[2][1,2]
        KB[i] = NUFFTcorrFKKB(P, t)[2][1,2]
        ES[i] = NUFFTcorrFKES(P, t)[2][1,2]
        FINUFFT[i] = NUFFTcorrFKFINUFFT(P, t)[2][1,2]
    end
    return CFT, FGG, KB, ES, FINUFFT
end


#---------------------------------------------------------------------------
# Obtain results

σ = collect(0.1:0.01:0.3)
ρ = collect(range(-0.99,0.99, length = 21))
σ_12 = sqrt(0.1*0.1) .* ρ


Dir11 = sigma11Dir(10^4, σ)
Dir12 = sigma12Dir(10^4, ρ)

Fej11 = sigma11Fej(10^4, σ)
Fej12 = sigma12Fej(10^4, ρ)

# Save
save("Computed Data/Accuracy/SensitivityDir.jld", "Dir11", Dir11, "Dir12", Dir12)
save("Computed Data/Accuracy/SensitivityFej.jld", "Fej11", Fej11, "Fej12", Fej12)

# Load
SensitivityDir = load("Computed Data/Accuracy/SensitivityDir.jld")
Dir11 = SensitivityDir["Dir11"]
Dir12 = SensitivityDir["Dir12"]

SensitivityFej = load("Computed Data/Accuracy/SensitivityFej.jld")
Fej11 = SensitivityFej["Fej11"]
Fej12 = SensitivityFej["Fej12"]


# Plot

# Dir

p1 = plot(σ, σ, label = "True", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}", legend = :bottomright)
plot!(p1, σ, Dir11[1], label = "CFT", color = :purple, line = (1, :solid), marker=([:utri :d],3,1,stroke(3.5,:purple)))
plot!(p1, σ, Dir11[2], label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p1, σ, Dir11[3], label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p1, σ, Dir11[4], label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p1, σ, Dir11[5], label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p1, L"\int_0^T \Sigma^{11}(t) dt")
ylabel!(p1, L"\hat{\Sigma}^{11}_{n_1,N}")
# title!(p1, L"\textrm{(a) Estimated and simulated (Dirichlet)}")

savefig(p1, "Plots/MM-NUFFT/SensitivityDir11.svg")


p2 = plot(σ_12, σ_12, label = "True", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}", legend = :bottomright)
plot!(p2, σ_12, Dir12[1], label = "CFT", color = :purple, line = (1, :solid), marker=([:utri :d],3,1,stroke(3.5,:purple)))
plot!(p2, σ_12, Dir12[2], label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p2, σ_12, Dir12[3], label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p2, σ_12, Dir12[4], label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p2, σ_12, Dir12[5], label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p2, L"\int_0^T \Sigma^{12}(t) dt")
ylabel!(p2, L"\hat{\Sigma}^{12}_{n_1,n_2,N}")
# title!(p2, L"\textrm{(b) Estimated and simulated (Dirichlet)}")

savefig(p2, "Plots/MM-NUFFT/SensitivityDir12.svg")


# Fej

p3 = plot(σ, σ, label = "True", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}", legend = :bottomright)
plot!(p3, σ, Fej11[1], label = "CFT", color = :purple, line = (1, :solid), marker=([:utri :d],3,1,stroke(3.5,:purple)))
plot!(p3, σ, Fej11[2], label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p3, σ, Fej11[3], label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p3, σ, Fej11[4], label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p3, σ, Fej11[5], label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p3, L"\int_0^T \Sigma^{11}(t) dt")
ylabel!(p3, L"\hat{\Sigma}^{11}_{n_1,N}")
# title!(p3, L"\textrm{(c) Estimated and simulated (Fej\'{e}r)}")

savefig(p3, "Plots/MM-NUFFT/SensitivityFej11.svg")


p4 = plot(σ_12, σ_12, label = "True", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}", legend = :bottomright)
plot!(p4, σ_12, Fej12[1], label = "CFT", color = :purple, line = (1, :solid), marker=([:utri :d],3,1,stroke(3.5,:purple)))
plot!(p4, σ_12, Fej12[2], label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p4, σ_12, Fej12[3], label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p4, σ_12, Fej12[4], label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p4, σ_12, Fej12[5], label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p4, L"\int_0^T \Sigma^{12}(t) dt")
ylabel!(p4, L"\hat{\Sigma}^{12}_{n_1,n_2,N}")
# title!(p4, L"\textrm{(d) Estimated and simulated (Fej\'{e}r)}")

savefig(p4, "Plots/MM-NUFFT/SensitivityFej12.svg")
