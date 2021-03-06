## Author: Patrick Chang
# Script file to investigate the impact of MSE and Bias under
# Regular non-synchronous trading. Comparing the implementations to
# ensure they recover the same MSE and Bias.

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
# Parameters
mu = [0.01, 0.01]
sigma = [0.1 sqrt(0.1)*0.35*sqrt(0.2);
                sqrt(0.1)*0.35*sqrt(0.2) 0.2]
#---------------------------------------------------------------------------
## Functions to steamline results
# Dirichlet

function rexp(n, mean)
    t = -mean .* log.(rand(n))
end

function MSEBiasDir(n, reps)
    N = collect(15:1:Int(floor(n/2)))

    BiasCFT11 = zeros(reps, length(N)); MSECFT11 = zeros(reps, length(N))
    BiasFGG11 = zeros(reps, length(N)); MSEFGG11 = zeros(reps, length(N))
    BiasKB11 = zeros(reps, length(N)); MSEKB11 = zeros(reps, length(N))
    BiasES11 = zeros(reps, length(N)); MSEES11 = zeros(reps, length(N))
    BiasFINUFFT11 = zeros(reps, length(N)); MSEFINUFFT11 = zeros(reps, length(N))

    BiasCFT12 = zeros(reps, length(N)); MSECFT12 = zeros(reps, length(N))
    BiasFGG12 = zeros(reps, length(N)); MSEFGG12 = zeros(reps, length(N))
    BiasKB12 = zeros(reps, length(N)); MSEKB12 = zeros(reps, length(N))
    BiasES12 = zeros(reps, length(N)); MSEES12 = zeros(reps, length(N))
    BiasFINUFFT12 = zeros(reps, length(N)); MSEFINUFFT12 = zeros(reps, length(N))

    @showprogress "Computing..." for i in 1:reps
        P = GBM2(n, mu, sigma, seed = i, dt = 1/n)
        t = reshape([collect(1:1:n); collect(1:1:n)], n, 2)
        t = t .* 1.0

        rm2 = collect(2:2:n-1)

        P[rm2, 2] .= NaN
        t[rm2, 2] .= NaN


        for j in 1:length(N)
            CFT = CFTcorrDK(P, t, N = N[j])[2]
            BiasCFT11[i,j] = (CFT[1,1] - 0.1) / 0.1
            BiasCFT12[i,j] = (CFT[1,2] - sqrt(0.1)*0.35*sqrt(0.2)) / (sqrt(0.1)*0.35*sqrt(0.2))
            MSECFT11[i,j] = BiasCFT11[i,j]^2
            MSECFT12[i,j] = BiasCFT12[i,j]^2

            FGG = NUFFTcorrDKFGG(P, t, N = N[j])[2]
            BiasFGG11[i,j] = (FGG[1,1] - 0.1) / 0.1
            BiasFGG12[i,j] = (FGG[1,2] - sqrt(0.1)*0.35*sqrt(0.2)) / (sqrt(0.1)*0.35*sqrt(0.2))
            MSEFGG11[i,j] = BiasFGG11[i,j]^2
            MSEFGG12[i,j] = BiasFGG12[i,j]^2

            KB = NUFFTcorrDKKB(P, t, N = N[j])[2]
            BiasKB11[i,j] = (KB[1,1] - 0.1) / 0.1
            BiasKB12[i,j] = (KB[1,2] - sqrt(0.1)*0.35*sqrt(0.2)) / (sqrt(0.1)*0.35*sqrt(0.2))
            MSEKB11[i,j] = BiasKB11[i,j]^2
            MSEKB12[i,j] = BiasKB12[i,j]^2

            ES = NUFFTcorrDKES(P, t, N = N[j])[2]
            BiasES11[i,j] = (ES[1,1] - 0.1) / 0.1
            BiasES12[i,j] = (ES[1,2] - sqrt(0.1)*0.35*sqrt(0.2)) / (sqrt(0.1)*0.35*sqrt(0.2))
            MSEES11[i,j] = BiasES11[i,j]^2
            MSEES12[i,j] = BiasES12[i,j]^2

            FINUFFT = NUFFTcorrDKFINUFFT(P, t, N = N[j])[2]
            BiasFINUFFT11[i,j] = (FINUFFT[1,1] - 0.1) / 0.1
            BiasFINUFFT12[i,j] = (FINUFFT[1,2] - sqrt(0.1)*0.35*sqrt(0.2)) / (sqrt(0.1)*0.35*sqrt(0.2))
            MSEFINUFFT11[i,j] = BiasFINUFFT11[i,j]^2
            MSEFINUFFT12[i,j] = BiasFINUFFT12[i,j]^2
        end
    end
    CFTres = Dict("Bias11" => BiasCFT11, "Bias12" => BiasCFT12, "MSE11" => MSECFT11, "MSE12" => MSECFT12)
    FGGres = Dict("Bias11" => BiasFGG11, "Bias12" => BiasFGG12, "MSE11" => MSEFGG11, "MSE12" => MSEFGG12)
    KBres = Dict("Bias11" => BiasKB11, "Bias12" => BiasKB12, "MSE11" => MSEKB11, "MSE12" => MSEKB12)
    ESres = Dict("Bias11" => BiasES11, "Bias12" => BiasES12, "MSE11" => MSEES11, "MSE12" => MSEES12)
    FINUFFTres = Dict("Bias11" => BiasFINUFFT11, "Bias12" => BiasFINUFFT12, "MSE11" => MSEFINUFFT11, "MSE12" => MSEFINUFFT12)

    return CFTres, FGGres, KBres, ESres, FINUFFTres
end

# Fejer

function MSEBiasFej(n, reps)
    N = collect(15:1:Int(floor(n/2)))

    BiasCFT11 = zeros(reps, length(N)); MSECFT11 = zeros(reps, length(N))
    BiasFGG11 = zeros(reps, length(N)); MSEFGG11 = zeros(reps, length(N))
    BiasKB11 = zeros(reps, length(N)); MSEKB11 = zeros(reps, length(N))
    BiasES11 = zeros(reps, length(N)); MSEES11 = zeros(reps, length(N))
    BiasFINUFFT11 = zeros(reps, length(N)); MSEFINUFFT11 = zeros(reps, length(N))

    BiasCFT12 = zeros(reps, length(N)); MSECFT12 = zeros(reps, length(N))
    BiasFGG12 = zeros(reps, length(N)); MSEFGG12 = zeros(reps, length(N))
    BiasKB12 = zeros(reps, length(N)); MSEKB12 = zeros(reps, length(N))
    BiasES12 = zeros(reps, length(N)); MSEES12 = zeros(reps, length(N))
    BiasFINUFFT12 = zeros(reps, length(N)); MSEFINUFFT12 = zeros(reps, length(N))

    @showprogress "Computing..." for i in 1:reps
        P = GBM2(n, mu, sigma, seed = i, dt = 1/n)
        t = reshape([collect(1:1:n); collect(1:1:n)], n, 2)
        t = t .* 1.0

        rm2 = collect(2:2:n-1)

        P[rm2, 2] .= NaN
        t[rm2, 2] .= NaN


        for j in 1:length(N)
            CFT = CFTcorrFK(P, t, N = N[j])[2]
            BiasCFT11[i,j] = (CFT[1,1] - 0.1) / 0.1
            BiasCFT12[i,j] = (CFT[1,2] - sqrt(0.1)*0.35*sqrt(0.2)) / (sqrt(0.1)*0.35*sqrt(0.2))
            MSECFT11[i,j] = BiasCFT11[i,j]^2
            MSECFT12[i,j] = BiasCFT12[i,j]^2

            FGG = NUFFTcorrFKFGG(P, t, N = N[j])[2]
            BiasFGG11[i,j] = (FGG[1,1] - 0.1) / 0.1
            BiasFGG12[i,j] = (FGG[1,2] - sqrt(0.1)*0.35*sqrt(0.2)) / (sqrt(0.1)*0.35*sqrt(0.2))
            MSEFGG11[i,j] = BiasFGG11[i,j]^2
            MSEFGG12[i,j] = BiasFGG12[i,j]^2

            KB = NUFFTcorrFKKB(P, t, N = N[j])[2]
            BiasKB11[i,j] = (KB[1,1] - 0.1) / 0.1
            BiasKB12[i,j] = (KB[1,2] - sqrt(0.1)*0.35*sqrt(0.2)) / (sqrt(0.1)*0.35*sqrt(0.2))
            MSEKB11[i,j] = BiasKB11[i,j]^2
            MSEKB12[i,j] = BiasKB12[i,j]^2

            ES = NUFFTcorrFKES(P, t, N = N[j])[2]
            BiasES11[i,j] = (ES[1,1] - 0.1) / 0.1
            BiasES12[i,j] = (ES[1,2] - sqrt(0.1)*0.35*sqrt(0.2)) / (sqrt(0.1)*0.35*sqrt(0.2))
            MSEES11[i,j] = BiasES11[i,j]^2
            MSEES12[i,j] = BiasES12[i,j]^2

            FINUFFT = NUFFTcorrFKFINUFFT(P, t, N = N[j])[2]
            BiasFINUFFT11[i,j] = (FINUFFT[1,1] - 0.1) / 0.1
            BiasFINUFFT12[i,j] = (FINUFFT[1,2] - sqrt(0.1)*0.35*sqrt(0.2)) / (sqrt(0.1)*0.35*sqrt(0.2))
            MSEFINUFFT11[i,j] = BiasFINUFFT11[i,j]^2
            MSEFINUFFT12[i,j] = BiasFINUFFT12[i,j]^2
        end
    end
    CFTres = Dict("Bias11" => BiasCFT11, "Bias12" => BiasCFT12, "MSE11" => MSECFT11, "MSE12" => MSECFT12)
    FGGres = Dict("Bias11" => BiasFGG11, "Bias12" => BiasFGG12, "MSE11" => MSEFGG11, "MSE12" => MSEFGG12)
    KBres = Dict("Bias11" => BiasKB11, "Bias12" => BiasKB12, "MSE11" => MSEKB11, "MSE12" => MSEKB12)
    ESres = Dict("Bias11" => BiasES11, "Bias12" => BiasES12, "MSE11" => MSEES11, "MSE12" => MSEES12)
    FINUFFTres = Dict("Bias11" => BiasFINUFFT11, "Bias12" => BiasFINUFFT12, "MSE11" => MSEFINUFFT11, "MSE12" => MSEFINUFFT12)

    return CFTres, FGGres, KBres, ESres, FINUFFTres
end

#---------------------------------------------------------------------------
# Obtain results
n = 100
reps = 10000

DirRes = MSEBiasDir(n, reps)
FejRes = MSEBiasFej(n, reps)

# Save
save("Computed Data/Accuracy/MSEBiasDir.jld", "DirRes", DirRes)
save("Computed Data/Accuracy/MSEBiasFej.jld", "FejRes", FejRes)

# Load
DirRes = load("Computed Data/Accuracy/MSEBiasDir.jld")
DirRes = DirRes["DirRes"]

FejRes = load("Computed Data/Accuracy/MSEBiasFej.jld")
FejRes = FejRes["FejRes"]

# Plot
N = collect(15:1:Int(floor(n/2)))

# Dir.
p1 = plot(N, mean(DirRes[1]["Bias11"], dims = 1)', ylims = (-1, 0.5), label = "CFT", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}", legend = :bottomright)
plot!(p1, N, mean(DirRes[2]["Bias11"], dims = 1)', ylims = (-1, 0.5), label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p1, N, mean(DirRes[3]["Bias11"], dims = 1)', ylims = (-1, 0.5), label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p1, N, mean(DirRes[4]["Bias11"], dims = 1)', ylims = (-1, 0.5), label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p1, N, mean(DirRes[5]["Bias11"], dims = 1)', ylims = (-1, 0.5), label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p1, L"\textrm{\# of Fourier coefficients }N")
ylabel!(p1, L"\textrm{Bias } \hat{\Sigma}^{11}_{n_1,N}")
# title!(p1, L"\textrm{(a) Bias and \# of Fourier coefficients (Dirichlet)}")


savefig(p1, "Plots/MM-NUFFT/Bias11Dir.svg")


p2 = plot(N, mean(DirRes[1]["Bias12"], dims = 1)', ylims = (-1, 0.5), label = "CFT", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}")
plot!(p2, N, mean(DirRes[2]["Bias12"], dims = 1)', ylims = (-1, 0.5), label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p2, N, mean(DirRes[3]["Bias12"], dims = 1)', ylims = (-1, 0.5), label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p2, N, mean(DirRes[4]["Bias12"], dims = 1)', ylims = (-1, 0.5), label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p2, N, mean(DirRes[5]["Bias12"], dims = 1)', ylims = (-1, 0.5), label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p2, L"\textrm{\# of Fourier coefficients }N")
ylabel!(p2, L"\textrm{Bias } \hat{\Sigma}^{12}_{n_1,n_2,N}")
# title!(p2, L"\textrm{(b) Bias and \# of Fourier coefficients (Dirichlet)}")


savefig(p2, "Plots/MM-NUFFT/Bias12Dir.svg")


p3 = plot(N, mean(DirRes[1]["MSE11"], dims = 1)', ylims = (0, 0.5), label = "CFT", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}")
plot!(p3, N, mean(DirRes[2]["MSE11"], dims = 1)', ylims = (0, 0.5), label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p3, N, mean(DirRes[3]["MSE11"], dims = 1)', ylims = (0, 0.5), label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p3, N, mean(DirRes[4]["MSE11"], dims = 1)', ylims = (0, 0.5), label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p3, N, mean(DirRes[5]["MSE11"], dims = 1)', ylims = (0, 0.5), label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p3, L"\textrm{\# of Fourier coefficients }N")
ylabel!(p3, L"\textrm{MSE } \hat{\Sigma}^{11}_{n_1,N}")
# title!(p3, L"\textrm{(c) MSE and \# of Fourier coefficients (Dirichlet)}")


savefig(p3, "Plots/MM-NUFFT/MSE11Dir.svg")


p4 = plot(N, mean(DirRes[1]["MSE12"], dims = 1)', ylims = (0, 0.5), label = "CFT", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}", legend = :bottomright)
plot!(p4, N, mean(DirRes[2]["MSE12"], dims = 1)', ylims = (0, 0.5), label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p4, N, mean(DirRes[3]["MSE12"], dims = 1)', ylims = (0, 0.5), label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p4, N, mean(DirRes[4]["MSE12"], dims = 1)', ylims = (0, 0.5), label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p4, N, mean(DirRes[5]["MSE12"], dims = 1)', ylims = (0, 0.5), label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p4, L"\textrm{\# of Fourier coefficients }N")
ylabel!(p4, L"\textrm{MSE } \hat{\Sigma}^{12}_{n_1,n_2,N}")
# title!(p4, L"\textrm{(d) MSE and \# of Fourier coefficients (Dirichlet)}")


savefig(p4, "Plots/MM-NUFFT/MSE12Dir.svg")



# Fej.
p1 = plot(N, mean(FejRes[1]["Bias11"], dims = 1)', ylims = (-1, 0.5), label = "CFT", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}", legend = :bottomright)
plot!(p1, N, mean(FejRes[2]["Bias11"], dims = 1)', ylims = (-1, 0.5), label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p1, N, mean(FejRes[3]["Bias11"], dims = 1)', ylims = (-1, 0.5), label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p1, N, mean(FejRes[4]["Bias11"], dims = 1)', ylims = (-1, 0.5), label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p1, N, mean(FejRes[5]["Bias11"], dims = 1)', ylims = (-1, 0.5), label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p1, L"\textrm{\# of Fourier coefficients }N")
ylabel!(p1, L"\textrm{Bias } \hat{\Sigma}^{11}_{n_1,N}")
# title!(p1, L"\textrm{(a) Bias and \# of Fourier coefficients (Fej\'{e}r)}")


savefig(p1, "Plots/MM-NUFFT/Bias11Fej.svg")


p2 = plot(N, mean(FejRes[1]["Bias12"], dims = 1)', ylims = (-1, 0.5), label = "CFT", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}")
plot!(p2, N, mean(FejRes[2]["Bias12"], dims = 1)', ylims = (-1, 0.5), label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p2, N, mean(FejRes[3]["Bias12"], dims = 1)', ylims = (-1, 0.5), label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p2, N, mean(FejRes[4]["Bias12"], dims = 1)', ylims = (-1, 0.5), label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p2, N, mean(FejRes[5]["Bias12"], dims = 1)', ylims = (-1, 0.5), label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p2, L"\textrm{\# of Fourier coefficients }N")
ylabel!(p2, L"\textrm{Bias } \hat{\Sigma}^{12}_{n_1,n_2,N}")
# title!(p2, L"\textrm{(b) Bias and \# of Fourier coefficients (Fej\'{e}r)}")


savefig(p2, "Plots/MM-NUFFT/Bias12Fej.svg")


p3 = plot(N, mean(FejRes[1]["MSE11"], dims = 1)', ylims = (0, 0.5), label = "CFT", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}")
plot!(p3, N, mean(FejRes[2]["MSE11"], dims = 1)', ylims = (0, 0.5), label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p3, N, mean(FejRes[3]["MSE11"], dims = 1)', ylims = (0, 0.5), label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p3, N, mean(FejRes[4]["MSE11"], dims = 1)', ylims = (0, 0.5), label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p3, N, mean(FejRes[5]["MSE11"], dims = 1)', ylims = (0, 0.5), label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p3, L"\textrm{\# of Fourier coefficients }N")
ylabel!(p3, L"\textrm{MSE } \hat{\Sigma}^{11}_{n_1,N}")
# title!(p3, L"\textrm{(c) MSE and \# of Fourier coefficients (Fej\'{e}r)}")


savefig(p3, "Plots/MM-NUFFT/MSE11Fej.svg")


p4 = plot(N, mean(FejRes[1]["MSE12"], dims = 1)', ylims = (0, 0.5), label = "CFT", color = :black, line = (2.5, :dash), dpi = 300, legendtitle=L"\textrm{Method}", legend = :topright)
plot!(p4, N, mean(FejRes[2]["MSE12"], dims = 1)', ylims = (0, 0.5), label = "FGG", color = :blue, line = (1, :solid), marker=([:circ :d],3,1,stroke(2,:blue)))
plot!(p4, N, mean(FejRes[3]["MSE12"], dims = 1)', ylims = (0, 0.5), label = "KB", color = :brown, line = (1, :solid), marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p4, N, mean(FejRes[4]["MSE12"], dims = 1)', ylims = (0, 0.5), label = "ES", color = :red, line = (1, :solid), marker=([:rect :d],3,1,stroke(2,:red)))
plot!(p4, N, mean(FejRes[5]["MSE12"], dims = 1)', ylims = (0, 0.5), label = "FINUFFT", color = :green, line = (1, :solid), marker=([:cross :d],3,1,stroke(3,:green)))
xlabel!(p4, L"\textrm{\# of Fourier coefficients }N")
ylabel!(p4, L"\textrm{MSE } \hat{\Sigma}^{12}_{n_1,n_2,N}")
# title!(p4, L"\textrm{(d) MSE and \# of Fourier coefficients (Fej\'{e}r)}")


savefig(p4, "Plots/MM-NUFFT/MSE12Fej.svg")
