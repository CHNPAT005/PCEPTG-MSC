## Author: Patrick Chang
# Script file to benchmark performance between various NUFFT algorithms
# for both the Dirichlet and Fejer implementaion.
# We plot the compute time as a function of the errors
# and we use the FFT and Zero Padded FFT as a baseline for comparison.

using ProgressMeter, JLD, LaTeXStrings, Plots

cd("/Users/patrickchang1/PCEPTG-MSC")

include("../../Functions/Correlation Estimators/Dirichlet/FFTcorrDK.jl")
include("../../Functions/Correlation Estimators/Dirichlet/FFTZPcorrDK.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-KB.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-ES.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FINUFFT.jl")

include("../../Functions/Correlation Estimators/Fejer/FFTcorrFK.jl")
include("../../Functions/Correlation Estimators/Fejer/FFTZPcorrFK.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FGG.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-KB.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-ES.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FINUFFT.jl")

include("../../Functions/SDEs/GBM.jl")
include("../../Functions/SDEs/RandCovMat.jl")

#---------------------------------------------------------------------------
# Timing Functions
#---------------------------------------------------------------------------

## Dirichlet

function timeFGGDK(n, tolrange, reps)
    result = zeros(reps, length(tolrange))
    @showprogress "Computing..." for i in 1:length(tolrange)
        P = GBM(n, mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed NUFFTcorrDKFGG(P, t, tol = tolrange[i])
        end
        GC.gc()
    end
    return result
end

function timeKBDK(n, tolrange, reps)
    result = zeros(reps, length(tolrange))
    @showprogress "Computing..." for i in 1:length(tolrange)
        P = GBM(n, mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed NUFFTcorrDKKB(P, t, tol = tolrange[i])
        end
        GC.gc()
    end
    return result
end

function timeESDK(n, tolrange, reps)
    result = zeros(reps, length(tolrange))
    @showprogress "Computing..." for i in 1:length(tolrange)
        P = GBM(n, mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed NUFFTcorrDKES(P, t, tol = tolrange[i])
        end
        GC.gc()
    end
    return result
end

function timeFINUFFTDK(n, tolrange, reps)
    result = zeros(reps, length(tolrange))
    @showprogress "Computing..." for i in 1:length(tolrange)
        P = GBM(n, mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed NUFFTcorrDKFINUFFT(P, t, tol = tolrange[i])
        end
        GC.gc()
    end
    return result
end

function timeFFTDK(n, reps)
    result = zeros(reps, 1)
    P = GBM(n, mu, sigma, seed = 1)
    t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
    for j in 1:reps
        result[j] = @elapsed FFTcorrDK(P)
    end
    GC.gc()
    return result
end

function timeFFTZPDK(n, reps)
    result = zeros(reps, 1)
    P = GBM(n, mu, sigma, seed = 1)
    t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
    for j in 1:reps
        result[j] = @elapsed FFTZPcorrDK(P, t)
    end
    GC.gc()
    return result
end

## Fejer

function timeFGGFK(n, tolrange, reps)
    result = zeros(reps, length(tolrange))
    @showprogress "Computing..." for i in 1:length(tolrange)
        P = GBM(n, mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed NUFFTcorrFKFGG(P, t, tol = tolrange[i])
        end
        GC.gc()
    end
    return result
end

function timeKBFK(n, tolrange, reps)
    result = zeros(reps, length(tolrange))
    @showprogress "Computing..." for i in 1:length(tolrange)
        P = GBM(n, mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed NUFFTcorrFKKB(P, t, tol = tolrange[i])
        end
        GC.gc()
    end
    return result
end

function timeESFK(n, tolrange, reps)
    result = zeros(reps, length(tolrange))
    @showprogress "Computing..." for i in 1:length(tolrange)
        P = GBM(n, mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed NUFFTcorrFKES(P, t, tol = tolrange[i])
        end
        GC.gc()
    end
    return result
end

function timeFINUFFTFK(n, tolrange, reps)
    result = zeros(reps, length(tolrange))
    @showprogress "Computing..." for i in 1:length(tolrange)
        P = GBM(n, mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed NUFFTcorrFKFINUFFT(P, t, tol = tolrange[i])
        end
        GC.gc()
    end
    return result
end

function timeFFTFK(n, reps)
    result = zeros(reps, 1)
    P = GBM(n, mu, sigma, seed = 1)
    t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
    for j in 1:reps
        result[j] = @elapsed FFTcorrFK(P)
    end
    GC.gc()
    return result
end

function timeFFTZPFK(n, reps)
    result = zeros(reps, 1)
    P = GBM(n, mu, sigma, seed = 1)
    t = reshape(repeat(collect(1:1:n), size(P)[2]), n , size(P)[2])
    for j in 1:reps
        result[j] = @elapsed FFTZPcorrFK(P, t)
    end
    GC.gc()
    return result
end

#---------------------------------------------------------------------------
## 2 Assets
#---------------------------------------------------------------------------
## Obtain and save results

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

n1 = collect(-3:-1:-14)
FGGtol = 10.0.^n1
n2 = collect(-4:-2:-14)
# gsp = Int.(floor.(-log.(tol)./(pi*(R-1)/(R-0.5)) .+ 0.5))
# ksp = Int.(floor.((ceil.(log.(10, 1 ./ tol)) .+ 2)./2))
reps = 10
KBtol = [10^-3; 10.0.^n2]

## Dirichlet

FGGtimesDK = timeFGGDK(100000, FGGtol, reps)
save("Computed Data/Error times/FGGtimesDK.jld", "FGGtimesDK", FGGtimesDK)

KBtimesDK = timeKBDK(100000, KBtol, reps)
save("Computed Data/Error times/KBtimesDK.jld", "KBtimesDK", KBtimesDK)

EStimesDK = timeESDK(100000, KBtol, reps)
save("Computed Data/Error times/EStimesDK.jld", "EStimesDK", EStimesDK)

FINUFFTtimesDK = timeFINUFFTDK(100000, KBtol, reps)
save("Computed Data/Error times/FINUFFTtimesDK.jld", "FINUFFTtimesDK", FINUFFTtimesDK)

FFTtimesDK = timeFFTDK(100000, reps)
save("Computed Data/Error times/FFTtimesDK.jld", "FFTtimesDK", FFTtimesDK)

FFTZPtimesDK = timeFFTZPDK(100000, reps)
save("Computed Data/Error times/FFTZPtimesDK.jld", "FFTZPtimesDK", FFTZPtimesDK)

## Fejer

FGGtimesFK = timeFGGFK(100000, FGGtol, reps)
save("Computed Data/Error times/FGGtimesFK.jld", "FGGtimesFK", FGGtimesFK)

KBtimesFK = timeKBFK(100000, KBtol, reps)
save("Computed Data/Error times/KBtimesFK.jld", "KBtimesFK", KBtimesFK)

EStimesFK = timeESFK(100000, KBtol, reps)
save("Computed Data/Error times/EStimesFK.jld", "EStimesFK", EStimesFK)

FINUFFTtimesFK = timeFINUFFTFK(100000, KBtol, reps)
save("Computed Data/Error times/FINUFFTtimesFK.jld", "FINUFFTtimesFK", FINUFFTtimesFK)

FFTtimesFK = timeFFTFK(100000, reps)
save("Computed Data/Error times/FFTtimesFK.jld", "FFTtimesFK", FFTtimesFK)

FFTZPtimesFK = timeFFTZPFK(100000, reps)
save("Computed Data/Error times/FFTZPtimesFK.jld", "FFTZPtimesFK", FFTZPtimesFK)

# Load and plot results
#---------------------------------------------------------------------------
## Load
# Dirichlet
FGGtimesDK = load("Computed Data/Error times/FGGtimesDK.jld")
FGGtimesDK = FGGtimesDK["FGGtimesDK"]

KBtimesDK = load("Computed Data/Error times/KBtimesDK.jld")
KBtimesDK = KBtimesDK["KBtimesDK"]

EStimesDK = load("Computed Data/Error times/EStimesDK.jld")
EStimesDK = EStimesDK["EStimesDK"]

FINUFFTtimesDK = load("Computed Data/Error times/FINUFFTtimesDK.jld")
FINUFFTtimesDK = FINUFFTtimesDK["FINUFFTtimesDK"]

FFTtimesDK = load("Computed Data/Error times/FFTtimesDK.jld")
FFTtimesDK = FFTtimesDK["FFTtimesDK"]

FFTZPtimesDK = load("Computed Data/Error times/FFTZPtimesDK.jld")
FFTZPtimesDK = FFTZPtimesDK["FFTZPtimesDK"]

# Fejer
FGGtimesFK = load("Computed Data/Error times/FGGtimesFK.jld")
FGGtimesFK = FGGtimesFK["FGGtimesFK"]

KBtimesFK = load("Computed Data/Error times/KBtimesFK.jld")
KBtimesFK = KBtimesFK["KBtimesFK"]

EStimesFK = load("Computed Data/Error times/EStimesFK.jld")
EStimesFK = EStimesFK["EStimesFK"]

FINUFFTtimesFK = load("Computed Data/Error times/FINUFFTtimesFK.jld")
FINUFFTtimesFK = FINUFFTtimesFK["FINUFFTtimesFK"]

FFTtimesFK = load("Computed Data/Error times/FFTtimesFK.jld")
FFTtimesFK = FFTtimesFK["FFTtimesFK"]

FFTZPtimesFK = load("Computed Data/Error times/FFTZPtimesFK.jld")
FFTZPtimesFK = FFTZPtimesFK["FFTZPtimesFK"]

## Plot

n1 = collect(-3:-1:-14)
FGGtol = 10.0.^n1
n2 = collect(-4:-2:-14)
KBtol = [10^-3; 10.0.^n2]

xticklabel = [L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

# Dirichlet
p1 = plot(n1, log.(minimum(FGGtimesDK, dims=1)'), xticks = (n1, xticklabel), legend=:topright, legendtitle="Method", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), dpi = 300, ylims = (-4, 5))
plot!(p1, [-3; n2], log.(minimum(KBtimesDK, dims=1)'), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p1, [-3; n2], log.(minimum(FINUFFTtimesDK, dims=1)'), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot!(p1, [-3; n2], log.(minimum(EStimesDK, dims=1)'), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
hline!(p1, log.([minimum(FFTtimesDK)]), label = "FFT", line=(1.5, [:dash]), color = :purple)
hline!(p1, log.([minimum(FFTZPtimesDK)]), label = "ZFFT", line=(1.5, [:dashdot]), color = :gray)
xlabel!(p1, L"\textrm{Tolerance }\epsilon")
ylabel!(p1, L"\textrm{Time } [\ln(s)]")

savefig(p1, "Plots/MM-NUFFT/ErrorTimesDK2.svg")


# Fejer
p2 = plot(n1, log.(minimum(FGGtimesFK, dims=1)'), xticks = (n1, xticklabel), legend=:topright, legendtitle="Method", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), dpi = 300, ylims = (-4, 5))
plot!(p2, [-3; n2], log.(minimum(KBtimesFK, dims=1)'), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p2, [-3; n2], log.(minimum(FINUFFTtimesFK, dims=1)'), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot!(p2, [-3; n2], log.(minimum(EStimesFK, dims=1)'), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
hline!(p2, log.([minimum(FFTtimesFK)]), label = "FFT", line=(1.5, [:dash]), color = :purple)
hline!(p2, log.([minimum(FFTZPtimesFK)]), label = "ZFFT", line=(1.5, [:dashdot]), color = :gray)
xlabel!(p2, L"\textrm{Tolerance }\epsilon")
ylabel!(p2, L"\textrm{Time } [\ln(s)]")

savefig(p2, "Plots/MM-NUFFT/ErrorTimesFK2.svg")


#---------------------------------------------------------------------------
## 10 Assets
#---------------------------------------------------------------------------
## Obtain and save results

mu = repeat([0.01/86400], 10)
sig = repeat([sqrt(0.1/86400)], 10)
sigma = gencovmatrix(10, sig)

n1 = collect(-3:-1:-14)
FGGtol = 10.0.^n1
n2 = collect(-4:-2:-14)
# gsp = Int.(floor.(-log.(tol)./(pi*(R-1)/(R-0.5)) .+ 0.5))
# ksp = Int.(floor.((ceil.(log.(10, 1 ./ tol)) .+ 2)./2))
reps = 10
KBtol = [10^-3; 10.0.^n2]

## Dirichlet

FGGtimesDK10 = timeFGGDK(100000, FGGtol, reps)
save("Computed Data/Error times/FGGtimesDK10.jld", "FGGtimesDK10", FGGtimesDK10)

KBtimesDK10 = timeKBDK(100000, KBtol, reps)
save("Computed Data/Error times/KBtimesDK10.jld", "KBtimesDK10", KBtimesDK10)

EStimesDK10 = timeESDK(100000, KBtol, reps)
save("Computed Data/Error times/EStimesDK10.jld", "EStimesDK10", EStimesDK10)

FINUFFTtimesDK10 = timeFINUFFTDK(100000, KBtol, reps)
save("Computed Data/Error times/FINUFFTtimesDK10.jld", "FINUFFTtimesDK10", FINUFFTtimesDK10)

FFTtimesDK10 = timeFFTDK(100000, reps)
save("Computed Data/Error times/FFTtimesDK10.jld", "FFTtimesDK10", FFTtimesDK10)

FFTZPtimesDK10 = timeFFTZPDK(100000, reps)
save("Computed Data/Error times/FFTZPtimesDK10.jld", "FFTZPtimesDK10", FFTZPtimesDK10)

## Fejer

FGGtimesFK10 = timeFGGFK(100000, FGGtol, reps)
save("Computed Data/Error times/FGGtimesFK10.jld", "FGGtimesFK10", FGGtimesFK10)

KBtimesFK10 = timeKBFK(100000, KBtol, reps)
save("Computed Data/Error times/KBtimesFK10.jld", "KBtimesFK10", KBtimesFK10)

EStimesFK10 = timeESFK(100000, KBtol, reps)
save("Computed Data/Error times/EStimesFK10.jld", "EStimesFK10", EStimesFK10)

FINUFFTtimesFK10 = timeFINUFFTFK(100000, KBtol, reps)
save("Computed Data/Error times/FINUFFTtimesFK10.jld", "FINUFFTtimesFK10", FINUFFTtimesFK10)

FFTtimesFK10 = timeFFTFK(100000, reps)
save("Computed Data/Error times/FFTtimesFK10.jld", "FFTtimesFK10", FFTtimesFK10)

FFTZPtimesFK10 = timeFFTZPFK(100000, reps)
save("Computed Data/Error times/FFTZPtimesFK10.jld", "FFTZPtimesFK10", FFTZPtimesFK10)

# Load and plot results
#---------------------------------------------------------------------------
## Load
# Dirichlet
FGGtimesDK10 = load("Computed Data/Error times/FGGtimesDK10.jld")
FGGtimesDK10 = FGGtimesDK10["FGGtimesDK10"]

KBtimesDK10 = load("Computed Data/Error times/KBtimesDK10.jld")
KBtimesDK10 = KBtimesDK10["KBtimesDK10"]

EStimesDK10 = load("Computed Data/Error times/EStimesDK10.jld")
EStimesDK10 = EStimesDK10["EStimesDK10"]

FINUFFTtimesDK10 = load("Computed Data/Error times/FINUFFTtimesDK10.jld")
FINUFFTtimesDK10 = FINUFFTtimesDK10["FINUFFTtimesDK10"]

FFTtimesDK10 = load("Computed Data/Error times/FFTtimesDK10.jld")
FFTtimesDK10 = FFTtimesDK10["FFTtimesDK10"]

FFTZPtimesDK10 = load("Computed Data/Error times/FFTZPtimesDK10.jld")
FFTZPtimesDK10 = FFTZPtimesDK10["FFTZPtimesDK10"]

# Fejer
FGGtimesFK10 = load("Computed Data/Error times/FGGtimesFK10.jld")
FGGtimesFK10 = FGGtimesFK10["FGGtimesFK10"]

KBtimesFK10 = load("Computed Data/Error times/KBtimesFK10.jld")
KBtimesFK10 = KBtimesFK10["KBtimesFK10"]

EStimesFK10 = load("Computed Data/Error times/EStimesFK10.jld")
EStimesFK10 = EStimesFK10["EStimesFK10"]

FINUFFTtimesFK10 = load("Computed Data/Error times/FINUFFTtimesFK10.jld")
FINUFFTtimesFK10 = FINUFFTtimesFK10["FINUFFTtimesFK10"]

FFTtimesFK10 = load("Computed Data/Error times/FFTtimesFK10.jld")
FFTtimesFK10 = FFTtimesFK10["FFTtimesFK10"]

FFTZPtimesFK10 = load("Computed Data/Error times/FFTZPtimesFK10.jld")
FFTZPtimesFK10 = FFTZPtimesFK10["FFTZPtimesFK10"]

## Plot

n1 = collect(-3:-1:-14)
FGGtol = 10.0.^n1
n2 = collect(-4:-2:-14)
KBtol = [10^-3; 10.0.^n2]

xticklabel = [L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

# Dirichlet
p3 = plot(n1, log.(minimum(FGGtimesDK10, dims=1)'), xticks = (n1, xticklabel), legend=:topright, legendtitle="Method", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), dpi = 300, ylims = (-4, 5))
plot!(p3, [-3; n2], log.(minimum(KBtimesDK10, dims=1)'), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p3, [-3; n2], log.(minimum(FINUFFTtimesDK10, dims=1)'), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot!(p3, [-3; n2], log.(minimum(EStimesDK10, dims=1)'), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
hline!(p3, log.([minimum(FFTtimesDK10)]), label = "FFT", line=(1.5, [:dash]), color = :purple)
hline!(p3, log.([minimum(FFTZPtimesDK10)]), label = "ZFFT", line=(1.5, [:dashdot]), color = :gray)
xlabel!(p3, L"\textrm{Tolerance }\epsilon")
ylabel!(p3, L"\textrm{Time } [\ln(s)]")

savefig(p3, "Plots/MM-NUFFT/ErrorTimesDK10.svg")


# Fejer
p4 = plot(n1, log.(minimum(FGGtimesFK10, dims=1)'), xticks = (n1, xticklabel), legend=:topright, legendtitle="Method", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), dpi = 300, ylims = (-4, 5))
plot!(p4, [-3; n2], log.(minimum(KBtimesFK10, dims=1)'), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p4, [-3; n2], log.(minimum(FINUFFTtimesFK10, dims=1)'), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot!(p4, [-3; n2], log.(minimum(EStimesFK10, dims=1)'), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
hline!(p4, log.([minimum(FFTtimesFK10)]), label = "FFT", line=(1.5, [:dash]), color = :purple)
hline!(p4, log.([minimum(FFTZPtimesFK10)]), label = "ZFFT", line=(1.5, [:dashdot]), color = :gray)
xlabel!(p4, L"\textrm{Tolerance }\epsilon")
ylabel!(p4, L"\textrm{Time } [\ln(s)]")

savefig(p4, "Plots/MM-NUFFT/ErrorTimesFK10.svg")


#---------------------------------------------------------------------------
## 100 Assets
#---------------------------------------------------------------------------
## Obtain and save results

mu = repeat([0.01/86400], 100)
sig = repeat([sqrt(0.1/86400)], 100)
sigma = gencovmatrix(100, sig)

n1 = collect(-3:-1:-14)
FGGtol = 10.0.^n1
n2 = collect(-4:-2:-14)
# gsp = Int.(floor.(-log.(tol)./(pi*(R-1)/(R-0.5)) .+ 0.5))
# ksp = Int.(floor.((ceil.(log.(10, 1 ./ tol)) .+ 2)./2))
reps = 10
KBtol = [10^-3; 10.0.^n2]

## Dirichlet

FGGtimesDK100 = timeFGGDK(1000000, FGGtol, reps)
save("Computed Data/Error times/FGGtimesDK100.jld", "FGGtimesDK100", FGGtimesDK100)

KBtimesDK100 = timeKBDK(1000000, KBtol, reps)
save("Computed Data/Error times/KBtimesDK100.jld", "KBtimesDK100", KBtimesDK100)

EStimesDK100 = timeESDK(1000000, KBtol, reps)
save("Computed Data/Error times/EStimesDK100.jld", "EStimesDK100", EStimesDK100)

FINUFFTtimesDK100 = timeFINUFFTDK(1000000, KBtol, reps)
save("Computed Data/Error times/FINUFFTtimesDK100.jld", "FINUFFTtimesDK100", FINUFFTtimesDK100)

FFTtimesDK100 = timeFFTDK(1000000, reps)
save("Computed Data/Error times/FFTtimesDK100.jld", "FFTtimesDK100", FFTtimesDK100)

FFTZPtimesDK100 = timeFFTZPDK(1000000, reps)
save("Computed Data/Error times/FFTZPtimesDK100.jld", "FFTZPtimesDK100", FFTZPtimesDK100)

## Fejer

FGGtimesFK100 = timeFGGFK(1000000, FGGtol, reps)
save("Computed Data/Error times/FGGtimesFK100.jld", "FGGtimesFK100", FGGtimesFK100)

KBtimesFK100 = timeKBFK(1000000, KBtol, reps)
save("Computed Data/Error times/KBtimesFK100.jld", "KBtimesFK100", KBtimesFK100)

EStimesFK100 = timeESFK(1000000, KBtol, reps)
save("Computed Data/Error times/EStimesFK100.jld", "EStimesFK100", EStimesFK100)

FINUFFTtimesFK100 = timeFINUFFTFK(1000000, KBtol, reps)
save("Computed Data/Error times/FINUFFTtimesFK100.jld", "FINUFFTtimesFK100", FINUFFTtimesFK100)

FFTtimesFK100 = timeFFTFK(1000000, reps)
save("Computed Data/Error times/FFTtimesFK100.jld", "FFTtimesFK100", FFTtimesFK100)

FFTZPtimesFK100 = timeFFTZPFK(1000000, reps)
save("Computed Data/Error times/FFTZPtimesFK100.jld", "FFTZPtimesFK100", FFTZPtimesFK100)

# Load and plot results
#---------------------------------------------------------------------------
## Load
# Dirichlet
FGGtimesDK100 = load("Computed Data/Error times/FGGtimesDK100.jld")
FGGtimesDK100 = FGGtimesDK100["FGGtimesDK100"]

KBtimesDK100 = load("Computed Data/Error times/KBtimesDK100.jld")
KBtimesDK100 = KBtimesDK100["KBtimesDK100"]

EStimesDK100 = load("Computed Data/Error times/EStimesDK100.jld")
EStimesDK100 = EStimesDK100["EStimesDK100"]

FINUFFTtimesDK100 = load("Computed Data/Error times/FINUFFTtimesDK100.jld")
FINUFFTtimesDK100 = FINUFFTtimesDK100["FINUFFTtimesDK100"]

FFTtimesDK100 = load("Computed Data/Error times/FFTtimesDK100.jld")
FFTtimesDK100 = FFTtimesDK100["FFTtimesDK100"]

FFTZPtimesDK100 = load("Computed Data/Error times/FFTZPtimesDK100.jld")
FFTZPtimesDK100 = FFTZPtimesDK100["FFTZPtimesDK100"]

# Fejer
FGGtimesFK100 = load("Computed Data/Error times/FGGtimesFK100.jld")
FGGtimesFK100 = FGGtimesFK100["FGGtimesFK100"]

KBtimesFK100 = load("Computed Data/Error times/KBtimesFK100.jld")
KBtimesFK100 = KBtimesFK100["KBtimesFK100"]

EStimesFK100 = load("Computed Data/Error times/EStimesFK100.jld")
EStimesFK100 = EStimesFK100["EStimesFK100"]

FINUFFTtimesFK100 = load("Computed Data/Error times/FINUFFTtimesFK100.jld")
FINUFFTtimesFK100 = FINUFFTtimesFK100["FINUFFTtimesFK100"]

FFTtimesFK100 = load("Computed Data/Error times/FFTtimesFK100.jld")
FFTtimesFK100 = FFTtimesFK100["FFTtimesFK100"]

FFTZPtimesFK100 = load("Computed Data/Error times/FFTZPtimesFK100.jld")
FFTZPtimesFK100 = FFTZPtimesFK100["FFTZPtimesFK100"]

## Plot

n1 = collect(-3:-1:-14)
FGGtol = 10.0.^n1
n2 = collect(-4:-2:-14)
KBtol = [10^-3; 10.0.^n2]

xticklabel = [L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}", L"10^{-7}", L"10^{-8}", L"10^{-9}", L"10^{-10}", L"10^{-11}", L"10^{-12}", L"10^{-13}", L"10^{-14}"]

# Dirichlet
p5 = plot(n1, log.(minimum(FGGtimesDK100, dims=1)'), xticks = (n1, xticklabel), legend=:bottomright, legendtitle="Method", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), dpi = 300, ylims = (-4, 5))
plot!(p5, [-3; n2], log.(minimum(KBtimesDK100, dims=1)'), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p5, [-3; n2], log.(minimum(FINUFFTtimesDK100, dims=1)'), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot!(p5, [-3; n2], log.(minimum(EStimesDK100, dims=1)'), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
hline!(p5, log.([minimum(FFTtimesDK100)]), label = "FFT", line=(1.5, [:dash]), color = :purple)
hline!(p5, log.([minimum(FFTZPtimesDK100)]), label = "ZFFT", line=(1.5, [:dashdot]), color = :gray)
xlabel!(p5, L"\textrm{Tolerance }\epsilon")
ylabel!(p5, L"\textrm{Time } [\ln(s)]")

savefig(p5, "Plots/MM-NUFFT/ErrorTimesDK100.svg")


# Fejer
p6 = plot(n1, log.(minimum(FGGtimesFK100, dims=1)'), xticks = (n1, xticklabel), legend=:bottomright, legendtitle="Method", color = :blue, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:blue)), dpi = 300, ylims = (-4, 5))
plot!(p6, [-3; n2], log.(minimum(KBtimesFK100, dims=1)'), color = :brown, line=(2, [:dot]), label = L"\textrm{KB}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p6, [-3; n2], log.(minimum(FINUFFTtimesFK100, dims=1)'), color = :green, line=(2, [:dot]), label = L"\textrm{FINUFFT}", marker=([:cross :d],3,1,stroke(3,:green)))
plot!(p6, [-3; n2], log.(minimum(EStimesFK100, dims=1)'), color = :red, line=(2, [:dot]), label = L"\textrm{ES}", marker=([:rect :d],3,1,stroke(2,:red)))
hline!(p6, log.([minimum(FFTtimesFK100)]), label = "FFT", line=(1.5, [:dash]), color = :purple)
hline!(p6, log.([minimum(FFTZPtimesFK100)]), label = "ZFFT", line=(1.5, [:dashdot]), color = :gray)
xlabel!(p6, L"\textrm{Tolerance }\epsilon")
ylabel!(p6, L"\textrm{Time } [\ln(s)]")

savefig(p6, "Plots/MM-NUFFT/ErrorTimesFK100.svg")
