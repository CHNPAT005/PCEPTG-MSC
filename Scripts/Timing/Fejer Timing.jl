## Author: Patrick Chang
# Script file to benchmark performance between various algorithms
# for the Fejer implementaion.
# We compare the Mancino-Sanfelici code to Legacy code and the FFT and NUFFT

using ProgressMeter, JLD, LaTeXStrings, Plots

cd("/Users/patrickchang1/PCEPTG-MSC")

include("../../Functions/Correlation Estimators/Fejer/CFTcorrFK.jl")
include("../../Functions/Correlation Estimators/Fejer/MScorrFK.jl")
include("../../Functions/Correlation Estimators/Fejer/FFTcorrFK.jl")
include("../../Functions/Correlation Estimators/Fejer/FFTZPcorrFK.jl")
include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FGG.jl")
include("../../Functions/SDEs/GBM.jl")
include("../../Functions/SDEs/RandCovMat.jl")

#---------------------------------------------------------------------------
# Timing Functions

function timeCFTcorrFK(nrange, reps)
    result = zeros(reps, length(nrange))
    @showprogress "Computing..." for i in 1:length(nrange)
        P = GBM(Int(nrange[i]), mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:nrange[i]), size(P)[2]), Int(nrange[i]) , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed CFTcorrFK(P, t)
        end
        GC.gc()
    end
    return result
end

function timeMScorrFK(nrange, reps)
    result = zeros(reps, length(nrange))
    @showprogress "Computing..." for i in 1:length(nrange)
        P = GBM(Int(nrange[i]), mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:nrange[i]), size(P)[2]), Int(nrange[i]) , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed MScorrFK(P, t)
        end
        GC.gc()
    end
    return result
end

function timeFFTcorrFK(nrange, reps)
    result = zeros(reps, length(nrange))
    @showprogress "Computing..." for i in 1:length(nrange)
        P = GBM(Int(nrange[i]), mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:nrange[i]), size(P)[2]), Int(nrange[i]) , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed FFTcorrFK(P)
        end
        GC.gc()
    end
    return result
end

function timeFFTZPcorrFK(nrange, reps)
    result = zeros(reps, length(nrange))
    @showprogress "Computing..." for i in 1:length(nrange)
        P = GBM(Int(nrange[i]), mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:nrange[i]), size(P)[2]), Int(nrange[i]) , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed FFTZPcorrFK(P, t)
        end
        GC.gc()
    end
    return result
end

function timeNUFFTcorrFKFGG(nrange, reps)
    result = zeros(reps, length(nrange))
    @showprogress "Computing..." for i in 1:length(nrange)
        P = GBM(Int(nrange[i]), mu, sigma, seed = i)
        t = reshape(repeat(collect(1:1:nrange[i]), size(P)[2]), Int(nrange[i]) , size(P)[2])
        for j in 1:reps
            result[j, i] = @elapsed NUFFTcorrFKFGG(P, t)
        end
        GC.gc()
    end
    return result
end

#---------------------------------------------------------------------------
## 2 Assets
#---------------------------------------------------------------------------
# Obtain and save results

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

nrange = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 40000.0]#, 100000.0]
reps = 10

# CFT can't go past 50,000 points - otherwise Julia just exits
CFTtimes = timeCFTcorrFK(nrange, reps)
save("Computed Data/ON times/CFTtimesFK.jld", "CFTtimes", CFTtimes)

nrange = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0]
MStimes = timeMScorrFK(nrange, reps)
save("Computed Data/ON times/MStimesFK.jld", "MStimes", MStimes)

FFTtimes = timeFFTcorrFK(nrange, reps)
save("Computed Data/ON times/FFTtimesFK.jld", "FFTtimes", FFTtimes)

FFTZPtimes = timeFFTZPcorrFK(nrange, reps)
save("Computed Data/ON times/FFTZPtimesFK.jld", "FFTZPtimes", FFTZPtimes)

FGGtimes = timeNUFFTcorrFKFGG(nrange, reps)
save("Computed Data/ON times/FGGtimesFK.jld", "FGGtimes", FGGtimes)

#---------------------------------------------------------------------------
# Load and plot results

CFTtimes = load("Computed Data/ON times/CFTtimesFK.jld")
CFTtimes = CFTtimes["CFTtimes"]

MStimes = load("Computed Data/ON times/MStimesFK.jld")
MStimes = MStimes["MStimes"]

FFTtimes = load("Computed Data/ON times/FFTtimesFK.jld")
FFTtimes = FFTtimes["FFTtimes"]

FFTZPtimes = load("Computed Data/ON times/FFTZPtimesFK.jld")
FFTZPtimes = FFTZPtimes["FFTZPtimes"]

FGGtimes = load("Computed Data/ON times/FGGtimesFK.jld")
FGGtimes = FGGtimes["FGGtimes"]


# Plot
nrange = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 40000.0, 100000.0]
nrange1 = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0]
p1 = plot(nrange, log.([minimum(CFTtimes, dims=1) NaN]'), legend = :bottomright, legendtitle="Method", color = :blue, line=(2, [:dot]), label = L"\textrm{CFT}", marker=([:+ :d],3,1,stroke(4,:blue)), dpi = 300, ylims = (-12, 8))
plot!(p1, nrange1, log.([minimum(MStimes, dims=1) NaN]'), color = :brown, line=(2, [:dot]), label = L"\textrm{MRS}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p1, nrange1, log.([minimum(FFTtimes, dims=1) NaN]'), color = :green, line=(2, [:dot]), label = L"\textrm{FFT}", marker=([:rect :d],3,1,stroke(2,:green)))
plot!(p1, nrange1, log.([minimum(FFTZPtimes, dims=1) NaN]'), color = :purple, line=(2, [:dot]), label = L"\textrm{ZFFT}", marker=([:utriangle :d],3,1,stroke(2,:purple)))
plot!(p1, nrange1, log.([minimum(FGGtimes, dims=1) NaN]'), color = :black, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:black)))
xlabel!(p1, L"\textrm{Data Points n, }\textrm{Cutoff } N = \frac{n}{2}")
ylabel!(p1, L"\textrm{Time } [\ln(s)]")

savefig(p1, "Plots/MM-NUFFT/ONFK2Assets.svg")


#---------------------------------------------------------------------------
## 10 Assets
#---------------------------------------------------------------------------
# Obtain and save results

mu = repeat([0.01/86400], 10)
sig = repeat([sqrt(0.1/86400)], 10)
sigma = gencovmatrix(10, sig)

nrange = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]
reps = 10

CFTtimes = timeCFTcorrFK(nrange, reps)
save("Computed Data/ON times/CFTtimesFK10.jld", "CFTtimes", CFTtimes)

MStimes = timeMScorrFK(nrange, reps)
save("Computed Data/ON times/MStimesFK10.jld", "MStimes", MStimes)

FFTtimes = timeFFTcorrFK(nrange, reps)
save("Computed Data/ON times/FFTtimesFK10.jld", "FFTtimes", FFTtimes)

FFTZPtimes = timeFFTZPcorrFK(nrange, reps)
save("Computed Data/ON times/FFTZPtimesFK10.jld", "FFTZPtimes", FFTZPtimes)

FGGtimes = timeNUFFTcorrFKFGG(nrange, reps)
save("Computed Data/ON times/FGGtimesFK10.jld", "FGGtimes", FGGtimes)

#---------------------------------------------------------------------------
# Load and plot results

CFTtimes = load("Computed Data/ON times/CFTtimesFK10.jld")
CFTtimes = CFTtimes["CFTtimes"]

MStimes = load("Computed Data/ON times/MStimesFK10.jld")
MStimes = MStimes["MStimes"]

FFTtimes = load("Computed Data/ON times/FFTtimesFK10.jld")
FFTtimes = FFTtimes["FFTtimes"]

FFTZPtimes = load("Computed Data/ON times/FFTZPtimesFK10.jld")
FFTZPtimes = FFTZPtimes["FFTZPtimes"]

FGGtimes = load("Computed Data/ON times/FGGtimesFK10.jld")
FGGtimes = FGGtimes["FGGtimes"]


# Plot
nrange = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]

p2 = plot(nrange, log.([minimum(CFTtimes, dims=1) NaN]'), legend = :bottomright, legendtitle="Method", color = :blue, line=(2, [:dot]), label = L"\textrm{CFT}", marker=([:+ :d],3,1,stroke(4,:blue)), dpi = 300, ylims = (-12, 8))
plot!(p2, nrange, log.([minimum(MStimes, dims=1) NaN]'), color = :brown, line=(2, [:dot]), label = L"\textrm{MRS}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p2, nrange, log.([minimum(FFTtimes, dims=1) NaN]'), color = :green, line=(2, [:dot]), label = L"\textrm{FFT}", marker=([:rect :d],3,1,stroke(2,:green)))
plot!(p2, nrange, log.([minimum(FFTZPtimes, dims=1) NaN]'), color = :purple, line=(2, [:dot]), label = L"\textrm{ZFFT}", marker=([:utriangle :d],3,1,stroke(2,:purple)))
plot!(p2, nrange, log.([minimum(FGGtimes, dims=1) NaN]'), color = :black, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:black)))
xlabel!(p2, L"\textrm{Data Points n, }\textrm{Cutoff } N = \frac{n}{2}")
ylabel!(p2, L"\textrm{Time } [\ln(s)]")

savefig(p2, "Plots/MM-NUFFT/ONFK10Assets.svg")

#---------------------------------------------------------------------------
## 100 Assets
#---------------------------------------------------------------------------
# Obtain and save results

mu = repeat([0.01/86400], 100)
sig = repeat([sqrt(0.1/86400)], 100)
sigma = gencovmatrix(100, sig)

nrange = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
reps = 10

CFTtimes = timeCFTcorrFK(nrange, reps)
save("Computed Data/ON times/CFTtimesFK100.jld", "CFTtimes", CFTtimes)

MStimes = timeMScorrFK(nrange, reps)
save("Computed Data/ON times/MStimesFK100.jld", "MStimes", MStimes)

FFTtimes = timeFFTcorrFK(nrange, reps)
save("Computed Data/ON times/FFTtimesFK100.jld", "FFTtimes", FFTtimes)

FFTZPtimes = timeFFTZPcorrFK(nrange, reps)
save("Computed Data/ON times/FFTZPtimesFK100.jld", "FFTZPtimes", FFTZPtimes)

FGGtimes = timeNUFFTcorrFKFGG(nrange, reps)
save("Computed Data/ON times/FGGtimesFK100.jld", "FGGtimes", FGGtimes)

#---------------------------------------------------------------------------
# Load and plot results

CFTtimes = load("Computed Data/ON times/CFTtimesFK100.jld")
CFTtimes = CFTtimes["CFTtimes"]

MStimes = load("Computed Data/ON times/MStimesFK100.jld")
MStimes = MStimes["MStimes"]

FFTtimes = load("Computed Data/ON times/FFTtimesFK100.jld")
FFTtimes = FFTtimes["FFTtimes"]

FFTZPtimes = load("Computed Data/ON times/FFTZPtimesFK100.jld")
FFTZPtimes = FFTZPtimes["FFTZPtimes"]

FGGtimes = load("Computed Data/ON times/FGGtimesFK100.jld")
FGGtimes = FGGtimes["FGGtimes"]

# Plot
nrange = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]

p3 = plot(nrange, log.([minimum(CFTtimes, dims=1) NaN]'), legend = :bottomright, legendtitle="Method", color = :blue, line=(2, [:dot]), label = L"\textrm{CFT}", marker=([:+ :d],3,1,stroke(4,:blue)), dpi = 300, ylims = (-12, 8))
plot!(p3, nrange, log.([minimum(MStimes, dims=1) NaN]'), color = :brown, line=(2, [:dot]), label = L"\textrm{MRS}", marker=([:x :d],3,1,stroke(2,:brown)))
plot!(p3, nrange, log.([minimum(FFTtimes, dims=1) NaN]'), color = :green, line=(2, [:dot]), label = L"\textrm{FFT}", marker=([:rect :d],3,1,stroke(2,:green)))
plot!(p3, nrange, log.([minimum(FFTZPtimes, dims=1) NaN]'), color = :purple, line=(2, [:dot]), label = L"\textrm{ZFFT}", marker=([:utriangle :d],3,1,stroke(2,:purple)))
plot!(p3, nrange, log.([minimum(FGGtimes, dims=1) NaN]'), color = :black, line=(2, [:dot]), label = L"\textrm{FGG}", marker=([:circ :d],3,1,stroke(2,:black)))
xlabel!(p3, L"\textrm{Data Points n, }\textrm{Cutoff } N = \frac{n}{2}")
ylabel!(p3, L"\textrm{Time } [\ln(s)]")

savefig(p3, "Plots/MM-NUFFT/ONFK100Assets.svg")
