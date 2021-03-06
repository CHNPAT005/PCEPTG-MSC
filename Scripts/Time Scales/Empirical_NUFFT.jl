## Author: Patrick Chang
# Script file to test the averaging scale of the MM estimator
# using real TAQ data from the JSE for the top 10 most liquid stocks

# The data is cleaned by aggregating trades with the same trade time using
# a VWAP average.

using JLD; using LaTeXStrings; using Plots; using Statistics; using CSV;
using Optim; using Distributions

#---------------------------------------------------------------------------

cd("/Users/patrickchang1/PCEPTG-MSC")

include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")

include("../../Functions/Correlation Estimators/Fejer/NUFFTcorrFK-FGG.jl")

## Read in the data
p = CSV.read("Real Data/JSE_prices_2019-06-24_2019-06-28.csv")
t = CSV.read("Real Data/JSE_times_2019-06-24_2019-06-28.csv")

tickers = String.(names(p))[2:11]

p = convert(Matrix, p[:,2:11])
t = convert(Matrix, t[:,2:11])

#---------------------------------------------------------------------------
## Run and time results
dt = collect(1:1:100)

T = (3600*7+50*60) * 5

N = Int.(floor.(((T ./dt).-1.0) ./ 2))

Dir = reshape(repeat(zeros(10, 10), length(dt)), 10, 10, length(dt))
Fej = reshape(repeat(zeros(10, 10), length(dt)), 10, 10, length(dt))

function timeDK()
    for i in 1:length(dt)
        Dir[:,:,i] = NUFFTcorrDKFGG(p, t, N = N[i])[1]
    end
end

function timeFK()
    for i in 1:length(dt)
        Fej[:,:,i] = NUFFTcorrFKFGG(p, t, N = N[i])[1]
    end
end

# Run this at least once - preferably use the result from thrid run onwards
DKtime = @elapsed timeDK()
FKtime = @elapsed timeFK()

#---------------------------------------------------------------------------
## Save and Load

save("Computed Data/Empirical/DKtime.jld", "DKtime", DKtime)
save("Computed Data/Empirical/FKtime.jld", "FKtime", FKtime)

save("Computed Data/Empirical/DKres.jld", "DKres", Dir)
save("Computed Data/Empirical/FKres.jld", "FKres", Fej)

Dir = load("Computed Data/Empirical/DKres.jld")
Dir = Dir["DKres"]

Fej = load("Computed Data/Empirical/FKres.jld")
Fej = Fej["FKres"]

#---------------------------------------------------------------------------
# MMZ plots all asset pairs
p1 = plot(x = dt, legend = :outertopleft, legendtitle="Asset Pair", legendfont = font("Times new roman", 7), size = (1000, 800),
            ylims = (-0.1, 0.65), dpi = 300)
ylabel!(p1, L"\rho_{\Delta t}^{ij}")
xlabel!(p1, L"\Delta t \textrm{ [sec]}")
for i in 1:9
    for j in i+1:10
        Dirres = zeros(length(dt), 1)
        for k in 1:length(dt)
            Dirres[k] = Dir[:,:,k][i,j]
        end
        plot!(p1, dt, Dirres, label = tickers[i]*"/"*tickers[j])
    end
end
current()

savefig(p1, "Plots/MM-NUFFT/EmpDKMMZALL.svg")



p2 = plot(x = dt, legend = :outertopleft, legendtitle="Asset Pair", legendfont = font("Times new roman", 7), size = (1000, 800),
            ylims = (-0.1, 0.65), dpi = 300)
ylabel!(p2, L"\rho_{\Delta t}^{ij}")
xlabel!(p2, L"\Delta t \textrm{ [sec]}")
for i in 1:9
    for j in i+1:10
        Fejres = zeros(length(dt), 1)
        for k in 1:length(dt)
            Fejres[k] = Fej[:,:,k][i,j]
        end
        plot!(p2, dt, Fejres, label = tickers[i]*"/"*tickers[j])
    end
end
current()

savefig(p2, "Plots/MM-NUFFT/EmpFKMMZALL.svg")

#---------------------------------------------------------------------------
## Heatmaps for various Δt's

# Dirichlet
p1 = plot(Dir[:,:,1], st=:heatmap, clim=(-1,1), color=cgrad([:red, :white, :blue]), colorbar_title=L"\rho_{\Delta t}^{ij}", xticks = (1:10, tickers), yticks = (1:10, tickers), dpi = 300, size = (800, 700), tickfontsize = 15)
savefig(p1, "Plots/MM-NUFFT/EmpDKHM1.png")

p2 = plot(Dir[:,:,30], st=:heatmap, clim=(-1,1), color=cgrad([:red, :white, :blue]), colorbar_title=L"\rho_{\Delta t}^{ij}",xticks = (1:10, tickers), yticks = (1:10, tickers), dpi = 300, size = (800, 700), tickfontsize = 15)
savefig(p2, "Plots/MM-NUFFT/EmpDKHM30.png")

p3 = plot(Dir[:,:,60], st=:heatmap, clim=(-1,1), color=cgrad([:red, :white, :blue]), colorbar_title=L"\rho_{\Delta t}^{ij}",xticks = (1:10, tickers), yticks = (1:10, tickers), dpi = 300, size = (800, 700), tickfontsize = 15)
savefig(p3, "Plots/MM-NUFFT/EmpDKHM60.png")

p4 = plot(Dir[:,:,100], st=:heatmap, clim=(-1,1), color=cgrad([:red, :white, :blue]), colorbar_title=L"\rho_{\Delta t}^{ij}",xticks = (1:10, tickers), yticks = (1:10, tickers), dpi = 300, size = (800, 700), tickfontsize = 15)
savefig(p4, "Plots/MM-NUFFT/EmpDKHM100.png")

# Fejer
p1 = plot(Fej[:,:,1], st=:heatmap, clim=(-1,1), color=cgrad([:red, :white, :blue]), colorbar_title=L"\rho_{\Delta t}^{ij}", xticks = (1:10, tickers), yticks = (1:10, tickers), dpi = 300, size = (800, 700), tickfontsize = 15)
savefig(p1, "Plots/MM-NUFFT/EmpFKHM1.png")

p2 = plot(Fej[:,:,30], st=:heatmap, clim=(-1,1), color=cgrad([:red, :white, :blue]), colorbar_title=L"\rho_{\Delta t}^{ij}",xticks = (1:10, tickers), yticks = (1:10, tickers), dpi = 300, size = (800, 700), tickfontsize = 15)
savefig(p2, "Plots/MM-NUFFT/EmpFKHM30.png")

p3 = plot(Fej[:,:,60], st=:heatmap, clim=(-1,1), color=cgrad([:red, :white, :blue]), colorbar_title=L"\rho_{\Delta t}^{ij}",xticks = (1:10, tickers), yticks = (1:10, tickers), dpi = 300, size = (800, 700), tickfontsize = 15)
savefig(p3, "Plots/MM-NUFFT/EmpFKHM60.png")

p4 = plot(Fej[:,:,100], st=:heatmap, clim=(-1,1), color=cgrad([:red, :white, :blue]), colorbar_title=L"\rho_{\Delta t}^{ij}",xticks = (1:10, tickers), yticks = (1:10, tickers), dpi = 300, size = (800, 700), tickfontsize = 15)
savefig(p4, "Plots/MM-NUFFT/EmpFKHM100.png")

#---------------------------------------------------------------------------
# Obtaining Bootstrap CIs by doing a CV estimation

p = p[:,[3,6,10]]
t = t[:,[3,6,10]]

steps = 555
n = size(p)[1]

p1 = p[steps+1:n,:]
t1 = t[steps+1:n,:]

p100 = p[1:steps*99,:]
t100 = t[1:steps*99,:]

DKFSRSBK_error = zeros(100, 100)
DKFSRAGL_error = zeros(100, 100)

FKFSRSBK_error = zeros(100, 100)
FKFSRAGL_error = zeros(100, 100)

for i in 1:100
    if i == 1
        for j in 1:length(dt)
            Dirichlet = NUFFTcorrDKFGG(p1, t1, N = N[j])[1]
            Fejer = NUFFTcorrFKFGG(p1, t1, N = N[j])[1]

            DKFSRSBK_error[i,j] = Dirichlet[2,3]
            DKFSRAGL_error[i,j] = Dirichlet[1,3]

            FKFSRSBK_error[i,j] = Fejer[2,3]
            FKFSRAGL_error[i,j] = Fejer[1,3]
        end
    elseif i == 100
        for j in 1:length(dt)
            Dirichlet = NUFFTcorrDKFGG(p100, t100, N = N[j])[1]
            Fejer = NUFFTcorrFKFGG(p100, t100, N = N[j])[1]

            DKFSRSBK_error[i,j] = Dirichlet[2,3]
            DKFSRAGL_error[i,j] = Dirichlet[1,3]

            FKFSRSBK_error[i,j] = Fejer[2,3]
            FKFSRAGL_error[i,j] = Fejer[1,3]
        end
    else
        index = [collect(1:1:steps*(i-1)); collect(i*steps+1:1:n)]
        pstore = p[index,:]
        tstore = t[index,:]
        for j in 1:length(dt)
            Dirichlet = NUFFTcorrDKFGG(pstore, tstore, N = N[j])[1]
            Fejer = NUFFTcorrFKFGG(pstore, tstore, N = N[j])[1]

            DKFSRSBK_error[i,j] = Dirichlet[2,3]
            DKFSRAGL_error[i,j] = Dirichlet[1,3]

            FKFSRSBK_error[i,j] = Fejer[2,3]
            FKFSRAGL_error[i,j] = Fejer[1,3]
        end
    end
end

save("Computed Data/Empirical/DKFSRSBK_error.jld", "DKFSRSBK_error", DKFSRSBK_error)
save("Computed Data/Empirical/DKFSRAGL_error.jld", "DKFSRAGL_error", DKFSRAGL_error)
save("Computed Data/Empirical/FKFSRSBK_error.jld", "FKFSRSBK_error", FKFSRSBK_error)
save("Computed Data/Empirical/FKFSRAGL_error.jld", "FKFSRAGL_error", FKFSRAGL_error)

DKFSRSBK_error = load("Computed Data/Empirical/DKFSRSBK_error.jld")
DKFSRSBK_error = DKFSRSBK_error["DKFSRSBK_error"]
DKFSRAGL_error = load("Computed Data/Empirical/DKFSRAGL_error.jld")
DKFSRAGL_error = DKFSRAGL_error["DKFSRAGL_error"]
FKFSRSBK_error = load("Computed Data/Empirical/FKFSRSBK_error.jld")
FKFSRSBK_error = FKFSRSBK_error["FKFSRSBK_error"]
FKFSRAGL_error = load("Computed Data/Empirical/FKFSRAGL_error.jld")
FKFSRAGL_error = FKFSRAGL_error["FKFSRAGL_error"]

#---------------------------------------------------------------------------
# MMZ plots 2 correlation pairs
q = quantile.(TDist(100-1), [0.975])

p1 = plot(x = dt, legend = :right, legendtitle="Asset Pair", legendfont = font("Times new roman", 7), dpi = 300, ylims = (-0.1, 0.7))
ylabel!(p1, L"\rho_{\Delta t}^{ij}")
xlabel!(p1, L"\Delta t \textrm{ [sec]}")
DKFSRSBK = zeros(length(dt), 1)
DKFSRAGL = zeros(length(dt), 1)
for k in 1:length(dt)
    DKFSRSBK[k] = Dir[:,:,k][6,10]
    DKFSRAGL[k] = Dir[:,:,k][3,10]
end
plot!(p1, dt, DKFSRSBK, label = "Measured-"*"FSR"*"/"*"SBK", color = :blue,
        ribbon = std(DKFSRSBK_error, dims=1)' .* q, fillalpha=.1, marker=([:x :d],3,1,stroke(2,:blue)))

plot!(p1, dt, DKFSRAGL, label = "Measured-"*"FSR"*"/"*"AGL", color = :red,
        ribbon = std(DKFSRAGL_error, dims=1)' .* q, fillalpha=.1, marker=([:+ :d],3,1,stroke(2,:red)))

lam2 = 1/13.52503
model = 6.21e-01 .* (1 .+ (exp.(-lam2 .* dt) .- 1) ./ (lam2 .* dt))
plot!(p1, dt, model, label ="Theoretical-"*"FSR"*"/"*"SBK", line = (2, :solid), color = :black)


savefig(p1, "Plots/MM-NUFFT/EmpDKMMZ2.svg")


p2 = plot(x = dt, legend = :right, legendtitle="Asset Pair", legendfont = font("Times new roman", 7), dpi = 300, ylims = (-0.1, 0.7))
ylabel!(p2, L"\rho_{\Delta t}^{ij}")
xlabel!(p2, L"\Delta t \textrm{ [sec]}")
FKFSRSBK = zeros(length(dt), 1)
FKFSRAGL = zeros(length(dt), 1)
for k in 1:length(dt)
    FKFSRSBK[k] = Fej[:,:,k][6,10]
    FKFSRAGL[k] = Fej[:,:,k][3,10]
end
plot!(p2, dt, FKFSRSBK, label = "Measured-"*"FSR"*"/"*"SBK", color = :blue,
        ribbon = std(FKFSRSBK_error, dims=1)' .* q, fillalpha=.1, marker=([:x :d],3,1,stroke(2,:blue)))

plot!(p2, dt, FKFSRAGL, label = "Measured-"*"FSR"*"/"*"AGL", color = :red,
        ribbon = std(FKFSRAGL_error, dims=1)' .* q, fillalpha=.1, marker=([:+ :d],3,1,stroke(2,:red)))


lam2 = 1/13.52503
model = 6.21e-01 .* (1 .+ (exp.(-lam2 .* dt) .- 1) ./ (lam2 .* dt))
plot!(p2, dt, model, label ="Theoretical-"*"FSR"*"/"*"SBK", line = (2, :solid), color = :black)

savefig(p1, "Plots/MM-NUFFT/EmpFKMMZ2.svg")



#---------------------------------------------------------------------------
# MMZ plots 2 correlation pairs - detailed plots with all the
# paths included.

p1 = plot(dt, DKFSRSBK_error', legend= false)
plot!(p1, dt, DKFSRSBK, label = "Measured-"*"FSR"*"/"*"AGL", color = :black,
        ribbon = std(DKFSRSBK_error, dims=1)' .* q, fillalpha=.3, line=(2, :dash))
plot!(p1, dt, DKFSRAGL_error')
plot!(p1, dt, DKFSRAGL, label = "Measured-"*"FSR"*"/"*"AGL", color = :black,
        ribbon = std(DKFSRAGL_error, dims=1)' .* q, fillalpha=.3, line=(2, :dash))
title!(p1, L"\textrm{\sffamily (a) Correlation and sampling interval (Dirichlet)}")
ylabel!(p1, L"\textrm{Correlation } \rho")
xlabel!(p1, L"\textrm{Sampling interval } \Delta t \textrm{ [seconds]}")

savefig(p1, "Plots/EmpDKMMZ2_detailed.pdf")

p2 = plot(dt, FKFSRSBK_error', legend= false)
plot!(p2, dt, FKFSRSBK, label = "Measured-"*"FSR"*"/"*"AGL", color = :black,
        ribbon = std(FKFSRSBK_error, dims=1)' .* q, fillalpha=.3, line=(2, :dash))
plot!(p2, dt, FKFSRAGL_error')
plot!(p2, dt, FKFSRAGL, label = "Measured-"*"FSR"*"/"*"AGL", color = :black,
        ribbon = std(FKFSRAGL_error, dims=1)' .* q, fillalpha=.3, line=(2, :dash))
title!(p2, L"\textrm{\sffamily (b) Correlation and sampling interval (Fej\'{e}r)}")
ylabel!(p2, L"\textrm{Correlation } \rho")
xlabel!(p2, L"\textrm{Sampling interval } \Delta t \textrm{ [seconds]}")

savefig(p2, "Plots/EmpFKMMZ2_detailed.pdf")
