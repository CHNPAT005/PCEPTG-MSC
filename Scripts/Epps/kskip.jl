

T = 3600*20

ρ = theoreticalCorr(0.023, 0.05, 0.11)

par_1 = BarcyParams(0.015, 0.023, 0.05, 0.11)
lambda0_1 = par_1[1]; alpha_1 = par_1[2]; beta_1 = par_1[3]
t1 = simulateHawkes(lambda0_1, alpha_1, beta_1, T, seed = 19549293)

p1_1 = getuniformPrices(0, 1, T, t1[1], t1[2])
p2_1 = getuniformPrices(0, 1, T, t1[3], t1[4])
P_1 = [p1_1 p2_1]


reps = 1
kskip = collect(1:1:50)
HYlam_lam1 = zeros(length(kskip), reps)
HYlam_lam10 = zeros(length(kskip), reps)
HYlam_lam25 = zeros(length(kskip), reps)

# Takes roughly 2.5 hours to compute
for k in 1:reps
    lam1 = 1
    Random.seed!(k)
    t1_lam1 = [0; rexp(T, lam1)]
    t1_lam1 = cumsum(t1_lam1)
    t1_lam1 = filter((x) -> x < T, t1_lam1)
    Random.seed!(k+reps)
    t2_lam1 = [0; rexp(T, lam1)]
    t2_lam1 = cumsum(t2_lam1)
    t2_lam1 = filter((x) -> x < T, t2_lam1)

    lam10 = 10
    Random.seed!(k)
    t1_lam10 = [0; rexp(T, lam10)]
    t1_lam10 = cumsum(t1_lam10)
    t1_lam10 = filter((x) -> x < T, t1_lam10)
    Random.seed!(k+reps)
    t2_lam10 = [0; rexp(T, lam10)]
    t2_lam10 = cumsum(t2_lam10)
    t2_lam10 = filter((x) -> x < T, t2_lam10)

    lam25 = 25
    Random.seed!(k)
    t1_lam25 = [0; rexp(T, lam25)]
    t1_lam25 = cumsum(t1_lam25)
    t1_lam25 = filter((x) -> x < T, t1_lam25)
    Random.seed!(k+reps)
    t2_lam25 = [0; rexp(T, lam25)]
    t2_lam25 = cumsum(t2_lam25)
    t2_lam25 = filter((x) -> x < T, t2_lam25)

    @showprogress "Computing..." for i in 1:length(kskip)
        t1_lam1_ind = collect(1:kskip[i]:length(t1_lam1))
        t2_lam1_ind = collect(1:kskip[i]:length(t2_lam1))

        t1_lam1_temp = t1_lam1[t1_lam1_ind]
        t2_lam1_temp = t2_lam1[t2_lam1_ind]

        P1_lam1 = exp.(P_1[Int.(floor.(t1_lam1_temp).+1), 1])
        P2_lam1 = exp.(P_1[Int.(floor.(t2_lam1_temp).+1), 2])

        t1_lam10_ind = collect(1:kskip[i]:length(t1_lam10))
        t2_lam10_ind = collect(1:kskip[i]:length(t2_lam10))

        t1_lam10_temp = t1_lam10[t1_lam10_ind]
        t2_lam10_temp = t2_lam10[t2_lam10_ind]

        P1_lam10 = exp.(P_1[Int.(floor.(t1_lam10_temp).+1), 1])
        P2_lam10 = exp.(P_1[Int.(floor.(t2_lam10_temp).+1), 2])

        t1_lam25_ind = collect(1:kskip[i]:length(t1_lam25))
        t2_lam25_ind = collect(1:kskip[i]:length(t2_lam25))

        t1_lam25_temp = t1_lam25[t1_lam25_ind]
        t2_lam25_temp = t2_lam25[t2_lam25_ind]

        P1_lam25 = exp.(P_1[Int.(floor.(t1_lam25_temp).+1), 1])
        P2_lam25 = exp.(P_1[Int.(floor.(t2_lam25_temp).+1), 2])

        HYlam_lam1[i,k] = HYcorr(P1_lam1,P2_lam1,t1_lam1_temp,t2_lam1_temp)[1][1,2]
        HYlam_lam10[i,k] = HYcorr(P1_lam10,P2_lam10,t1_lam10_temp,t2_lam10_temp)[1][1,2]
        HYlam_lam25[i,k] = HYcorr(P1_lam25,P2_lam25,t1_lam25_temp,t2_lam25_temp)[1][1,2]
    end
end

p1 = plot(kskip, HYlam_lam1)
plot!(p1, kskip, HYlam_lam10)
plot!(p1, kskip, HYlam_lam25)


T = 3600*20
ρ = theoreticalCorr(0.023, 0.05, 0.11)

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*ρ*sqrt(0.2/86400);
        sqrt(0.1/86400)*ρ*sqrt(0.2/86400) 0.2/86400]

P_GBM = GBM(T+1, mu, sigma)


HYlam_lam1_GBM = zeros(length(kskip), reps)
HYlam_lam10_GBM = zeros(length(kskip), reps)
HYlam_lam25_GBM = zeros(length(kskip), reps)

# Takes roughly 2.5 hours to compute
for k in 1:reps
    lam1 = 1
    Random.seed!(k)
    t1_lam1 = [0; rexp(T, lam1)]
    t1_lam1 = cumsum(t1_lam1)
    t1_lam1 = filter((x) -> x < T, t1_lam1)
    Random.seed!(k+reps)
    t2_lam1 = [0; rexp(T, lam1)]
    t2_lam1 = cumsum(t2_lam1)
    t2_lam1 = filter((x) -> x < T, t2_lam1)

    lam10 = 10
    Random.seed!(k)
    t1_lam10 = [0; rexp(T, lam10)]
    t1_lam10 = cumsum(t1_lam10)
    t1_lam10 = filter((x) -> x < T, t1_lam10)
    Random.seed!(k+reps)
    t2_lam10 = [0; rexp(T, lam10)]
    t2_lam10 = cumsum(t2_lam10)
    t2_lam10 = filter((x) -> x < T, t2_lam10)

    lam25 = 25
    Random.seed!(k)
    t1_lam25 = [0; rexp(T, lam25)]
    t1_lam25 = cumsum(t1_lam25)
    t1_lam25 = filter((x) -> x < T, t1_lam25)
    Random.seed!(k+reps)
    t2_lam25 = [0; rexp(T, lam25)]
    t2_lam25 = cumsum(t2_lam25)
    t2_lam25 = filter((x) -> x < T, t2_lam25)

    @showprogress "Computing..." for i in 1:length(kskip)
        t1_lam1_ind = collect(1:kskip[i]:length(t1_lam1))
        t2_lam1_ind = collect(1:kskip[i]:length(t2_lam1))

        t1_lam1_temp = t1_lam1[t1_lam1_ind]
        t2_lam1_temp = t2_lam1[t2_lam1_ind]

        P1_lam1 = (P_GBM[Int.(floor.(t1_lam1_temp).+1), 1])
        P2_lam1 = (P_GBM[Int.(floor.(t2_lam1_temp).+1), 2])

        t1_lam10_ind = collect(1:kskip[i]:length(t1_lam10))
        t2_lam10_ind = collect(1:kskip[i]:length(t2_lam10))

        t1_lam10_temp = t1_lam10[t1_lam10_ind]
        t2_lam10_temp = t2_lam10[t2_lam10_ind]

        P1_lam10 = (P_GBM[Int.(floor.(t1_lam10_temp).+1), 1])
        P2_lam10 = (P_GBM[Int.(floor.(t2_lam10_temp).+1), 2])

        t1_lam25_ind = collect(1:kskip[i]:length(t1_lam25))
        t2_lam25_ind = collect(1:kskip[i]:length(t2_lam25))

        t1_lam25_temp = t1_lam25[t1_lam25_ind]
        t2_lam25_temp = t2_lam25[t2_lam25_ind]

        P1_lam25 = (P_GBM[Int.(floor.(t1_lam25_temp).+1), 1])
        P2_lam25 = (P_GBM[Int.(floor.(t2_lam25_temp).+1), 2])

        HYlam_lam1_GBM[i,k] = HYcorr(P1_lam1,P2_lam1,t1_lam1_temp,t2_lam1_temp)[1][1,2]
        HYlam_lam10_GBM[i,k] = HYcorr(P1_lam10,P2_lam10,t1_lam10_temp,t2_lam10_temp)[1][1,2]
        HYlam_lam25_GBM[i,k] = HYcorr(P1_lam25,P2_lam25,t1_lam25_temp,t2_lam25_temp)[1][1,2]
    end
end

p1 = plot(kskip, HYlam_lam1_GBM)
plot!(p1, kskip, HYlam_lam10_GBM)
plot!(p1, kskip, HYlam_lam25_GBM)




function computecorrs(data, T = 28200, kskip = collect(1:1:50))
    m = size(data)[1]
    HY = zeros(m, length(kskip))

    @showprogress "Computing..." for k in 1:m
        temp = data[k]
        t1 = temp[findall(!isnan, temp[:,2]),1]
        t2 = temp[findall(!isnan, temp[:,3]),1]

        P1 = filter(!isnan, temp[:,2])
        P2 = filter(!isnan, temp[:,3])
        for i in 1:length(kskip)
            t1_ind = collect(1:kskip[i]:length(t1))
            t2_ind = collect(1:kskip[i]:length(t2))

            t1_temp = t1[t1_ind]
            t2_temp = t2[t2_ind]

            P1_temp = P1[t1_ind]
            P2_temp = P2[t2_ind]

            HY[k,i] = HYcorr(P1_temp,P2_temp,t1_temp,t2_temp)[1][1,2]
        end
    end
    return HY
end


SBKFSR = Empirical(7, 11, JSE_Data)
NEDABG = Empirical(8, 9, JSE_Data)

p1 = plot([mean(SBKFSR[4])], ribbon=(q .* std(SBKFSR[4], dims = 1)), fillalpha=.15, color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")


p1 = plot(collect(1:1:50), mean(SBKFSR,dims=1)', label = "SBK/FSR", legend = :bottomright)
plot!(p1 , collect(1:1:50), mean(NEDABG,dims=1)', label = "NED/ABG")
