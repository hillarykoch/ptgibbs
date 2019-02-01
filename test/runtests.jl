
using ptgibbs

import ptgibbs: run_mcmc
import StatsBase: rle
import RLEVectors: rep
import DataFrames: DataFrame, colwise, categorical!

using Test
using Distributed
using LinearAlgebra
using Lazy
using Distributions

"""
These are tests for the gibbs sampler
"""

# Instantiate parameters for test model
nw = 2; # number of walkers per temp
nt = 3; # number of temperatures
dm = 2; # dimension of data
mu1 = [0,0];
mu2 = [4,4];
mu3 = [-2.5,6];
#labs = ["11", "22", "02"]
sig1 = Matrix{Float64}(I,dm,dm)
sig2 = Matrix{Float64}(I,dm,dm)*1.2 .+ .7;
sig3 = reshape([.85, -.6, -.6,.95], (2,2));
prop = [.2,.45, .35];
nm = size(prop)[1]; # Number of clusters
n = 1000;

# Simulate the labels/data
zprop = @as zprop Multinomial(n, prop) begin
        @> rand(zprop)
        @> rep(collect(1:1:nm); each = zprop)
end;

dat = hcat(
        hcat(rand(MvNormal(mu1, sig1), sum(zprop .== 1)),
           rand(MvNormal(mu2, sig2), sum(zprop .== 2)),
           rand(MvNormal(mu3, sig3), sum(zprop .== 3)))',
        zprop
        );

df1 = @> begin
        DataFrame(dat, [:x, :y, :cluster])
        @> categorical!(:cluster)
end;

# Select hyperparameters
mu0 = hcat(mu1, mu2, mu3)';
nu0 = kappa0 = round.(prop*n);
alpha = prop;
Psi0 = reshape(hcat(sig1, sig2, sig3), (dm,dm,nm));
hyp = (kappa0, mu0, Psi0);

# Initial estimates at NIW, prop, and z for each walker and each temperature
param = Array{Tuple{Array{Dict{String,Array{Float64,N} where N},1},Array{Float64,1},Array{Int64,1}}}(undef, (nw, nt));
for i in 1:nw
        for j in 1:nt
                dictionary = Dict{String,Array{Float64,N} where N}[]
                for m in 1:nm
                        #Sigma = rand(InverseWishart(nu0[m], nu0[m] * Matrix{Float64}(I,dm,dm)))
                        #mu = rand(MvNormal(mu0[m,:], Sigma / nu0[m]))

                        Sigma = rand_constrained_IW(Psi0[:,:,m], nu0[m], [[0,0], [1,1], [-1,1]][m]) #./ nu0[m]
                        mu = rand_constrained_MVN(Sigma/nu0[m], mu0[m,:], [[0,0], [1,1], [-1,1]][m])

                        push!(dictionary, Dict("mu" => mu, "Sigma" => Sigma))
                end
                initprop = rand(Dirichlet(alpha))
                z = rand(1:nm, n)
                param[i,j] = (dictionary, initprop, z)
        end
end;

# inverse temperatures, number of steps, and burn-in steps
betas = collect(range(1, stop=0.001, length=nt));

nstep = 500;
burnin = 10;

# Define log likelihood and log prior functions
ll = ptgibbs.loglike;
lp = ptgibbs.logprior;

# Run chain
chain, _, _, _ =
        ptgibbs.run_mcmc(df1[[:x,:y]], param, hyp, alpha, ll, lp, betas, nstep, burnin);
        #run_constr_mcmc(df1[[:x,:y]], param, hyp, alpha, ll, lp, betas, nstep, burnin, labs);
#chain, _, _ =
        #run_gibbs(df1[[:x,:y]], param, hyp, alpha, ll, lp, nstep, burnin);
        #run_constr_gibbs(df1[[:x,:y]], param, hyp, alpha, ll, lp, nstep, burnin, labs);

# Compute some estimates and get cluster labels
norm_chain = map(x -> x[1], chain)[:,1,:]);
mu_ests = [
        [
        @> begin
                norm_chain
                @>> map(x -> x[1])
                @>> map(x -> get(x, "mu", 0)[1])
                @> mean
        end
        ,
        @> begin
                norm_chain
                @>> map(x -> x[1])
                @>> map(x -> get(x, "mu", 0)[2])
                @> mean
        end
        ],
        [
        @> begin
                norm_chain
                @>> map(x -> x[2])
                @>> map(x -> get(x, "mu", 0)[1])
                @> mean
        end
        ,
        @> begin
                norm_chain
                @>> map(x -> x[2])
                @>> map(x -> get(x, "mu", 0)[2])
                @> mean
        end
        ],
        [
        @> begin
                norm_chain
                @>> map(x -> x[3])
                @>> map(x -> get(x, "mu", 0)[1])
                @> mean
        end
        ,
        @> begin
                norm_chain
                @>> map(x -> x[3])
                @>> map(x -> get(x, "mu", 0)[2])
                @> mean
        end
        ]];

prop_chain = map(x -> x[2], chain)[:,1,:];
prop_est = [
        @> begin
                prop_chain
                @>> map(x -> x[1])
                @> mean
        end
        ,
        @> begin
                prop_chain
                @>> map(x -> x[2])
                @> mean
        end
        ,
        @> begin
                prop_chain
                @>> map(x -> x[3])
                @> mean
        end];

z_chain = map(x -> x[3], chain)[:,1,:]);
z_est = @> begin
                hcat(hcat(z_chain[1,:]...), hcat(z_chain[2,:]...))
                @>> mapslices(sort; dims = 2)
                @>> mapslices(rle; dims = 2)
                @>> map(x -> x[1][argmax(x[2])])
                @> reshape((n,))
end;

# Test for reasonable parameter estimates and classification accuracy
mutest = isapprox.(hcat(mu_ests...)', mu0; atol = .1);
proptest = isapprox.(prop .- prop_est, 0; atol = .05);
correct_class = sum(z_est[collect(1:1:sum(zprop.==1))] .== 1) +
                sum(z_est[collect((sum(zprop.==1) + 1):1:(sum(zprop.==1) + sum(zprop .== 2)))] .== 2) +
                sum(z_est[collect((sum(zprop.==1) + sum(zprop.==2) + 1):1:n)] .== 3);

@testset "reasonable parameter estimates" begin
        [@test x for x in mutest]
        [@test x for x in proptest]
        @test correct_class >= (.9*n)
end;


"""
These are tests for the constrained distributions
"""
nu = 10
Psi01 = reshape([1.2, -.8,0,-.8,1.4,0,0,0,1], (3,3))
h1 = [1, -1, 0]
Psi02 = reshape([1.1, -.7, -.4, 0, -.7, .9, .8, 0, -.4, .8, 1.2, 0,0,0,0,1], (4,4))
h2 = [1, -1, -1, 0]

# Simulate NIW data
sim1 = [rand_constrained_IW(Psi01, nu, h1) for i=1:10000]
sim2 = [rand_constrained_IW(Psi02, nu, h2) for i=1:10000]

meantest1 = isapprox.([
                mean(map(x -> x[1,1], sim1) * nu),
                mean(map(x -> x[2,2], sim1) * nu),
                mean(map(x -> x[3,3], sim1))
                ],
                diag(Psi01); atol = .05)

rhotest1 = isapprox(mean(map(x -> x[1,2], sim1)) * nu, Psi01[1,2]; atol = .05)
meantest2 = isapprox.(
                [
                mean(map(x -> x[1,1], sim2)) * nu,
                mean(map(x -> x[2,2], sim2)) * nu,
                mean(map(x -> x[3,3], sim2)) * nu,
                mean(map(x -> x[4,4], sim2))
                ],
                diag(Psi02); atol = .05)
rhotest2 = isapprox.(
                [
                mean(map(x -> x[1,2], sim2)) * nu,
                mean(map(x -> x[1,3], sim2)) * nu,
                mean(map(x -> x[2,3], sim2)) * nu
                ],
                [
                Psi02[1,2],
                Psi02[1,3],
                Psi02[2,3]
                ]; atol = .05)
otest1 = isequal.(
                round.([
                mean(map(x -> x[1,3], sim1)),
                mean(map(x -> x[2,3], sim1)),
                mean(map(x -> x[3,3], sim1))
                ]; digits = 6),
                [0, 0, 1]
)

otest2 = isequal.(
        round.([
        mean(map(x -> x[1,4], sim2)),
        mean(map(x -> x[2,4], sim2)),
        mean(map(x -> x[3,4], sim2)),
        mean(map(x -> x[4,4], sim2))
        ]; digits = 6),
        [0, 0, 0, 1]
)

@testset "unbiased random matrices" begin
        [@test x for x in meantest1]
        [@test x for x in rhotest1]
        [@test x for x in meantest2]
        [@test x for x in rhotest2]
end

@testset "0 and 1 constraints correctly imposed" begin
        [@test x for x in otest1]
        [@test x for x in otest2]
end
