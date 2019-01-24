import StatsBase: rle, pweights
import RLEVectors: rep
import DataFrames: DataFrame, colwise

using Distributed
using Statistics
using Distributions
using LinearAlgebra
using Lazy
using ProgressMeter


export make_constr_mcmc_move
function make_constr_mcmc_move(dat, param, hyp, alpha, ll, lp, betas, labels)
    nw, nt = size(param)
    NIW = map(x -> x[1], param)
    prop = map(x -> x[2], param)
    z = map(x -> x[3], param)

    """
    NB: the particular constraints imposed on mu, Sigma allow these constrained
        updates to remain Gibbs, thanks to the uncorrelatedness in the covariance matrix
        of constrained values
    Consider the conditional distribution of a MVN proposal for mu
        (e.g., here: https://www.wikiwand.com/en/Multivariate_normal_distribution#/Conditional_distributions)
        and the partitioning of an Inverse Wishart proposal for Sigma
            (e.g., here: https://projecteuclid.org/euclid.lnms/1196285114)
    """
    z, NIW, prop = make_constr_gibbs_update(dat, hyp, z, prop, alpha, labels)

    for i in 1:nw
        for j in 1:nt
            map!(x -> x, param[i,j][1], NIW[i,j,:])
            map!(x -> x, param[i,j][2], prop[i,j])
            map!(x -> x, param[i,j][3], z[i,j])
        end
    end

    llps, lpps = lnlikes_lnpriors(dat, param, alpha, ll, lp)

    (param, llps, lpps)
end

export run_constr_mcmc
function run_constr_mcmc(dat::DataFrame,
                    param::Array,
                    hyp::Tuple,
                    alpha::Array{Float64,1},
                    loglike,
                    logprior,
                    betas::Array{Float64,1},
                    nstep::Int64,
                    burnin::Int64,
                    labels::Array{String,1})
    # Reformat labels from R to be good for this application
    labels = hcat([split(x, "") for x in labels])

    ll = loglike
    lp = logprior

    nw, nt = size(param)
    n, nd = size(dat)

    @assert size(betas,1)==nt "size(betas,1) must be equal to the number of temperatures"

    @showprogress 1 "Computing for burn-in..." for i in 1:burnin
        param, lnlikes, lnpriors = make_constr_mcmc_move(dat, param, hyp, alpha, loglike, logprior, betas, labels)

        param, lnlikes, lnpriors, tswaps = make_tswap(param, lnlikes, lnpriors, betas)
        swapfraction = tswaps / nw
        tc = 10.0 + i/10.0

        betas = tevolve(swapfraction, betas, tc)
    end

    chain = Array{
                    Tuple{Array{Dict{String,Array{Float64,N} where N},1},
                    Array{Float64,1},
                    Array{Int64,1}}
            }(undef, nw, nt, nstep)
    chainlnlike = zeros(nw, nt, nstep)
    chainlnprior = zeros(nw, nt, nstep)

    @showprogress 1 "Computing for main Markov chain..."  for i in 1:nstep
        param, lnlikes, lnpriors = make_constr_mcmc_move(dat, param, hyp, alpha, loglike, logprior, betas, labels)
        param, lnlikes, lnpriors, _ = make_tswap(param, lnlikes, lnpriors, betas)

        @inbounds chain[:,:,i] = deepcopy(param)
        @inbounds chainlnlike[:,:,i] = lnlikes
        @inbounds chainlnprior[:,:,i] = lnpriors
    end

    chain, chainlnlike, chainlnprior, betas
end



export make_constr_beta1_gibbs_update
function make_constr_beta1_gibbs_update(dat, hyp, z, prop, alpha, labels)
    nw = size(prop, 1)
    nm = size(prop[1])[1]
    kappa0, mu0, Psi0 = hyp
    n, dm = size(dat)

    NIW = Array{Dict{String,Array{Float64,N} where N}}(undef,nw,nm)
    for i in 1:nw
        rles = @> begin
                @inbounds z[i]
                @>sort()
                @>rle()
            end
        nz = zeros(nm)
        @inbounds nz[rles[1]] = rles[2]

        # xbar is an array of d dictionaries
        xbar = @views colwise(x -> tapply_mean(z[i], x), dat)
        for m in 1:nm
            if nz[m] >= dm
                # Compute this only once because it gets reused a lot here
                xbarmap = map(x -> x[m], xbar)
                # Draw from the posterior (I don't have the additional ifelse that is in my R code here)
                @inbounds Sigma = rand(
                            InverseWishart(kappa0[m] + nz[m],
                                              round.(Psi0[:,:,m] * kappa0[m] +
                                                 (Matrix(dat[z[i] .== m,:]) .- xbarmap')' *
                                                      (Matrix(dat[z[i] .== m,:]) .- xbarmap') +
                                                      (kappa0[m] * nz[m]) / (kappa0[m] + nz[m]) *
                                                      (xbarmap - mu0[m,:]) *  (xbarmap - mu0[m,:])'; digits=6)
                        )
                )

                for d1 in 1:dm
                    if labels[m][d1] == "1"
                        @inbounds Sigma[d1, 1:end .!= d1] = zeros(dm-1)
                        @inbounds Sigma[1:end .!= d1, d1] = zeros(dm-1)
                        @inbounds Sigma[d1, d1] = 1
                    end
                end

                @inbounds mu = rand(
                        MvNormal(
                            (kappa0[m] * mu0[m,:] + nz[m] * xbarmap) / (kappa0[m] + nz[m]),
                            Sigma / (kappa0[m] + nz[m])
                        )
                )
                @inbounds zero_loc = labels[m] .== "1"
                @inbounds mu[zero_loc] = zeros(sum(zero_loc))

                @inbounds NIW[i,m] = Dict("mu" => mu, "Sigma" => Sigma)
            else
                # Draw from the prior
                @inbounds Sigma = rand(
                            InverseWishart(kappa0[m],
                                              Psi0[:,:,m] * kappa0[m]
                        )
                )

                for d1 in 1:dm
                    if labels[m][d1] == "1"
                        @inbounds Sigma[d1, 1:end .!= d1] = zeros(dm-1)
                        @inbounds Sigma[1:end .!= d1, d1] = zeros(dm-1)
                        @inbounds Sigma[d1, d1] = 1
                    end
                end

                @inbounds mu = rand(
                        MvNormal(
                            mu0[m,:],
                            Sigma / kappa0[m]
                        )
                )
                @inbounds zero_loc = labels[m] .== "1"
                @inbounds mu[zero_loc] = zeros(sum(zero_loc))

                @inbounds NIW[i,m] = Dict("mu" => mu, "Sigma" => Sigma)
            end
        end
    end

    zout = copy(z)
    for i in 1:nw
        distns = map(x -> MvNormal(x["mu"], x["Sigma"]), NIW[i,:])
        p = Array{Float64,2}(undef,n,nm)
        for m in 1:nm
            p[:,m] = pdf(distns[m], Matrix(dat)') * prop[i][m]
        end
        @inbounds zout[i] = mapslices(x -> sample(1:nm, pweights(x)), p, dims = 2)[:,1]
    end

    propout = copy(prop)
    for i in 1:nw
        rles = @> begin
            z[i]
            @>sort()
            @>rle()
        end
        nz = zeros(nm)
        nz[rles[1]] = rles[2]
        propout[i] = rand(Dirichlet(alpha + nz))
    end
    (zout, NIW, propout)
end


export make_constr_beta1_mcmc_move
function make_constr_beta1_mcmc_move(dat, param, hyp, alpha, ll, lp, labels)
    nw = size(param, 1)
    prop = map(x -> x[2], param)
    z = map(x -> x[3], param)

    z, NIW, prop = make_constr_beta1_gibbs_update(dat, hyp, z, prop, alpha, labels)

    # Update param to reflect gibbs updates
    for i in 1:nw
        map!(x -> x, param[i,1][1], NIW[i,1,:])
        map!(x -> x, param[i,1][2], prop[i])
        map!(x -> x, param[i,1][3], z[i])
    end

    llps, lpps = lnlikes_lnpriors_beta1(dat, param, alpha, ll, lp)

    (param, llps, lpps)
end

export run_constr_gibbs
function run_constr_gibbs(dat::DataFrame,
                    param::Array,
                    hyp::Tuple,
                    alpha::Array{Float64,1},
                    loglike,
                    logprior,
                    nstep::Int64,
                    burnin::Int64,
                    labels::Array{String,1})

    # Reformat labels from R to be good for this application
    labels = hcat([split(x, "") for x in labels])

    ll = loglike
    lp = logprior

    nw = size(param, 1)
    n, nd = size(dat)

    @showprogress 1 "Computing for burn-in..." for i in 1:burnin
        param, lnlikes, lnpriors = make_constr_beta1_mcmc_move(dat, param, hyp, alpha, loglike, logprior, labels)
    end

    chain = Array{
                    Tuple{Array{Dict{String,Array{Float64,N} where N},1},
                    Array{Float64,1},
                    Array{Int64,1}}
            }(undef, nw, nstep)
    chainlnlike = zeros(nw, nstep)
    chainlnprior = zeros(nw, nstep)

    @showprogress 1 "Computing for main Markov chain..."  for i in 1:nstep
        param, lnlikes, lnpriors = make_constr_beta1_mcmc_move(dat, param, hyp, alpha, loglike, logprior, labels)

        @inbounds chain[:,i] = deepcopy(param)
        @inbounds chainlnlike[:,i] = lnlikes
        @inbounds chainlnprior[:,i] = lnpriors
    end

    chain, chainlnlike, chainlnprior
end
