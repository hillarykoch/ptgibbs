module ptgibbs

include("constrained_distributions.jl")
include("helpers.jl")
include("processors.jl")

import StatsBase: rle, pweights
import RLEVectors: rep
import DataFrames: DataFrame, colwise

using Distributed
using Statistics
using Distributions
using LinearAlgebra
using Lazy
using ProgressMeter

export make_gibbs_update
function make_gibbs_update(dat::DataFrame, param::Array, hyp::Tuple, alpha::Array, labels::Array; tune_df::Int64 = 100)
    NIW = map(x -> x[1], param)
    prop = map(x -> x[2], param)
    z = map(x -> x[3], param)

    kappa0, mu0, Psi0 = hyp

    nw = size(prop, 1)
    nm = size(prop[1],1)
    n, dm = size(dat)

    # Sample new mean estimates from constrained MVN distribution in each cluster
    NIW_out = copy(NIW)
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

                # Extract the current estimate for Sigma for this cluster
                Sigma_hat = get(NIW[i,1][m], "Sigma", 0)

                # Pre-compute some inverses we use multiple times
                iS0 = inv(Sigma_hat .* tune_df ./ nz[m])
                iS = inv(Sigma_hat .* tune_df)

                # Check if we are ok for mu
                likelihood_check = rep(false, times = dm)
                rcMVN_in = inv(iS0 + nz[m] .* iS) * (iS0 * mu0[m,:] + nz[m] .* iS * xbarmap)

                for j=1:dm
                    if sign(rcMVN_in[j]) != sign(labels[m][j]) & sign(labels[m][j]) != 0
                        likelihood_check[j] = true
                    end
                end

                if any(likelihood_check)
                    # Sample from the prior for mu
                    @inbounds mu =
                    rand_constrained_MVN(
                        round.(
                            Matrix(Hermitian(Sigma_hat)); digits = 8),
                        mu0[m,:],
                        labels[m]
                    )
                else
                    @inbounds mu =
                        rand_constrained_MVN(
                            inv(iS0 + nz[m] .* iS),
                            rcMVN_in,
                            labels[m]
                        )
                end
                @inbounds NIW_out[i,1][m]["mu"] = mu
            else
                # Draw from the prior
                @inbounds mu = rand_constrained_MVN(
                    round.(
                        Matrix(Hermitian(Sigma_hat)); digits = 8),
                    mu0[m,:],
                    labels[m]
                )
                @inbounds NIW_out[i,1][m]["mu"] = mu
            end
        end
    end

    # Sample new cluster labels
    zout = copy(z)
    for i in 1:nw
        distns = map(x -> MvNormal(x["mu"], x["Sigma"]), NIW[i,1])
        p = Array{Float64,2}(undef,n,nm)
        for m in 1:nm
            @inbounds p[:,m] = pdf(distns[m], Matrix(dat)') * prop[i][m]
        end
        @inbounds zout[i] = mapslices(x -> sample(1:nm, pweights(x)), p, dims = 2)[:,1]
    end

    # Sample new cluster weights
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
    (zout, NIW_out, propout)
end

export log_likelihood
function log_likelihood(dat::DataFrame,
                        param::Array,
                        nz::Float64,
                        walker_num::Int64,
                        cluster_num::Int64,
                        curr_Sigma::Array,
                        Sigma_star::Array,
                        mu_hat::Array)
    NIW = map(x -> x[1], param)
    prop = map(x -> x[2], param)
    z = map(x -> x[3], param)

    n, dm = size(dat)
    nm = size(prop[walker_num], 1)
    subdat = @inbounds Matrix(dat[z[walker_num] .== cluster_num,:])

    # If there are enough observations in the cluster, compute the log difference of the likelihoods
    if nz >= dm
        mvn_distn_star = MvNormal(mu_hat, Sigma_star)
        mvn_distn_curr = MvNormal(mu_hat, curr_Sigma)
        ll_normal = sum(logpdf(mvn_distn_star, subdat')) - sum(logpdf(mvn_distn_curr, subdat'))
    else
        ll_normal = 0
    end

    # Establish the multivariate distributions that describe the mixture
    distns_curr = map(x -> MvNormal(get(x, "mu", 0), Matrix(Hermitian(get(x, "Sigma", 0)))), NIW[walker_num,1])
    distns_star = deepcopy(distns_curr)
    distns_star[cluster_num] = MvNormal(get(NIW[walker_num,1][cluster_num], "mu", 0), Sigma_star)

    # Compute density of the data in each mixture component
    p_curr = Array{Float64,2}(undef,n,nm)
    for m in 1:nm
        @inbounds p_curr[:,m] = pdf(distns_curr[m], Matrix(dat)') * prop[walker_num][m]
    end

    p_star = deepcopy(p_curr)
    @inbounds p_star[:,cluster_num] = pdf(distns_star[cluster_num], Matrix(dat)') * prop[walker_num][cluster_num]

    # Convert density to probabilities
    @views prob_curr = @> begin
            p_curr
            @>> mapslices(x -> all(x .== 0.0) ? rep(1, times = nm) : x, dims = 2)
            @>> mapslices(x -> x/sum(x), dims = 2)
    end

    @views prob_star = @> begin
            p_star
            @>> mapslices(x -> all(x .== 0.0) ? rep(1, times = nm) : x, dims = 2)
            @>> mapslices(x -> x/sum(x), dims = 2)
    end

    # Plug probabilities into the multinomial likelihood
    mult_distns_curr = mapslices(x -> Multinomial(1,x), prob_curr, dims = 2)
    ll_mult_curr = zeros(n)
    for i in 1:n
        occ = Int64.(zeros(nm))
        @inbounds occ[z[walker_num][i]] = 1
        @inbounds ll_mult_curr[i] = logpdf(mult_distns_curr[i], occ)
    end

    mult_distns_star = mapslices(x -> Multinomial(1,x), prob_star, dims = 2)
    ll_mult_star = zeros(n)
    for i in 1:n
        occ = Int64.(zeros(nm))
        @inbounds occ[z[walker_num][i]] = 1
        @inbounds ll_mult_star[i] = logpdf(mult_distns_star[i], occ)
    end

    ll_mult = sum(ll_mult_star) - sum(ll_mult_curr)

    ll_normal + ll_mult
end

export logprior
function logprior(curr_Sigma::Array, Sigma_star::Array, mu_hat::Array, hyp::Tuple, cluster_num::Int64; tune_df::Int64 = 100)
    """
    The prior is based on two main parts
        1. the density of curr_Sigma and Sigma_star given the prior on Sigma
        2. the density of the current estimate of mu given the MVNormal
            parameterized by either curr_Sigma or Sigma_star
    """
    kappa0, mu0, Psi0 = hyp
    dm = size(curr_Sigma, 1)
    nz = @inbounds max(kappa0[cluster_num], dm)

    norm_distn_curr = @inbounds MvNormal(mu0[cluster_num,:], curr_Sigma .* tune_df ./ nz)#./ nz)
    norm_distn_star = @inbounds MvNormal(mu0[cluster_num,:], Sigma_star .* tune_df ./ nz)#./ nz)
    invwish_distn = @inbounds InverseWishart(nz, Psi0[:,:,cluster_num])

    linvwish_ratio = logpdf(invwish_distn, Sigma_star ./ (nz - dm - 1)) - logpdf(invwish_distn, curr_Sigma ./ (nz - dm - 1))
    lnorm_ratio = logpdf(norm_distn_star, mu_hat) - logpdf(norm_distn_curr, mu_hat)

    linvwish_ratio + lnorm_ratio
end

export get_lhastings
function get_lhastings(curr_Sigma::Array,
                        Sigma_star::Array;
                        tune_df::Int64 = 100)
    """
    The ratio of two Wishart distributions (all indicators cancel out)
    """
    dm = size(Sigma_star, 1)

    term1 = ((2*tune_df-dm-1) / 2) * (logdet(curr_Sigma) - logdet(Sigma_star))
    term2 = (tr(inv(curr_Sigma) * Sigma_star) - tr(inv(Sigma_star) * curr_Sigma)) .* tune_df ./ 2

    term1 + term2
end

export propose_Sigma
function propose_Sigma(curr_Sigma::Array,
                        lab::Array;
                        tune_df::Int64 = 100)
    """
      Draw a covariance from the constrained Wishart distribution
    """
    rand_constrained_Wish(curr_Sigma, tune_df, lab) ./ tune_df
end


export make_mcmc_move
function make_mcmc_move(dat::DataFrame,
                        param::Array,
                        hyp::Tuple,
                        alpha::Array{Float64,1},
                        labels::Array;
                        tune_df::Int64 = 100)
    """
      Make proposals cluster-at-a-time
      For current parameter estimates and current cluster,
        propose a new covariance according to the constrained Wishart
      Return acpt, which tracks acceptances in each cluster
        (1 => accept, 0 => reject)
    """
    NIW = map(x -> x[1], param)
    prop = map(x -> x[2], param)
    z = map(x -> x[3], param)
    kappa0, mu0, Psi0 = hyp

    nm = size(prop[1], 1)
    nw = size(prop, 1)

    # Tracking the acceptance rate
    acpt = zeros(nm)

    # For each walker, for each cluster, sample a covariance from the cWISH
    for i in 1:nw
        rles = @> begin
                @inbounds z[i]
                @>sort()
                @>rle()
            end
        nz = zeros(nm)
        @inbounds nz[rles[1]] = rles[2]

        mu_hats = map(x -> get(x, "mu", 0), NIW[i,1])
        curr_Sigmas = map(x -> get(x, "Sigma", 0), NIW[i,1])
        for m in 1:nm
            # Draw Sigma star from constrained Wishart distribution
            Sigma_star = @inbounds propose_Sigma(curr_Sigmas[m], labels[m], tune_df=100)
            lhaste = @inbounds get_lhastings(curr_Sigmas[m], Sigma_star, tune_df=100)

            # Compute the log prior for this proposal - log prior for the current estimate
            lp = @inbounds logprior(Matrix(Hermitian(curr_Sigmas[m])), Matrix(Hermitian(Sigma_star)), mu_hats[m], hyp, m; tune_df = tune_df)

            # Compute the log likelihood for this proposal - log likelihood for the current estimate
            ll = @inbounds log_likelihood(dat, param, nz[m], i, m, Matrix(Hermitian(curr_Sigmas[m])), Matrix(Hermitian(Sigma_star)), mu_hats[m])

            # If random uniform small enough, update Sigma to Sigma_star
            if log(rand(Uniform(0,1))[1]) < (ll + lp + lhaste)
                NIW[i,1][m]["Sigma"] = Sigma_star
                acpt[m] = 1
            end
        end
    end

    (NIW, acpt)
end

export run_mcmc
function run_mcmc(dat::DataFrame,
                    param::Array,
                    hyp::Tuple,
                    alpha::Array{Float64,1},
                    nstep::Int64,
                    labels::Array{String,1};
                    tune_df::Int64 = 100) #tune_df might become an array
    """
      The function will return `(chain, chainloglike, chainlogprior)` where
        each returned chain value has an extra dimension appended counting steps of the
        chain (so `chain` is of shape `(ndim, nwalkers, nstep)`, for example).
      * dat is an n x nd array of observations
      * alpha is an nm array of hyperparameters for the mixing proportions
      * nstep = the number of steps of the already tuned mcmc
      * nd = number of dimensions
      * nw = number of walkers
      * param contains current parameter estimates for each dimension
            across walkers
    """

    labels = hcat([(parse.(Int64, split(x, "")) .-1) for x in labels])

    nw = size(param, 1)
    n, nd = size(dat)

    chain = Array{
                    Tuple{Array{Dict{String,Array{Float64,N} where N},1},
                    Array{Float64,1},
                    Array{Int64,1}}
                }(undef, nw, nstep + 1)


    for j in 1:nw
        chain[j,1] = deepcopy(param[j,1])
    end
    acpt_tracker = zeros(nm)

    @showprogress 1 "Running the MCMC..."  for i in 1:nstep
        # Proposes IW cluster at a time, accepts/rejects/returns new IW
        NIW, acpt = make_mcmc_move(dat, param, hyp, alpha, labels; tune_df=tune_df)
        acpt_tracker = acpt_tracker .+ acpt

        # Makes Gibbs updates of means, mixing weights, and class labels
        z, NIW, prop = make_gibbs_update(dat, param, hyp, alpha, labels; tune_df = tune_df)

        for j in 1:nw
            chain_link = deepcopy(param)
            map!(x -> x, chain_link[j,1][1], NIW[j,1])
            map!(x -> x, chain_link[j,1][2], prop[j,1])
            map!(x -> x, chain_link[j,1][3], z[j,1])

            @inbounds chain[j,i+1] = deepcopy(chain_link[j,1])
        end
    end
    (chain, acpt_tracker)
end

end # end the module
