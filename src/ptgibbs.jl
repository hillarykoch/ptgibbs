module ptgibbs

include("constrained_distributions.jl")
include("helpers.jl")
include("processors.jl")
include("plotters.jl")
include("normalizing.jl")

import StatsBase: rle, pweights
import RLEVectors: rep
import DataFrames: DataFrame, colwise

using Statistics
using Distributions
using LinearAlgebra
using Lazy
using ProgressMeter

export make_gibbs_update
function make_gibbs_update(dat::DataFrame, param::Array, hyp::Tuple, alpha::Array{Float64,1}, labels::Array{Int64,1})
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
                @inbounds xbarmap = map(x -> x[m], xbar)

                # Extract the current estimate for Sigma for this cluster
                @inbounds Sigma_hat = get(NIW[i,1][m], "Sigma", 0)

                # Pre-compute some inverses we use multiple times
                iS0 = inv(Sigma_hat ./ max(kappa0[m], dm))
                iS = inv(Sigma_hat)

                # Check if we are ok for mu
                likelihood_check = rep(false, times = dm)
                @inbounds rcMVN_in = inv(iS0 + nz[m] .* iS) * (iS0 * mu0[m,:] + nz[m] .* iS * xbarmap)

                for j=1:dm
                    if @inbounds sign(rcMVN_in[j]) != sign(labels[m][j]) & sign(labels[m][j]) != 0
                        @inbounds likelihood_check[j] = true
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
                Sigma_hat = get(NIW[i,1][m], "Sigma", 0)
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
        distns = map(x -> MvNormal(x["mu"], Matrix(Hermitian(x["Sigma"]))), NIW_out[i,1])
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
            zout[i]
            @>sort()
            @>rle()
        end
        nz = zeros(nm)
        @inbounds nz[rles[1]] = rles[2]
        @inbounds propout[i] = rand(Dirichlet(alpha .+ nz))
    end

    (zout, NIW_out, propout)
end

export log_likelihood
function log_likelihood(dat::DataFrame,
                        param::Array,
                        nz::Float64,
                        walker_num::Int64,
                        cluster_num::Int64,
                        curr_Sigma::Array{Float64,2},
                        Sigma_star::Array{Float64,2},
                        mu_hat::Array{Float64,1})
    NIW = map(x -> x[1], param)
    prop = map(x -> x[2], param)
    z = map(x -> x[3], param)

    n, dm = size(dat)
    nm = @inbounds size(prop[walker_num], 1)
    subdat = @inbounds Matrix(dat[z[walker_num] .== cluster_num,:])

    # If there are enough observations in the cluster, compute the log difference of the likelihoods
    if nz >= dm
        mvn_distn_star = MvNormal(mu_hat, Sigma_star)
        mvn_distn_curr = MvNormal(mu_hat, curr_Sigma)
        ll_normal = sum(logpdf(mvn_distn_star, subdat')) - sum(logpdf(mvn_distn_curr, subdat'))
    else
        ll_normal = 0.0
    end

    # Establish the multivariate distributions that describe the mixture
    @inbounds distns_curr = map(x -> MvNormal(get(x, "mu", 0), Matrix(Hermitian(get(x, "Sigma", 0)))), NIW[walker_num,1])
    distns_star = deepcopy(distns_curr)
    @inbounds distns_star[cluster_num] = MvNormal(get(NIW[walker_num,1][cluster_num], "mu", 0), Sigma_star)

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
        occ = zeros(nm)#Int64.(zeros(nm))
        @inbounds occ[z[walker_num][i]] = 1
        @inbounds ll_mult_star[i] = logpdf(mult_distns_star[i], occ)
    end

    ll_mult = sum(ll_mult_star) - sum(ll_mult_curr)

    ll_normal + ll_mult
end

export logprior
function logprior(curr_Sigma::Array{Float64,2}, Sigma_star::Array{Float64,2}, mu_hat::Array{Float64,1}, hyp::Tuple, cluster_num::Int64)
    """
    The prior is based on two main parts
        1. the density of curr_Sigma and Sigma_star given the prior on Sigma
        2. the density of the current estimate of mu given the MVNormal
            parameterized by either curr_Sigma or Sigma_star
    """
    kappa0, mu0, Psi0 = hyp
    dm = size(curr_Sigma, 1)
    mnz = @inbounds max(kappa0[cluster_num], dm + 2)

    #norm_distn_curr = @inbounds MvNormal(mu0[cluster_num,:], curr_Sigma .* old_tune_df ./ mnz)
    #norm_distn_star = @inbounds MvNormal(mu0[cluster_num,:], Sigma_star .* tune_df ./ mnz)
    norm_distn_curr = @inbounds MvNormal(mu0[cluster_num,:], curr_Sigma)
    norm_distn_star = @inbounds MvNormal(mu0[cluster_num,:], Sigma_star)
    invwish_distn = @inbounds InverseWishart(mnz, Psi0[:,:,cluster_num])

    linvwish_ratio = logpdf(invwish_distn, Sigma_star ./ (mnz - dm - 1)) - logpdf(invwish_distn, curr_Sigma ./ (mnz - dm - 1))
    lnorm_ratio = logpdf(norm_distn_star, mu_hat) - logpdf(norm_distn_curr, mu_hat)

    linvwish_ratio + lnorm_ratio
end

export get_lhastings
function get_lhastings(curr_Sigma::Array{Float64,2}, Sigma_star::Array{Float64,2}, tune_df::Float64)
    """
    The ratio of two Wishart distributions (all indicators cancel out)
    """
    dm = size(Sigma_star, 1)

    term1 = ((2*tune_df-dm-1) / 2) * (logdet(curr_Sigma) - logdet(Sigma_star))
    term2 = (tr(inv(curr_Sigma) * Sigma_star) - tr(inv(Sigma_star) * curr_Sigma)) .* tune_df ./ 2

    term1 + term2
end

export propose_Sigma
function propose_Sigma(curr_Sigma::Array{Float64,2}, lab::Array{Int64,1}, tune_df::Float64)
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
                        labels::Array{Int64,1},
                        tune_df::Array{Float64,1})
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
            Sigma_star = @inbounds propose_Sigma(curr_Sigmas[m], labels[m], tune_df[m])

            # Compute the normalizing constants first
            #norm_const_cwish = approx_cwish_norm_const(tune_df[m], curr_Sigmas[m], Sigma_star)
            #norm_const_tmvn = approx_tmvn_norm_const(mu_hats[m], curr_Sigmas[m], Sigma_star)

            #if all([norm_const_cwish == Inf, norm_const_tmvn == Inf])
                #lhaste = @inbounds get_lhastings(curr_Sigmas[m], Sigma_star, tune_df[m])
                #lp = @inbounds logprior(Matrix(Hermitian(curr_Sigmas[m])), Matrix(Hermitian(Sigma_star)), mu_hats[m], hyp, m)
                #ll = @inbounds log_likelihood(dat, param, max(nz[m], kappa0[m]), i, m, Matrix(Hermitian(curr_Sigmas[m])), Matrix(Hermitian(Sigma_star)), mu_hats[m])
                #if log(rand(Uniform(0,1))[1]) < (ll + lp + lhaste)
                    #@inbounds NIW[i,1][m]["Sigma"] = Sigma_star
                    #@inbounds acpt[m] = 1
                #end
            #elseif any([norm_const_cwish == Inf, norm_const_tmvn == Inf]) & all([norm_const_cwish != 0, norm_const_tmvn != 0])
                ## just accept because the ratio will be
                #@inbounds NIW[i,1][m]["Sigma"] = Sigma_star
                #@inbounds acpt[m] = 1
            #elseif any([norm_const_cwish == -Inf, norm_const_tmvn == -Inf]) & all([norm_const_cwish != 0, norm_const_tmvn != 0])
                ## do nothing, we dont update
            #elseif any([abs(norm_const_cwish) == Inf, abs(norm_const_tmvn) == Inf]) & any([norm_const_cwish == 0, norm_const_tmvn == 0])
                #lhaste = @inbounds get_lhastings(curr_Sigmas[m], Sigma_star, tune_df[m])
                #lp = @inbounds logprior(Matrix(Hermitian(curr_Sigmas[m])), Matrix(Hermitian(Sigma_star)), mu_hats[m], hyp, m)
                #ll = @inbounds log_likelihood(dat, param, max(nz[m], kappa0[m]), i, m, Matrix(Hermitian(curr_Sigmas[m])), Matrix(Hermitian(Sigma_star)), mu_hats[m])
                #if log(rand(Uniform(0,1))[1]) < (ll + lp + lhaste)
                    #@inbounds NIW[i,1][m]["Sigma"] = Sigma_star
                    #@inbounds acpt[m] = 1
                #end
            #else
                #logratio = log(norm_const_cwish) + log(norm_const_tmvn)
                lhaste = @inbounds get_lhastings(curr_Sigmas[m], Sigma_star, tune_df[m])

                # Compute the log prior for this proposal - log prior for the current estimate
                lp = @inbounds logprior(Matrix(Hermitian(curr_Sigmas[m])), Matrix(Hermitian(Sigma_star)), mu_hats[m], hyp, m)

                # Compute the log likelihood for this proposal - log likelihood for the current estimate
                ll = @inbounds log_likelihood(dat, param, max(nz[m], kappa0[m]), i, m, Matrix(Hermitian(curr_Sigmas[m])), Matrix(Hermitian(Sigma_star)), mu_hats[m])

                # If random uniform small enough, update Sigma to Sigma_star
                if log(rand(Uniform(0,1))[1]) < (ll + lp + lhaste)# + logratio)
                    @inbounds NIW[i,1][m]["Sigma"] = Sigma_star
                    @inbounds acpt[m] = 1
                end
            #end
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
                    labels::Array{String,1},
                    tune_df::Array{Float64,1};
                    opt_rate::Float64 = 0.3)
    """
      The function will return `(chain, acpt_chain, tune_df_chain)` where
        each returned chain value has an extra dimension appended counting steps of the
        chain (so `chain` is of shape `(ndim, nwalkers, nstep)`, for example).
      * acpt_chain tracks the acceptance rate for each cluster across the chain
      * tune_df_chain tracks the tuning degrees of freedom in the wishart proposal across the chain
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
    nm = size(param[1,1][2],1)

    chain = Array{
                    Tuple{Array{Dict{String,Array{Float64,N} where N},1},
                    Array{Float64,1},
                    Array{Int64,1}}
                }(undef, nw, nstep + 1)

    for j in 1:nw
        chain[j,1] = deepcopy(param[j,1])
    end
    acpt_tracker = zeros(nm)

    # For tracking the acceptance rate, per cluster, for the adaptive tuning variance
    win_len = min(nstep, 50)
    acpt_win = zeros(nm, win_len)
    acpt_chain = zeros(nm, nstep)

    old_tune_df = deepcopy(tune_df)
    tune_df_chain = zeros(nm, nstep)

    @showprogress 1 "Running the MCMC..."  for i in 1:nstep
        # Proposes IW cluster at a time, accepts/rejects/returns new IW
        NIW, acpt = make_mcmc_move(dat, chain[:,i], hyp, alpha, labels, tune_df)
        [ acpt_tracker[mm] += acpt[mm] for mm in 1:nm ]
        #acpt_tracker = acpt_tracker .+ acpt
        @inbounds acpt_win[:, (i-1) % win_len + 1] = acpt

        # Makes Gibbs updates of means, mixing weights, and class labels
        newz, newNIW, newprop = make_gibbs_update(dat, chain[:,i], hyp, alpha, labels)

        if i > 50
            # Update tuning parameter per cluster
            gamma1 = 10 / (i ^ 0.8)
            old_tune_df = copy(tune_df)
            @inbounds [ tune_df[m] = update_tune_df(tune_df[m], mean(acpt_win[m,:]), opt_rate, gamma1) for m in 1:1:nm ]
        end

        for j in 1:nw
            chain_link = deepcopy(param)

            @inbounds @views map!(x -> x, chain_link[j,1][1], newNIW[j,1])
            @inbounds @views map!(x -> x, chain_link[j,1][2], newprop[j,1])
            @inbounds @views map!(x -> x, chain_link[j,1][3], newz[j,1])

            @inbounds chain[j,i+1] = deepcopy(chain_link[j,1])
            @inbounds acpt_chain[:,i] = acpt_tracker ./ i
            @inbounds tune_df_chain[:,i] = tune_df
        end
    end
    (chain, acpt_chain, tune_df_chain)
end

end # end the module
