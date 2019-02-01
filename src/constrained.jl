
import StatsBase: rle, pweights
import RLEVectors: rep
import DataFrames: DataFrame, colwise

using Distributed
using Statistics
using Distributions
using LinearAlgebra
using Lazy
using ProgressMeter

"""
In R, can obtain an MxD matrix corresponding to the M classes described
  by a length-D ternary vector. Call this *red_class*
In R, can run apply(red_class + 1, 1, function(X) paste0(X, collapse = "")) to
  get M collapsed ternary vectors. These are made up of 0, 1, and 2, where
  0 corresponds to negative association
  1 corresponds to no association
  2 corresponds to positive association
These can be stepped over in the Gibbs sampler to apply constraints appropriately
Here, the only constraint is that a 1 label implies that mu = 0, sigma = 1,
    and all appropriate rho are 0. This constraint is easily accounted for in the acceptance ratio
    as the Gibbs updates actually become asymmetric proposals
"""

export make_constr_gibbs_update
function make_constr_gibbs_update(dat, hyp, z, prop, alpha, labels)
    """
    * dat is an n x dm data matrix (maybe need it to be data frame)
    * hyp is a tuple with
        hyp[1] = vector of kappa0, and sum(kappa0) = n
        hyp[2] = dm x nm array of mu0
        hyp[3] = dm x dm x nm array of Psi0
    * z is an int array of size nw x nt, where each entry is of length n
        classifying each observation for each walker and temperature
    * prop is an nw x nt array, where each entry is of length nm
    * alpha is an nm array of prior weights
    """
    nw, nt = size(prop)
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

                rcIW_in =
                Psi0[:,:,m] * max(kappa0[m], dm) +
                    (Matrix(dat[z[i] .== m,:]) .- xbarmap')' *
                    (Matrix(dat[z[i] .== m,:]) .- xbarmap') +
                    (max(kappa0[m], dm) * nz[m]) / (max(kappa0[m], dm) + nz[m]) *
                    (xbarmap - mu0[m,:]) *  (xbarmap - mu0[m,:])'

                # Check if we are ok with Sigma
                likelihood_check = reshape(rep(false, times = dm^2), (dm,dm))
                for j = 1:(dm-1)
                    for k = (j+1):dm
                        if sign(rcIW_in[j,k]) != sign((labels[m][j] - 1) * (labels[m][k] - 1)) &
                                sign((labels[m][j] - 1) * (labels[m][k] - 1)) != 0
                            likelihood_check[j,k] = true
                        end
                    end
                end

                if any(likelihood_check)
                    # Sample from the prior for Sigma
                    @inbounds Sigma = rand_constrained_IW(
                        round.(Psi0[:,:,m] * max(kappa0[m], dm); digits=8),
                        max(kappa0[m], dm),
                        labels[m] .- 1
                    )
                else
                    @inbounds Sigma = rand_constrained_IW(
                        rcIW_in,
                        max(kappa0[m], dm) + nz[m],
                        labels[m] .- 1)
                end

                # Check if we are ok for mu
                likelihood_check2 = rep(false, times = dm)
                rcMVN_in = (max(kappa0[m], dm) * mu0[m,:] + nz[m] * xbarmap) ./ (max(kappa0[m], dm) + nz[m])
                for j=1:dm
                    if sign(rcMVN_in[j]) != sign(labels[m][j] - 1) & sign(labels[m][j] - 1) != 0
                        likelihood_check2[j] = true
                    end
                end

                if any(likelihood_check2)
                    # Sample from the prior for mu
                    @inbounds mu = rand_constrained_MVN(
                        round.(Sigma ./ max(kappa0[m], dm); digits = 8),
                        mu0[m,:],
                        labels[m] .- 1
                    )
                else
                    @inbounds mu = rand_constrained_MVN(
                        Sigma ./ (max(kappa0[m], dm) + nz[m]),
                        rcMVN_in,
                        labels[m] .- 1
                        )
                end

                @inbounds NIW[i,j,m] = Dict("mu" => mu, "Sigma" => Sigma)
            else
                # Draw from the prior
                @inbounds Sigma = rand_constrained_IW(
                    round.(Psi0[:,:,m] * max(kappa0[m], dm); digits=8),
                    max(kappa0[m], dm),
                    labels[m] .- 1
                )

                @inbounds mu = rand_constrained_MVN(
                    round.(Sigma ./ max(kappa0[m], dm); digits = 8),
                    mu0[m,:],
                    labels[m] .- 1
                )

                @inbounds NIW[i,j,m] = Dict("mu" => mu, "Sigma" => Sigma)
            end
        end
    end

    """
    Draw new cluster labels
    Store in an nw x nt array, where each entry is of length n
    """
    zout = copy(z)
    for i in 1:nw
        for j in 1:nt
            distns = map(x -> MvNormal(x["mu"], x["Sigma"]), NIW[i,j,:])
            p = Array{Float64,2}(undef,n,nm)
            for m in 1:nm
                p[:,m] = pdf(distns[m], Matrix(dat)') * prop[i,j][m]
            end

            @inbounds zout[i,j] = mapslices(x -> sample(1:nm, pweights(x)), p, dims = 2)[:,1]
        end
    end

    """
    Draw new mixing weights
    Store in an nw x nt array, where each entry is of length nm
    """
    propout = copy(prop)
    for i in 1:nw
        for j in 1:nt
            """
            Count the number of observations in each class
                for the current walker, current temperature
            """
            rles = @> begin
                z[i,j]
                @>sort()
                @>rle()
            end
            nz = zeros(nm)
            nz[rles[1]] = rles[2]
            propout[i,j] = rand(Dirichlet(alpha + nz))
        end
    end
    (zout, NIW, propout)
end

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
    labels = hcat([parse.(Int64, split(x, "")) for x in labels])

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
