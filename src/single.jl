import StatsBase: rle, pweights
import RLEVectors: rep
import DataFrames: DataFrame, colwise

using Distributed
using Statistics
using Distributions
using LinearAlgebra
using Lazy
using ProgressMeter

export make_beta1_gibbs_update
function make_beta1_gibbs_update(dat, hyp, z, prop, alpha)
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
                @inbounds mu = rand(
                        MvNormal(
                            (kappa0[m] * mu0[m,:] + nz[m] * xbarmap) / (kappa0[m] + nz[m]),
                            Sigma / (kappa0[m] + nz[m])
                        )
                )
                @inbounds NIW[i,m] = Dict("mu" => mu, "Sigma" => Sigma)
            else
                # Draw from the prior
                @inbounds Sigma = rand(
                            InverseWishart(kappa0[m],
                                              Psi0[:,:,m] * kappa0[m]
                        )
                )
                @inbounds mu = rand(
                        MvNormal(
                            mu0[m,:],
                            Sigma / kappa0[m]
                        )
                )
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

export lnlikes_lnpriors_beta1
function lnlikes_lnpriors_beta1(dat, param, alpha, ll, lp)
    nw = size(param, 1)
    lls = zeros(nw, 1)
    lps = zeros(nw, 1)
    for i in 1:nw
        @inbounds lls[i,1], lps[i,1] = lnlike_lnprior(dat, param[i], alpha, ll, lp)
    end
    (lls, lps)
end

export make_beta1_mcmc_move
function make_beta1_mcmc_move(dat, param, hyp, alpha, ll, lp)
    nw = size(param, 1)
    prop = map(x -> x[2], param)
    z = map(x -> x[3], param)

    z, NIW, prop = make_beta1_gibbs_update(dat, hyp, z, prop, alpha) # added prop here (and alpha passed to make_gibbs_update)

    # Update param to reflect gibbs updates
    for i in 1:nw
        map!(x -> x, param[i,1][1], NIW[i,1,:])
        map!(x -> x, param[i,1][2], prop[i])
        map!(x -> x, param[i,1][3], z[i])
    end

    llps, lpps = lnlikes_lnpriors_beta1(dat, param, alpha, ll, lp)

    (param, llps, lpps)
end

export run_gibbs
function run_gibbs(dat::DataFrame,
                    param::Array,
                    hyp::Tuple,
                    alpha::Array{Float64,1},
                    loglike,
                    logprior,
                    nstep::Int64,
                    burnin::Int64)
    ll = loglike
    lp = logprior

    nw = size(param, 1)
    n, nd = size(dat)

    @showprogress 1 "Computing for burn-in..." for i in 1:burnin
        param, lnlikes, lnpriors = make_beta1_mcmc_move(dat, param, hyp, alpha, loglike, logprior)
    end

    chain = Array{
                    Tuple{Array{Dict{String,Array{Float64,N} where N},1},
                    Array{Float64,1},
                    Array{Int64,1}}
            }(undef, nw, nstep)
    chainlnlike = zeros(nw, nstep)
    chainlnprior = zeros(nw, nstep)

    @showprogress 1 "Computing for main Markov chain..."  for i in 1:nstep
        param, lnlikes, lnpriors = make_beta1_mcmc_move(dat, param, hyp, alpha, loglike, logprior)

        @inbounds chain[:,i] = deepcopy(param)
        @inbounds chainlnlike[:,i] = lnlikes
        @inbounds chainlnprior[:,i] = lnpriors
    end
    chain, chainlnlike, chainlnprior
end

export get_gibbs_mu_chain
function get_gibbs_mu_chain(chain, walker_num, cluster_num)
    """
    For a given walker and cluster number, return
        the corresponding chain of mu estimates in each dimension
    This is used to pass results back to R via JuliaCall
    """
    nw, nstep = size(chain)
    nm = size(chain[1,1][1], 1)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"
    @assert nm >= cluster_num "cluster_num must be less than or equal to the number of clusters"

    # The first line extracts NIW part of the chain
    # Then, get mu for given walker and cluster number
    # Shoud be a dm x nstep chain (dm is dimension of data)
    mu_chain = @as mu_chain map(x -> x[1], chain)[walker_num,:] begin
        @>> @views map(x -> x[cluster_num], mu_chain)
        @>> map(x -> get(x, "mu", 0), mu_chain)
        @> hcat(mu_chain...)
    end

    return mu_chain
end

export get_gibbs_Sigma_chain
function get_gibbs_Sigma_chain(chain, walker_num, cluster_num)
    """
    For a given walker and cluster number, return
        the corresponding chain of covariance estimates in each dimension
    This is used to pass results back to R via JuliaCall
    """
    nw, nstep = size(chain)
    nm =  size(chain[1,1][1], 1)
    dm = size(get(chain[1,1,1][1][1], "Sigma", 0), 1)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"
    @assert nm >= cluster_num "cluster_num must be less than or equal to the number of clusters"

    # The first line extracts NIW part of the chain
    # Then, get mu for given walker and cluster number
    # Shoud be a dm x dm x nstep array of Sigma estimates (dm is dimension of data)
    Sigma_chain = @as Sigma_chain map(x -> x[1], chain)[walker_num, :] begin
        @>> @views map(x -> x[cluster_num], Sigma_chain)
        @>> map(x -> get(x, "Sigma", 0), Sigma_chain)
        @> hcat(Sigma_chain...)
        @> reshape(Sigma_chain, (dm, dm, nstep))
    end

    return Sigma_chain
end

export get_gibbs_prop_chain
function get_gibbs_prop_chain(chain, walker_num)
    """
    For a given walker, return the corresponding
        chain of mixing weight estimates for each cluster
    This is used to pass results back to R via JuliaCall
    """
    nw, nstep = size(chain)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"

    # Shoud be a nm x nstep chain
    prop_chain = hcat(map(x -> x[2], chain)[walker_num,:]...)

    return prop_chain
end

export get_gibbs_z_chain
function get_gibbs_z_chain(chain, walker_num)
    """
    For a given walker, return the
        corresponding chain of z estimates for each cluster and each observation
    This is used to pass results back to R via JuliaCall
    """
    nw, nstep = size(chain)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"

    # Returns an n x nstep chain of cluster labels for each observation across the whole chain
    z_chain = @as z_chain map(x -> x[3], chain) begin
                @> z_chain[walker_num,:]
                @> hcat(z_chain...)
            end

    return z_chain
end
