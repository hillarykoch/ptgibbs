module ptgibbs

include("constrained.jl")

import StatsBase: rle
import RLEVectors: rep
import DataFrames: colwise

using Distributed
using Statistics
using Distributions
using LinearAlgebra
using Lazy
using ProgressMeter
#using DataFrames

export tapply_mean
function tapply_mean(subs, val, sz=(maximum(subs),))
    A = zeros(eltype(val), sz...)
    counter = zeros(Int64, sz...)
    for i = 1:length(val)
        @inbounds A[subs[i]] += val[i]
        @inbounds counter[subs[i]] += 1
    end
    A./counter
end

export make_gibbs_update
function make_gibbs_update(dat, hyp, z, prop, alpha)
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

    NIW = Array{Dict{String,Array{Float64,N} where N}}(undef,nw,nt,nm)
    for i in 1:nw
        for j in 1:nt
            """
            Count the number of observations in each class
                for the current walker, current temperature
            """
            rles = @> begin
                    @inbounds z[i,j]
                    @>sort()
                    @>StatsBase.rle()
                end
            nz = zeros(nm)
            @inbounds nz[rles[1]] = rles[2]

            """
            Compute the d-dimensional sample mean for each class
                for the current walker, current temperature
            """
            # xbar is an array of d dictionaries
            #xbar = @views DataFrames.colwise(x -> tapply_mean(z[i,j], x), dat)
            xbar = @views colwise(x -> tapply_mean(z[i,j], x), dat)

            """
            Draw NIW random variables
            If there are enough (dm) observations in the class, sample from the posterior
                Otherwise, just draw from the prior
            Store the NIW in an array of dictionaries
            """
            for m in 1:nm
                if nz[m] >= dm
                    # Compute this only once because it gets reused a lot here
                    xbarmap = map(x -> x[m], xbar)

                    # Draw from the posterior (I don't have the additional ifelse that is in my R code here)
                    Sigma = rand(
                                InverseWishart(kappa0[m] + nz[m],
                                               round.(Psi0[:,:,m] * kappa0[m] +
                                                  (Matrix(dat[z[i,j] .== m,:]) .- xbarmap')' *
                                                       (Matrix(dat[z[i,j] .== m,:]) .- xbarmap') +
                                                       (kappa0[m] * nz[m]) / (kappa0[m] + nz[m]) *
                                                       (xbarmap - mu0[m,:]) *  (xbarmap - mu0[m,:])'; digits=6)
                            )
                    )
                    mu = rand(
                            MvNormal(
                                (kappa0[m] * mu0[m,:] + nz[m] * xbarmap) / (kappa0[m] + nz[m]),
                                Sigma / (kappa0[m] + nz[m])
                            )
                    )
                    @inbounds NIW[i,j,m] = Dict("mu" => mu, "Sigma" => Sigma)
                else
                    # Draw from the prior
                    Sigma = rand(
                                InverseWishart(kappa0[m],
                                               Psi0[:,:,m] * kappa0[m]
                            )
                    )
                    mu = rand(
                            MvNormal(
                                mu0[m,:],
                                Sigma / kappa0[m]
                            )
                    )
                    @inbounds NIW[i,j,m] = Dict("mu" => mu, "Sigma" => Sigma)
                end
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

            # WRITE SOMETHING THAT SAMPLES RANDOMLY IF WEIGHTS ARE EQUAL?
            # RIGHT NOW ITS BIASED TOWARDS SAMPLING JUST 1.
            # IDK IF IT REALLY MATTERS THOUGH
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
                @>StatsBase.rle()
            end
            nz = zeros(nm)
            nz[rles[1]] = rles[2]
            propout[i,j] = rand(Dirichlet(alpha + nz))
        end
    end
    (zout, NIW, propout)
end

export loglike
function loglike(param::Tuple{Array{Dict{String,Array{Float64,N} where N},1},Array{Float64,1},Array{Int64,1}}, dat)
    """
    Define a log-likelihood function to get passed around
    * param is a tuple containing mus, sigmas, prop, and z for a given walker, temperature
    * specifically, param = ( Dict(NIW), Array(prop), Array(z) )
    """
    NIW, prop, z = param
    nm = size(prop, 1)
    n, dm = size(dat)

    # Compute the log-likelihood of the data given the NIW param and class labels
    ll_normal = zeros(nm)
    for m in 1:nm
        if sum(z .== m) >= dm
            ll_normal[m] =
                sum(
                    logpdf(
                            MvNormal(get(NIW[m], "mu", 0), get(NIW[m], "Sigma", 0)),
                            Matrix(dat[z .== m,:])'
                    )
                )
        else
            ll_normal[m] = -Inf # used to be 0
        end
    end

    # Compute the log-likelihood of the class labels given the NIW param and mixing proportions
    distns = map(x -> MvNormal(get(x, "mu", 0), get(x, "Sigma", 0)), NIW)
    p = Array{Float64,2}(undef,n,nm)
    for m in 1:nm
        p[:,m] = pdf(distns[m], Matrix(dat)') * prop[m]
    end

    # Convert density to probabilities
    @views prob = @> begin
            p'
            @>exp.()
            @>> mapslices(x -> all(x .== 0.0) ? rep(1, times = nm) : x, dims = 1)
            @>> mapslices(x -> x/sum(x), dims = 1)
    end

    mult_distns = mapslices(x -> Multinomial(1,x), prob, dims = 1)
    ll_mult = zeros(n)
    for i in 1:n
        occ = Int64.(zeros(nm))
        @inbounds occ[z[i]] = 1
        @inbounds ll_mult[i] = logpdf(mult_distns[i], occ)
    end

    return sum(ll_mult) + sum(ll_normal)
end

export logprior
function logprior(prop, alpha)
    """
    Define a log-likelihood function to get passed around
    * alpha is an nm array of hyperparameters for the mixing proportions
    * prop is an nm array of a current sample of mixing proportions
    """
    return logpdf(Dirichlet(alpha), prop)
end

export lnlike_lnprior
function lnlike_lnprior(dat, param, alpha, ll, lp)
    """
    For the mixing proportions at a given walker, temperature
        evaluates log-likelihood and log prior at the proposal
    This is:
    1. log-likelihood of data given all param (except mixing prop)
    2. log-likelihood of labels given param and mixing prop
    3. Dirichlet prior on the mixing prop
    Returns Inf if log prior is Inf at the parameters
    """
    NIW, prop, z = param
    p = lp(prop, alpha)
    if p == -Inf
        (-Inf, -Inf)
    else
        (ll(param, dat), p)
    end
end

export lnlikes_lnpriors
function lnlikes_lnpriors(dat, param, alpha, ll, lp)
  """
  * dat is the n x dm data matrix (currently treating it like a DataFrame I think)
  * param is a multidimensional nw x nt array that contains the tuple ( Dict(NIW), Array(prop), Array(z) )
        for each walker at each temperature
  * alpha is an nm array of prior weights for the mixing proportions
  * ll = log-likelihood function
  * lp = log prior function
  """
    nw, nt = size(param)

    # If there is capacity to parallelize, then parallelize
    if nworkers() > 1
        # unfolds param column-wise
        parr = reshape(param, nw*nt)
        lls_lps = pmap(x -> lnlike_lnprior(dat, x, alpha, ll, lp), parr)
        lls = reshape(Float64[l for (l,p) in lls_lps], (nw, nt))
        lps = reshape(Float64[p for (l,p) in lls_lps], (nw, nt))
    else
        lls = zeros(nw, nt)
        lps = zeros(nw, nt)
        for i in 1:nw
            for j in 1:nt
                @inbounds lls[i,j], lps[i,j] = lnlike_lnprior(dat, param[i,j], alpha, ll, lp)
            end
        end
    end
    (lls, lps)
end

export make_mcmc_move
function make_mcmc_move(dat, param, hyp, alpha, ll, lp, betas)
    nw, nt = size(param)
    NIW = map(x -> x[1], param)
    prop = map(x -> x[2], param)
    z = map(x -> x[3], param)

    """
    1. Make gibbs updates of mus, Sigmas, mixing weights, and cluster labels
        * z is an int nw x nt dimensional array, where each entry is of length n
            classifying each observation for each walker and temperature
        * NIW is an an nw x nt array of dictionaries, each with nm keys "mu" and "Sigma"
    3. Compute the log-likelihood and log prior at the prop_star proposal
    4. Compute the hastings ratio for the proposed mixing proportions
    """
    z, NIW, prop = make_gibbs_update(dat, hyp, z, prop, alpha) # added prop here (and alpha passed to make_gibbs_update)

    # Update param to reflect gibbs updates
    for i in 1:nw
        for j in 1:nt
            map!(x -> x, param[i,j][1], NIW[i,j,:])
            map!(x -> x, param[i,j][2], prop[i,j])
            map!(x -> x, param[i,j][3], z[i,j])
        end
    end

    """
    * llps is an nw x nt array of log-likelihoods of given parameters
    * lpps is an nw x nt array of log priors of given parameters
    """
    llps, lpps = lnlikes_lnpriors(dat, param, alpha, ll, lp)

    (param, llps, lpps)
end

export make_tswap
function make_tswap(param, lnlikes, lnpriors, betas)
    """
    * twaps counts the number of swaps that happen between various temps
    """
    nw, nt = size(param)
    new_param = copy(param)
    new_lnlikes = copy(lnlikes)
    new_lnpriors = copy(lnpriors)
    tswaps = zeros(Int, nt-1)

    for k in nt:-1:2 # From nt to 2, decreasing by 1
        for j in 1:nw
            # Randomly select another walker index
            jj = rand(1:nw)

            """
            log-likelihood for current walker, temperature
            and some other randomly selected walker, adjacent temperature
            """
            llhigh = new_lnlikes[j,k]
            lllow = new_lnlikes[jj, k-1]

            """
            Computing the difference in temperatures of two walkers selected
             (higher temp - lower temp)
            """
            diff_beta = betas[k] - betas[k-1]

            """
            probability of acceptance = [beta1-beta2] * (log-likelihood2 - log-likelihood1)
            This maintains detailed balance, and note that if diff_beta were too large,
              then we would almost never accept. That is why we propose swaps between consecutive temperatures
            """
            lpacc = diff_beta*(lllow - llhigh)

            # I think i am wasting some memory in this function with the update step
            if log(rand()) < lpacc
                """
                If accepted, swap the log-likelihood, log prior, and parameter estimates
                  between the chains at the two given temperatures
                """
                # Copy these so I don't write over them and lose them
                phigh = new_param[j,k]
                lphigh = new_lnpriors[j,k]

                # Make the swap
                new_param[j,k] = new_param[jj,k-1]
                new_lnlikes[j,k] = new_lnlikes[jj,k-1]
                new_lnpriors[j,k] = new_lnpriors[jj,k-1]

                new_param[jj,k-1] = phigh
                new_lnlikes[jj,k-1] = llhigh
                new_lnpriors[jj,k-1] = lphigh

                # Note that I have make a swap at this between temps k and k-1
                tswaps[k-1] += 1
            end # If not accepted, do nothing
        end
    end
    new_param, new_lnlikes, new_lnpriors, tswaps
end

export tevolve
function tevolve(swapfraction, betas, tc)
    """
    Temperature evolution is formulated in terms of a quantity `x[i]`, which is the logit
    transform of `beta[i]` relative to its neighbours (`beta` is the
    inverse temperature, so decreases as `i` increases):

        x[i] = log(beta[i] - beta[i+1]) - log(beta[i-1] - beta[i])

    `x` maps the range between `beta[i+1]` and `beta[i-1]` to `-Inf` to `Inf`.  We
    evolve `x` via

        x_new[i] = x[i] + 2.0*(swapfraction[i] - swapfraction[i-1])/(tc*(swapfraction[i] + swapfraction[i-1]))

    where `swapfraction[i]` measures the fraction of accepted swaps between
    temperature `i` and temperature `i+1` (i.e. between chain `i` and the
    next-highest temperature chain) and `tc` is a "time constant" that controls the
    (exponential) rate of convergence of `x`.

    This is similar to the evolution rule in [Vousden, Farr, & Mandel
    (2016)](https://ui.adsabs.harvard.edu/#abs/2016MNRAS.455.1919V/abstract).
    """
    new_betas = deepcopy(betas)

    # This seems problematic if number of temperatures is < 3
    for i in 2:size(betas, 1)-1
        if swapfraction[i] == 0 && swapfraction[i-1] == 0
            # Do nothing since we don't know which way to move
        else
            x = log(betas[i] - betas[i+1]) - log(betas[i-1] - betas[i])
            xnew = x + 2.0*(swapfraction[i] - swapfraction[i-1])/(tc*(swapfraction[i] + swapfraction[i-1]))
            exnew = exp(xnew)
            new_betas[i] = (new_betas[i+1] + exnew*new_betas[i-1])/(1.0 + exnew)
        end
    end

    new_betas
end

export run_mcmc
function run_mcmc(dat::DataFrame,
                    param::Array,
                    hyp::Tuple,
                    alpha::Array{Float64,1},
                    loglike,
                    logprior,
                    betas::Array{Float64,1},
                    nstep::Int64,
                    burnin::Int64)
    """
      The function will return `(chain, chainloglike, chainlogprior, new_betas)` where
        each returned chain value has an extra dimension appended counting steps of the
        chain (so `chain` is of shape `(ndim, nwalkers, ntemp, nstep)`, for example).
      * dat is an n x nd array of observations
      * alpha is an nm array of hyperparameters for the mixing proportions
      * loglike = function giving the log-likelihood for parameters.
      * logprior = function giving the log-prior for parameters.
      * betas = an [ntemp,] array of inverse temperatures.
      * burnin = number of steps used to tune the temperature (to be thrown away)
      * nstep = the number of steps of the already tuned mcmc
      * nd = number of dimensions
      * nw = number of walkers
      * nt = number of temperatures
      * param contains current parameter estimates for each dimension
            across walkers and temperatures
      * the time constant is one quarter of burnin
    """
    ll = loglike
    lp = logprior

    # Instantiate each object to the correct size
    nw, nt = size(param)
    n, nd = size(dat)

    @assert size(betas,1)==nt "size(betas,1) must be equal to the number of temperatures"

    # The burn-in period
    @showprogress 1 "Computing for burn-in..." for i in 1:burnin
        # Update NIW, z, and mixing weights with Gibbs, store results
        param, lnlikes, lnpriors = make_mcmc_move(dat, param, hyp, alpha, loglike, logprior, betas)

        # Propose swaps between various chains at various temperatures
        # Also track (during this burnin period) how often and which walkers swap
        param, lnlikes, lnpriors, tswaps = make_tswap(param, lnlikes, lnpriors, betas)
        swapfraction = tswaps / nw
        tc = 10.0 + i/10.0 # tc > 10 always, but asymptotes to "current length / 10"

        # Adaptively tune the temperatures (looks a lot like formula for a spring)
        betas = tevolve(swapfraction, betas, tc)
    end

    """
    Pre-allocating memory, and setting up thinning (defaults to thin=1 though)
    Chain stores parameter estimates, chainlnlike the log-likelihoods, and
      chainlnprior the log priors for each dimension, walker, and temperature
    """
    chain = Array{
                    Tuple{Array{Dict{String,Array{Float64,N} where N},1},
                    Array{Float64,1},
                    Array{Int64,1}}
            }(undef, nw, nt, nstep)
    chainlnlike = zeros(nw, nt, nstep)
    chainlnprior = zeros(nw, nt, nstep)

    @showprogress 1 "Computing for main Markov chain..."  for i in 1:nstep
      """
      Alternate between usual MCMC stuff and swapping temperatures
      Temperatures no longer evolve (to satisfy detailed balance)
      May also want to adjust -- multiple MCMC updates in between each swap step
      """
        param, lnlikes, lnpriors = make_mcmc_move(dat, param, hyp, alpha, loglike, logprior, betas)
        param, lnlikes, lnpriors, _ = make_tswap(param, lnlikes, lnpriors, betas)

        @inbounds chain[:,:,i] = deepcopy(param)
        @inbounds chainlnlike[:,:,i] = lnlikes
        @inbounds chainlnprior[:,:,i] = lnpriors
    end

    chain, chainlnlike, chainlnprior, betas
end

export get_mu_chain
function get_mu_chain(chain, walker_num, cluster_num)
    """
    For a given walker and cluster number, return
        the corresponding chain of mu estimates in each dimension
    This is used to pass results back to R via JuliaCall
    """
    nw, nt, nstep = size(chain)
    nm = size(chain[1,1,1][1], 1)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"
    @assert nm >= cluster_num "cluster_num must be less than or equal to the number of clusters"

    # The first line extracts NIW part of the chain
    # Then, get mu for given walker and cluster number
    # Shoud be a dm x nstep chain (dm is dimension of data)
    mu_chain = @as mu_chain map(x -> x[1], chain[:,1,:]) begin
        @>> @views mapslices(x -> x[walker_num], mu_chain; dims = 1)
        @>> @views mapslices(x -> x[cluster_num], mu_chain; dims = 1)
        @>> map(x -> get(x, "mu", 0), mu_chain)
        @> hcat(mu_chain...)
    end
    return mu_chain
end

export get_Sigma_chain
function get_Sigma_chain(chain, walker_num, cluster_num)
    """
    For a given walker and cluster number, return
        the corresponding chain of covariance estimates in each dimension
    This is used to pass results back to R via JuliaCall
    """
    nw, nt, nstep = size(chain)
    nm =  size(chain[1,1,1][1], 1)
    dm = size(get(chain[1,1,1][1][1], "Sigma", 0), 1)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"
    @assert nm >= cluster_num "cluster_num must be less than or equal to the number of clusters"

    # The first line extracts NIW part of the chain
    # Then, get mu for given walker and cluster number
    # Shoud be a dm x dm x nstep array of Sigma estimates (dm is dimension of data)
    Sigma_chain = @as Sigma_chain map(x -> x[1], chain[:,1,:]) begin
        @>> @views mapslices(x -> x[walker_num], Sigma_chain; dims = 1)
        @>> @views mapslices(x -> x[cluster_num], Sigma_chain; dims = 1)
        @>> map(x -> get(x, "Sigma", 0), Sigma_chain)
        @> hcat(Sigma_chain...)
        @> reshape(Sigma_chain, (dm, dm, nstep))
    end

    return Sigma_chain
end

export get_prop_chain
function get_prop_chain(chain, walker_num)
    """
    For a given walker, return the corresponding
        chain of mixing weight estimates for each cluster
    This is used to pass results back to R via JuliaCall
    """
    nw, nt, nstep = size(chain)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"

    # Shoud be a nm x nstep chain
    prop_chain = hcat(map(x -> x[2], chain[:,1,:])[walker_num,:]...)

    return prop_chain
end

export get_z_chain
function get_z_chain(chain, walker_num)
    """
    For a given walker, return the
        corresponding chain of z estimates for each cluster and each observation
    This is used to pass results back to R via JuliaCall
    """
    nw, nt, nstep = size(chain)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"

    # Returns an n x nstep chain of cluster labels for each observation across the whole chain
    z_chain = @as z_chain map(x -> x[3], chain[:,1,:]) begin
                @> z_chain[walker_num,:]
                @> hcat(z_chain...)
            end

    return z_chain
end

end # End the module
