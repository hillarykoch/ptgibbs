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
                    @>rle()
                end
            nz = zeros(nm)
            @inbounds nz[rles[1]] = rles[2]

            """
            Compute the d-dimensional sample mean for each class
                for the current walker, current temperature
            """
            # xbar is an array of d dictionaries
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
                    @inbounds Sigma = rand(
                                InverseWishart(kappa0[m] + nz[m],
                                               round.(Psi0[:,:,m] * kappa0[m] +
                                                  (Matrix(dat[z[i,j] .== m,:]) .- xbarmap')' *
                                                       (Matrix(dat[z[i,j] .== m,:]) .- xbarmap') +
                                                       (kappa0[m] * nz[m]) / (kappa0[m] + nz[m]) *
                                                       (xbarmap - mu0[m,:]) *  (xbarmap - mu0[m,:])'; digits=6)
                            )
                    )

                    # Possibly turn this ifelse statement into a separate function,
                    # fix_rhos, to improve performance
                    # Consider also, instead of post-hoc setting to zero, only drawing relevant dimensions
                    # and then updating those alone
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

                    # Possibly turn this ifelse statement into a separate function,
                    # fix_mus, to improve performance
                    @inbounds zero_loc = labels[m] .== "1"
                    @inbounds mu[zero_loc] = zeros(sum(zero_loc))


                    @inbounds NIW[i,j,m] = Dict("mu" => mu, "Sigma" => Sigma)
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
