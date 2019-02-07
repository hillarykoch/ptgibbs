import StatsBase: rle, pweights
import RLEVectors: rep
import DataFrames: DataFrame, colwise

using Distributed
using Statistics
using Distributions
using LinearAlgebra
using Lazy
using ProgressMeter

export make_constr_beta1_gibbs_update
function make_constr_beta1_gibbs_update(dat, hyp, z, prop, alpha, labels)
    nw = size(prop, 1)
    nm = size(prop[1],1)
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
                round.(
                Matrix(Hermitian(
                    Psi0[:,:,m] * max(kappa0[m], dm) +
                    (Matrix(dat[z[i] .== m,:]) .- xbarmap')' *
                    (Matrix(dat[z[i] .== m,:]) .- xbarmap') +
                    (max(kappa0[m], dm) * nz[m]) / (max(kappa0[m], dm) + nz[m]) *
                    (xbarmap - mu0[m,:]) *  (xbarmap - mu0[m,:])'
                )); digits = 8)

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
                        round.(
                        Matrix(Hermitian(
                        Psi0[:,:,m])),; digits=8),
                        max(kappa0[m], dm),
                        labels[m] .- 1
                    ) ./ max(kappa0[m], dm)
                else
                    @inbounds Sigma = rand_constrained_IW(
                        rcIW_in,
                        max(kappa0[m], dm) + nz[m],
                        labels[m] .- 1) ./ (max(kappa0[m], dm) + nz[m])
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
                    @inbounds mu =
                    rand_constrained_MVN(
                        round.(
                        Matrix(Hermitian(
                        Sigma)); digits = 8),
                        mu0[m,:],
                        labels[m] .- 1
                    )
                else
                    @inbounds mu =
                        rand_constrained_MVN(
                            round.(
                            Matrix(Hermitian(
                            Sigma)); digits = 8),
                            rcMVN_in,
                            labels[m] .- 1
                            )
                @inbounds NIW[i,m] = Dict("mu" => mu, "Sigma" => Sigma)
            else
                # Draw from the prior
                @inbounds Sigma = rand_constrained_IW(
                    round.(
                    Matrix(Hermitian(Psi0[:,:,m])); digits=8),
                    max(kappa0[m], dm),
                    labels[m] .- 1
                ) ./ max(kappa0[m], dm)

                @inbounds mu = rand_constrained_MVN(
                    round.(
                    Matrix(Hermitian(Sigma)); digits = 8),
                    mu0[m,:],
                    labels[m] .- 1
                )

                @inbounds NIW[i,m] = Dict("mu" => mu, "Sigma" => Sigma)
            end
        end
    end

    zout = copy(z)
    for i in 1:nw
        distns = map(x -> MvNormal(x["mu"], Matrix(Hermitian(x["Sigma"]))), NIW[i,:])
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
        map!(x -> x, param[i,1][1], NIW[i,:])
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
    labels = hcat([parse.(Int64, split(x, "")) for x in labels])

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
    #global param
        param, lnlikes, lnpriors = make_constr_beta1_mcmc_move(dat, param, hyp, alpha, loglike, logprior, labels)

        @inbounds chain[:,i] = deepcopy(param)
        @inbounds chainlnlike[:,i] = lnlikes
        @inbounds chainlnprior[:,i] = lnpriors
    end

    chain, chainlnlike, chainlnprior
end
