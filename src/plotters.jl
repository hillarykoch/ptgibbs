using Gadfly
using Statistics
using Clustering
using DataFrames

import Distances: pairwise, Euclidean
import StatsBase: cov2cor
import RLEVectors: rep

#include("helpers.jl")

export plot_corr
function plot_corr(chain, m::Int64, labs::Array{String,1}; reorder = true, key = false, min_value = -1, max_value = 1, linkage = :ward)
"""
    Plot various correlation matrices
    1. Obtain some covariance chain and average over the estimates
    2. Convert to a correlation matrix
    3. Cluster using hierarchical clustering, and reorder the dimensions based on the output
    4. Convert to a molten data frame and plot
"""
    Schain = get_Sigma_chain(chain, m)
    covmat = mapslices(x -> mean(x), Schain; dims = 3)[:,:,1]
    cormat = cov2cor(covmat, sqrt.(diag(covmat)))

    if reorder
            D = pairwise(Euclidean(), cormat)
            try
                    cl = Clustering.hclust(D, linkage=linkage)
                    ordering = getproperty(cl, :order)
            catch
                    ordering = 1:size(cormat,1)
            end
    else
            ordering = 1:size(cormat,1)
    end

    df = DataFrame(cormat[ordering, ordering], Symbol.(labs[ordering]))
    molten = melt(df)
    molten.v2 = rep(labs[ordering], times = size(df,1))
    Gadfly.plot(molten, x="variable", y="v2", color="value",
        Geom.rectbin,
        Coord.cartesian(fixed = true),
        Guide.xlabel(nothing), Guide.ylabel(nothing),
        Scale.color_continuous(minvalue=min_value, maxvalue=max_value, colormap = Scale.lab_gradient("white", "red")),
        Guide.colorkey(title=""),
        key ? Theme(key_position = :right) : Theme(key_position = :none))
end

export plot_eigvec
function plot_eigvec(chain, m::Int64, labs::Array{String,1}, eig_num::Int64; pal=missing, reorder=missing)
        Schain = get_Sigma_chain(chain, m)
        covmat = mapslices(x -> mean(x), Schain; dims = 3)[:,:,1]

        if !ismissing(reorder)
                covmat = covmat[reorder,reorder]
                labs = labs[reorder]
                pal = pal[reorder]
        end

        evals, evecs = (eigvals(covmat), eigvecs(covmat))

        # Make names dataframe so plot has names
        # The kth ROW is the kth eigenvector here
        edf = DataFrame(evecs', Symbol.(labs))
        edf.vec = 1:size(covmat, 1)
        molten = melt(edf)

        # Plot first eigenvector
        if ismissing(pal)
                p = Gadfly.plot(molten[molten.vec .== eig_num, :],
                        x=:variable,
                        y=:value,
                        Guide.xlabel(nothing), Guide.ylabel(nothing),
                        Geom.bar)#,
                        #yintercept = 0,
                        #Geom.hline(color="black", style = :dash))
        else
                p = Gadfly.plot(molten[molten.vec .== eig_num, :],
                        x=:variable,
                        y=:value,
                        color=pal,
                        Guide.xlabel(nothing), Guide.ylabel(nothing),
                        Geom.bar)#,
                        #yintercept = 0,
                        #Geom.hline(color="black", style = :dash))
        end
        (p, sort(evals; rev=true))
end

export plot_effects
function plot_effects(chain, obsidx::Int64, labs::Array{String,1}, M::Int64, mu_ests; pal = missing)
        z_chain = get_z_chain(chain)
        MAPs = mapslices(x -> [ mean(x .== y) for y in 1:M ], z_chain; dims = 2)
        emap = MAPs[obsidx,:]

        # Effect size estimates in each cell type,
        #       weighted by the posterior probability the this site is in any given cluster
        #       plus the standard errors
        emeans = mapslices(x -> sum(x), hcat([ mu_ests[m,:] .* emap[m] for m in 1:M ]...)'; dims = 1)[1,:]
        eses = mapslices(y -> sum(y), hcat([ mapslices(x -> std(x), get_mu_chain(chain, m);
                                dims = 2) .* emap[m] for m in 1:M ]...)'; dims = 1)[1,:]


        # Plot first eigenvector
        if ismissing(pal)
                Gadfly.plot(y = labs, x = emeans,
                                xmin = emeans .- eses,
                                xmax = emeans .+ eses,
                                Geom.point,
                                Guide.xlabel(nothing), Guide.ylabel(nothing),
                                Geom.xerrorbar)
        else
                Gadfly.plot(y = labs, x = emeans,
                                xmin = emeans .- eses,
                                xmax = emeans .+ eses,
                                Geom.point,
                                Geom.xerrorbar,
                                Guide.xlabel(nothing), Guide.ylabel(nothing),
                                color = pal)
        end
end
