using Gadfly
using Statistics
using Distances
using Clustering
using DataFrames

import StatsBase: cov2cor
import RLEVectors: rep

export plot_corr
function plot_corr(chain, m::Int64, labs::Array{String,1}; reorder = true)
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
            cl = hclust(D, linkage=:ward)
            order = getproperty(cl, :order)
    else
            order = 1:size(cormat,1)
    end

    df = DataFrame(cormat[order, order], Symbol.(labs[order]))
    molten = melt(df)
    molten.v2 = rep(labs[order], times = size(df,1))
    Gadfly.plot(molten, x="variable", y="v2", color="value",
        Geom.rectbin,
        Coord.cartesian(fixed = true),
        Guide.xlabel(nothing), Guide.ylabel(nothing))
end
