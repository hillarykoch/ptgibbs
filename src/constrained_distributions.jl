using Distributions
using LinearAlgebra

export rand_constrained_Wish
function rand_constrained_Wish(Psi0, nu, h)
    """
    Psi0 is some pos-def matrix that complies with restrictions imposed by h
    dm is the dimension
    L is the Upper Triangular cholesky factor for Psi0
    A is some random matrix where we impose restrictions to meet our needs
    """

    # On real data, force 0s and 1s where necessary
    # It's like we don't even sample this which is ok I think
    dm = size(Psi0, 1)
    zeroidx = findall(h .== 0)
    nonzeroidx = setdiff(1:1:dm, zeroidx)

    if size(zeroidx,1) == dm
        return Matrix{Float64}(I, dm, dm) * nu
    elseif size(zeroidx,1) == 0
        subh = h
        subPsi0 = Psi0
    else
        subh = h[nonzeroidx]
        subPsi0 = @inbounds Psi0[nonzeroidx, nonzeroidx]
    end

    subdm = size(subPsi0, 1)

    U = cholesky(Hermitian(subPsi0)).U
    A = zeros((subdm, subdm))

    # Sample the diagonal of A
    for i in 1:subdm
        A[i,i] = sqrt(rand(Chisq(nu - i + 1)))
    end

    # Construst the pairs of dimensions of A
    # in the order in which we need to simulate them
    npairs = binomial(dm,2)
    dimpairs = Matrix{Int64}(undef, npairs, 2)
    counter = 1
    for k = 1:1:(dm-1)
        for i=1:1:k
            #global counter
            @inbounds dimpairs[counter,:] = [i, k+1]
            counter = counter + 1
        end
    end

    if size(zeroidx, 1) > 0
        rmpairs = findall(mapslices(x -> size(intersect(x, zeroidx), 1), dimpairs; dims = 2)[:,1] .!= 0)
        dimpairs = dimpairs[setdiff(1:1:npairs, rmpairs), :]
        npairs = size(dimpairs,1)
    end

    # Simulate the IW matrix, starting from the top left corner
    # and working our way down
    for idx = 1:1:npairs
        @inbounds m = dimpairs[idx,1]
        @inbounds p = dimpairs[idx,2]

        mstar = findall(nonzeroidx .== m)[1]
        pstar = findall(nonzeroidx .== p)[1]

        #@inbounds Atest[nstar, mstar] = samp

        # Compute bounds on A[m,n]
        term1 = 0.0
        term2 = 0.0
        outterm1 = 0.0
        if m == 1
            premult = U[mstar,mstar] * A[mstar,mstar]

            term4 = 0.0
            for i in mstar:(pstar-1)
                #global term4
                term4 += U[i,pstar] * A[i,mstar]
            end
            outterm4 = premult * term4
            denom = premult * U[pstar,pstar]

            if @inbounds sign(h[m] * h[p]) == -1
                lb = @inbounds (-Inf - outterm4) / denom
                ub = @inbounds (0 - outterm4) / denom
            else
                lb = @inbounds (0 - outterm4) / denom
                ub = @inbounds (Inf - outterm4) / denom
            end
            llb = min(lb, ub)
            uub = max(lb, ub)
            @inbounds A[pstar,mstar] = rand(Truncated(Normal(), llb, uub))
        else
            for k in 1:(mstar-1)
                for j in k:mstar
                    #global term1
                    term1 += U[j,mstar] * A[j,k]
                end
                for i in k:pstar
                    #global term2
                    term2 += U[i,pstar] * A[i,k]
                end
                #global outterm1
                outterm1 += (term1 * term2)
            end

            premult = U[mstar,mstar] * A[mstar,mstar]

            term4 = 0.0
            for i in m:(pstar-1)
                #global term4
                term4 += U[i,pstar] * A[i,mstar]
            end
            outterm4 = premult * term4
            denom = premult * U[pstar,pstar]

            if @inbounds sign(h[m] * h[p]) == -1
                lb = @inbounds (-Inf - outterm1 - outterm4) / denom
                ub = @inbounds (0 - outterm1 - outterm4) / denom
            else
                lb = @inbounds (0 - outterm1 - outterm4) / denom
                ub = @inbounds (Inf - outterm1 - outterm4) / denom
            end
            llb = min(lb, ub)
            uub = max(lb, ub)
            @inbounds A[pstar,mstar] = rand(Truncated(Normal(), llb, uub))
        end
    end

    if size(zeroidx,1) == 0
        return U' * A * A' * U
    else
        matmult = (U' * A * A' * U)
        out = Matrix{Float64}(I, dm, dm)
        out[nonzeroidx, nonzeroidx] = matmult

        [out[zz, zz] = nu for zz in zeroidx]
        return out
    end
end

export rand_constrained_MVN
function rand_constrained_MVN(Sigma, mu0, h)
    dm = size(Sigma, 1)
    zeroidx = findall(h .== 0)
    nonzeroidx = setdiff(1:1:dm, zeroidx)

    if size(zeroidx,1) == dm
        return zeros(dm)
    elseif size(zeroidx,1) == 0
        subh = h
        subSigma = Sigma
        submu0 = mu0
    else
        subh = @inbounds h[nonzeroidx]
        subSigma = @inbounds Sigma[nonzeroidx, nonzeroidx]
        submu0 = @inbounds mu0[nonzeroidx]
    end

    subdm = size(nonzeroidx, 1)
    A = cholesky(Hermitian(subSigma)).L

    z = Array{Float64,1}(undef, subdm)
    for i in 1:1:subdm
        bound = -mu0[i]
        if i == 1
            bound /= A[i,i]
        else
            bound = @inbounds (bound - A[i,1:1:(i-1)]' * z[1:1:(i-1)]) / A[i,i]
        end

        if subh[i] == 1
            z[i] = rand(Truncated(Normal(), bound, Inf))
        else
            z[i] = rand(Truncated(Normal(), -Inf, bound))
        end
        bound = nothing
    end

    subout = A * z .+ submu0
    out = zeros(dm)
    out[nonzeroidx] = subout
    out
end
