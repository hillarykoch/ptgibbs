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

    # Construct the pairs of dimensions of A
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

        # Compute bounds on A[m,n]
        if mstar == 1
            @inbounds premult = U[mstar,mstar] * A[mstar,mstar]
            term4 = 0.0
            for i in mstar:(pstar-1)
                #global term4
                @inbounds term4 += U[i,pstar] * A[i,mstar]
            end

            if @inbounds sign(h[m] * h[p]) == -1
                lb = -Inf
                ub = @inbounds ( -term4 ) / U[pstar, pstar]#(0 - outterm4) / denom
            else
                lb = @inbounds ( -term4 ) / U[pstar, pstar]#(0 - outterm4) / denom
                ub = Inf
            end
            #llb = min(lb, ub)
            #uub = max(lb, ub)
            #@inbounds A[pstar,mstar] = rand(Truncated(Normal(), llb, uub))
            @inbounds A[pstar,mstar] = rand(Truncated(Normal(), lb, ub))
        else
            term1 = 0.0
            for k in 1:(mstar-1)
                outerterm = 0.0
                for j in 1:k
                    innerterm = 0.0
                    for i in j:pstar
                        @inbounds innerterm += (A[i,j] * U[i,pstar])
                    end
                    @inbounds outerterm += (innerterm * A[k,j])
                end
                #global term1
                @inbounds term1 += (U[k,mstar] * outerterm)
            end

            outerterm2 = 0.0
            for j in 1:(mstar-1)
                innerterm2 = 0.0
                for i in j:pstar
                    @inbounds innerterm2 += (A[i,j] * U[i,pstar])
                end
                #global outerterm2
                @inbounds outerterm2 += (innerterm2 * A[mstar,j])
            end
            @inbounds term2 = U[mstar,mstar] * outerterm2

            term3 = 0.0
            for i in mstar:(pstar-1)
                #global term3
                @inbounds term3 += (A[i,mstar] * U[i,pstar])
            end
            @inbounds term3 *= (A[mstar,mstar] * U[mstar,mstar])


            if @inbounds sign(h[m] * h[p]) == -1
                lb = -Inf
                ub = @inbounds ((-term1 - term2 - term3) / (U[pstar,pstar] * U[mstar,mstar] * A[mstar,mstar]) )
            else
                lb = @inbounds ((-term1 - term2 - term3) / (U[pstar,pstar] * U[mstar,mstar] * A[mstar,mstar]) )
                ub = Inf
            end
            #llb = min(lb, ub)
            #uub = max(lb, ub)
            #@inbounds A[pstar,mstar] = rand(Truncated(Normal(), llb, uub))
            @inbounds A[pstar,mstar] = rand(Truncated(Normal(), lb, ub))
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
