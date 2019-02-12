using Distributions
using LinearAlgebra

export test_Bmn
function test_Bmn(A::Array{Float64,2}, U::UpperTriangular{Float64,Array{Float64,2}}, m::Int64, n::Int64)
    # WLOG, m < n
    subA = @views @inbounds A[1:1:n, 1:1:n]
    subU = @views @inbounds U[1:1:n, 1:1:n]

    @inbounds ( subU' * subA * subA' * subU )[m, n]
end

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
    zeroidx = findall(h .== 0)
    dm = size(Psi0, 1)

    if size(zeroidx,1) == dm
        return Matrix{Float64}(I, dm, dm) * nu
    elseif size(zeroidx,1) == 0
        subh = h
        subPsi0 = Psi0
    else
        subh = h[setdiff(1:1:dm, zeroidx)]
        subPsi0 = @inbounds Psi0[setdiff(1:1:dm, zeroidx), setdiff(1:1:dm, zeroidx)]
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
    npairs = binomial(subdm,2)
    dimpairs = Matrix{Int64}(undef, npairs, 2)
    counter = 1
    for k = 1:1:(subdm-1)
        for i=1:1:k
            #global counter
            @inbounds dimpairs[counter,:] = [i, k+1]
            counter = counter + 1
        end
    end

    # Simulate the IW matrix, starting from the top left corner
    # and working our way down
    for i = 1:1:npairs
        @inbounds m = dimpairs[i,1]
        @inbounds n = dimpairs[i,2]
        if @inbounds h[m] * h[n] != 0
            #global A
            pass = false
            while !pass
                samp = rand(Normal())
                Atest = A

                @inbounds Atest[n, m] = samp
                testbmn = test_Bmn(Atest, U, m, n)
                if sign(testbmn) == @inbounds sign(h[m] *h[n])
                    pass = true
                    A = deepcopy(Atest)
                end
            end
        end
    end

    if size(zeroidx,1) == 0
        return U' * A * A' * U
    else
        matmult = (U' * A * A' * U)
        out = Matrix{Float64}(I, dm, dm)
        out[setdiff(1:1:dm, zeroidx), setdiff(1:1:dm, zeroidx)] = matmult
        out[setdiff(1:1:dm, zeroidx), setdiff(1:1:dm, zeroidx)] = [nu]
        return out
    end
end


export rand_constrained_MVN
function rand_constrained_MVN(Sigma, mu0, h)
    dm = size(h,1)

    A = cholesky(Hermitian(Sigma)).L

    zeroidx = findall(h .== 0)
    sz = size(zeroidx,1)
    if sz > 0
        [A[zz, :] = zeros(dm) for zz in zeroidx]
        [A[:, zz] = zeros(dm) for zz in zeroidx]
        [A[zz,zz] = 1 for zz in zeroidx]
    end

    # Find lower/upper bound for z
    lub = inv(Matrix(A)) * (-1 * mu0)

    # Simulate truncated normal noise
    z = [ ( sign(h[b]) == 1 ? rand(Truncated(Normal(), lub[b], Inf)) :
            rand(Truncated(Normal(), -Inf, lub[b])) ) for b in 1:1:dm]

    # Adjust the sign as appropriate
    if sz > 0
        z[zeroidx] = zeros(sz)
        mu0[zeroidx] = zeros(sz)
    end

    A * z .+ mu0
end
