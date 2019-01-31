using Distributions
using LinearAlgebra

export test_Bmn
function test_Bmn(A::Array{Float64,2}, U::UpperTriangular{Float64,Array{Float64,2}}, m::Int64, n::Int64)
    # WLOG, m < n
    subA = @views @inbounds A[1:1:n, 1:1:n]
    subU = @views @inbounds U[1:1:n, 1:1:n]

    @inbounds ( subU' * subA * subA' * subU )[m, n]
end

export rand_constrained_IW
function rand_constrained_IW(Psi0, nu, h)
    """
    Psi0 is some pos-def matrix that complies with restrictions imposed by h
    dm is the dimension
    L is the Upper Triangular cholesky factor for Psi0
    A is some random matrix where we impose restrictions to meet our needs
    """

    dm = size(Psi0, 1)
    U = cholesky(Psi0).U
    A = zeros((dm, dm))

    # Sample the diagonal of A
    for i =  1:1:dm
        @inbounds h[i] != 0 ? A[i,i] = sqrt(rand(Chisq(nu - i + 1))) : A[i,i] = sqrt(nu)
    end

    # Construst the pairs of dimensions of A
    # in the order in which we need to simulate them
    npairs = binomial(dm,2)
    dimpairs = Matrix{Int64}(undef, npairs, 2)
    counter = 1
    for k = 1:1:(dm-1)
        for i=1:1:k
            @inbounds dimpairs[counter,:] = [i, k+1]
            counter = counter + 1
        end
    end

    # Simulate the IW matrix, starting from the top left corner
    # and working our way down
    for i = 1:1:npairs
        @inbounds m = dimpairs[i,1]
        @inbounds n = dimpairs[i,2]
        if @inbounds Psi0[m, n] != 0
            pass = false
            while !pass
                samp = rand(Normal())
                Atest = A

                @inbounds Atest[n, m] = samp
                testbmn = test_Bmn(Atest, U, m, n)
                if sign(testbmn) == @inbounds sign(Psi0[m,n])
                    pass = true
                    A = deepcopy(Atest)
                end
            end
        end
    end
    (U' * A * A' * U) / nu
end
