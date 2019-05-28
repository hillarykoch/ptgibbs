export approx_tmvn_norm_const
function approx_tmvn_norm_const(mu::Array{Float64,1},
    Sigma_t::Array{Float64,2},
    Sigma_star::Array{Float64,2})

    subidx = mu .!= 0
    nonzerotot = sum(subidx)
    if nonzerotot == 0
        return 1
    else
        @inbounds sub_Sigma_t = Sigma_t[subidx, subidx]
        @inbounds sub_Sigma_star = Sigma_star[subidx, subidx]
        @inbounds sub_mu = mu[subidx]
        sub_h = [ sign(sub_mu[i]) for i in 1:nonzerotot ]
        num = rand(MvNormal(sub_mu, Matrix(Hermitian(sub_Sigma_star))), 1000)
        denom = rand(MvNormal(sub_mu, Matrix(Hermitian(sub_Sigma_t))), 1000)
        @inbounds @views num_in = mean(mapslices(x -> all([sign(x[i]) == sub_h[i] for i in 1:nonzerotot ]), num; dims = 1))
        @inbounds @views denom_in = mean(mapslices(x -> all([sign(x[i]) == sub_h[i] for i in 1:nonzerotot ]), denom; dims = 1))

        # If it is rare for both, just say they are both equally likely
        if all([num_in == 0.0, denom_in == 0.0])
            return 1
        else
            return num_in/denom_in
        end
    end
end

export approx_cwish_norm_const
function approx_cwish_norm_const(tune_df::Float64,
    Sigma_t::Array{Float64,2},
    Sigma_star::Array{Float64,2})

    subidx = diag(Sigma_t) .!= 1.0
    nonzerotot = sum(subidx)
    if nonzerotot == 0
        return 1
    else
        @inbounds sub_Sigma_t = Sigma_t[subidx, subidx]
        @inbounds sub_Sigma_star = Sigma_star[subidx, subidx]
        sub_h = sign.(UpperTriangular(sub_Sigma_t))
        num = @views map(x -> UpperTriangular(x), rand(Wishart(tune_df, Matrix(Hermitian(sub_Sigma_star))), 1000))
        denom = @views map(x -> UpperTriangular(x), rand(Wishart(tune_df, Matrix(Hermitian(sub_Sigma_t))), 1000))

        @inbounds @views num_in = mean(map(x -> all([ sign(x[i]) == sub_h[i] for i in 1:nonzerotot ]), num))
        @inbounds @views denom_in = mean(map(x -> all([ sign(x[i]) == sub_h[i] for i in 1:nonzerotot ]), denom))

        # If it is rare for both, just say they are both equally likely
        if @inbounds all([num_in == 0.0, denom_in == 0.0])
            return 1
        else
            return num_in/denom_in
        end
    end
end
