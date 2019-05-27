

function approx_tmvn_norm_const(mu::Array{Float64,1},
    Sigma_t::Array{Float64,2},
    Sigma_star::Array{Float64,2})

    subidx = mu .!= 0
    if sum(subidx) == 0
        return 1
    else
        sub_Sigma_t = Sigma_t[subidx, subidx]
        sub_Sigma_star = Sigma_star[subidx, subidx]
        sub_mu = mu[subidx]
        sub_h = sign.(sub_mu)
        num = rand(MvNormal(sub_mu, sub_Sigma_star), 10000)
        denom = rand(MvNormal(sub_mu, sub_Sigma_t), 10000)
        num_in = mean(mapslices(x -> all(sign.(x) .== sub_h), num; dims = 1))
        denom_in = mean(mapslices(x -> all(sign.(x) .== sub_h), denom; dims = 1))

        # If it is rare for both, just say they are both equally likely
        if all([num_in, denom_in] .== 0.0)
            return 1
        else
            return num_in/denom_in
        end
    end
end

function approx_cwish_norm_const(tune_df::Float64,
    Sigma_t::Array{Float64,2},
    Sigma_star::Array{Float64,2},
    h::Array)

    subidx = diag(Sigma_t) .!= 1.0
    if sum(subidx) == 0
        return 1
    else
        sub_Sigma_t = Sigma_t[subidx, subidx]
        sub_Sigma_star = Sigma_star[subidx, subidx]
        num = rand(Wishart(tune_df, sub_Sigma_star), 10000)
        denom = rand(Wishart(tune_df, sub_Sigma_t), 10000)

        num_in = mean(map(x -> all(sign.(x) .== sign.(sub_h)), num))
        denom_in = mean(map(x -> all(sign.(x) .== sign.(sub_h)), denom))

        # If it is rare for both, just say they are both equally likely
        if all([num_in, denom_in] .== 0.0)
            return 1
        else
            return num_in/denom_in
        end
    end
end
