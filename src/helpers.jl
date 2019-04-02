using ProgressMeter

import DataFrames: DataFrame, colwise

export tapply_mean
function tapply_mean(subs, val, sz=(maximum(subs),))
    A = zeros(eltype(val), sz...)
    counter = zeros(Int64, sz...)
    for i = 1:length(val)
        @inbounds A[subs[i]] += val[i]
        @inbounds counter[subs[i]] += 1
    end
    A./counter
end

export update_tune_df
function update_tune_df(tune_df::Float64, acpt_rt::Float64, opt_rt::Float64, gamma1::Float64)
    max(exp(log(tune_df) - gamma1 * (acpt_rt - opt_rt)), 50)
end

export extend_mcmc
function extend_mcmc(dat::DataFrame,
                        chain::Array,
                        tune_df_chain::Array,
                        hyp::Tuple,
                        alpha::Array{Float64,1},
                        nstep::Int64,
                        labels::Array{String,1};
                        opt_rate::Float64 = 0.3)
    """
    Take old chain as input and add new iterations
    """
    nw = size(chain, 1)
    n, nd = size(dat)
    nm = size(chain[1,1][2], 1)
    labels = hcat([(parse.(Int64, split(x, "")) .-1) for x in labels])
    curLen = size(chain, 2)
    tune_df = tune_df_chain[:,curLen-1]

    # Initial estimates at NIW, prop, and z for each walker and each temperature
    param = Array{Tuple{Array{Dict{String,Array{Float64,N} where N},1},Array{Float64,1},Array{Int64,1}}}(undef, (nw, 1))
    for i in 1:nw
        dictionary = Dict{String,Array{Float64,N} where N}[]
        for m in 1:nm
                Sigma = chain[i,curLen][1][m]["Sigma"]
                mu = chain[i,curLen][1][m]["mu"]
                push!(dictionary, Dict("mu" => mu, "Sigma" => Sigma))
        end
        initprop = chain[i,curLen][2]
        z = chain[i,curLen][3]
        param[i,1] = (dictionary, initprop, z)
    end

    chain = Array{
                    Tuple{Array{Dict{String,Array{Float64,N} where N},1},
                    Array{Float64,1},
                    Array{Int64,1}}
                }(undef, nw, nstep + 1)

    for j in 1:nw
        chain[j,1] = deepcopy(param[j,1])
    end
    acpt_tracker = zeros(nm)

    # For tracking the acceptance rate, per cluster, for the adaptive tuning variance
    win_len = min(nstep, 50)
    acpt_win = zeros(nm, win_len)
    acpt_chain = zeros(nm, nstep)

    old_tune_df = deepcopy(tune_df)
    tune_df_chain = zeros(nm, nstep)

    @showprogress 1 "Running the MCMC..."  for i in 1:nstep
        # Proposes IW cluster at a time, accepts/rejects/returns new IW
        NIW, acpt = make_mcmc_move(dat, chain[:,i], hyp, alpha, labels, tune_df)
        acpt_tracker = acpt_tracker .+ acpt
        @inbounds acpt_win[:, (i-1) % win_len + 1] = acpt

        # Makes Gibbs updates of means, mixing weights, and class labels
        newz, newNIW, newprop = make_gibbs_update(dat, chain[:,i], hyp, alpha, labels)

        if i > 50
            # Update tuning parameter per cluster
            gamma1 = 10 / (i ^ 0.8)
            old_tune_df = copy(tune_df)
            @inbounds [tune_df[m] = update_tune_df(tune_df[m], mean(acpt_win[m,:]), opt_rate, gamma1) for m in 1:1:nm]
        end

        for j in 1:nw
            chain_link = deepcopy(param)

            @inbounds map!(x -> x, chain_link[j,1][1], newNIW[j,1])
            @inbounds map!(x -> x, chain_link[j,1][2], newprop[j,1])
            @inbounds map!(x -> x, chain_link[j,1][3], newz[j,1])

            @inbounds chain[j,i+1] = deepcopy(chain_link[j,1])
            @inbounds acpt_chain[:,i] = acpt_tracker ./ i
            @inbounds tune_df_chain[:,i] = tune_df
        end
    end
    (chain, acpt_chain, tune_df_chain)
end
