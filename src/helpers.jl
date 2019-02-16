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
