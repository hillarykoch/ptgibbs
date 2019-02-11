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
