using Lazy

export get_mu_chain
function get_mu_chain(chain, cluster_num; walker_num = 1)
    """
    For a given walker and cluster number, return
        the corresponding chain of mu estimates in each dimension
    This is used to pass results back to R via JuliaCall
    """
    nw, nstep = size(chain)
    nm = size(chain[1,1][2], 1)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"
    @assert nm >= cluster_num "cluster_num must be less than or equal to the number of clusters"

    # The first line extracts NIW part of the chain
    # Then, get mu for given walker and cluster number
    # Shoud be a dm x nstep chain (dm is dimension of data)
    mu_chain = @as mu_chain map(x -> x[1], chain[walker_num,:]) begin
        @>> map(x -> x[cluster_num], mu_chain)
        @>> map(x -> get(x, "mu", 0), mu_chain)
        @> hcat(mu_chain...)
    end
    return mu_chain
end

export get_Sigma_chain
function get_Sigma_chain(chain, cluster_num; walker_num = 1)
    """
    For a given walker and cluster number, return
        the corresponding chain of covariance estimates in each dimension
    This is used to pass results back to R via JuliaCall
    """
    nw, nstep = size(chain)
    nm =  size(chain[1,1,1][1], 1)
    dm = size(get(chain[1,1,1][1][1], "Sigma", 0), 1)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"
    @assert nm >= cluster_num "cluster_num must be less than or equal to the number of clusters"

    # The first line extracts NIW part of the chain
    # Then, get mu for given walker and cluster number
    # Shoud be a dm x dm x nstep array of Sigma estimates (dm is dimension of data)
    Sigma_chain = @as Sigma_chain map(x -> x[1], chain[walker_num,:]) begin
        @>> map(x -> x[cluster_num], Sigma_chain)
        @>> map(x -> get(x, "Sigma", 0), Sigma_chain)
        @> hcat(Sigma_chain...)
        @> reshape(Sigma_chain, (dm, dm, nstep))
    end

    return Sigma_chain
end

export get_prop_chain
function get_prop_chain(chain; walker_num = 1)
    """
    For a given walker, return the corresponding
        chain of mixing weight estimates for each cluster
    This is used to pass results back to R via JuliaCall
    """
    nw, nstep = size(chain)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"

    # Shoud be a nm x nstep chain
    prop_chain = hcat(map(x -> x[2], chain[walker_num,:])...)

    return prop_chain
end

export get_z_chain
function get_z_chain(chain; walker_num = 1)
    """
    For a given walker, return the
        corresponding chain of z estimates for each cluster and each observation
    This is used to pass results back to R via JuliaCall
    """
    nw, nstep = size(chain)
    @assert nw >= walker_num "walker_num must be less than or equal to the number of walkers"

    # Returns an n x nstep chain of cluster labels for each observation across the whole chain
    z_chain = hcat(map(x -> x[3], chain[walker_num,:])...)

    return z_chain
end

export get_z_ests
function get_z_ests(z_chain)
    n = size(z_chain,1)
    rles = @> begin
                    z_chain
                    @>> mapslices(sort; dims = 2)
                    @>> mapslices(rle; dims = 2)
    end
    maxidx = map(x -> findmax(x[2])[2], rles)
    z_ests = @as z_ests map(x -> x[1], rles) begin
            @>> map( (x,y) -> x[y], z_ests, maxidx)
            @> reshape(z_ests, (n,))
    end
     z_ests
 end
