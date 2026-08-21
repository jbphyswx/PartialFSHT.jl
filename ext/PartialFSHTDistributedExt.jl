"""
    PartialFSHTDistributedExt

Farm independent plans across `Distributed` workers. Plans are built on the worker
that owns the node set (they are not serialized). Loaded by `using Distributed`.
"""
module PartialFSHTDistributedExt

using PartialFSHT: PartialFSHT
using ComputationalBackends: ComputationalBackends
using Distributed: Distributed
using LinearAlgebra: LinearAlgebra

function _farm(work, n::Integer)
    if Distributed.nworkers() == 1
        return [work(i) for i in 1:n]
    else
        return Distributed.pmap(work, 1:n)
    end
end

function PartialFSHT._mul_plan_dist!(fs, plans, alms, ::ComputationalBackends.AbstractDistributedBackend)
    throw(ArgumentError(
        "DistributedBackend cannot apply a prebuilt Plan (plans hold FFT/NUFFT state that does not serialize). " *
        "Build the plan on the worker from the node set, or use SerialBackend / ThreadedBackend locally."))
end

function PartialFSHT._mul_adj_dist!(alm, plan, f, ::ComputationalBackends.AbstractDistributedBackend)
    throw(ArgumentError(
        "DistributedBackend cannot apply a prebuilt Plan. Build the plan on the worker from the node set."))
end

"""
    mul_nodeset(θs, φs, alms, lmax, DistributedBackend(); kwargs...) -> Vector

Synthesize each `(θs[i], φs[i], alms[i])` on a worker that builds its own plan.
"""
function PartialFSHT.mul_nodeset(θs, φs, alms, lmax,
                                 ::ComputationalBackends.AbstractDistributedBackend; kwargs...)
    length(θs) == length(φs) == length(alms) || throw(DimensionMismatch("θs, φs, alms lengths"))
    return _farm(length(θs)) do i
        plan = PartialFSHT.Plan(θs[i], φs[i]; lmax = lmax, kwargs...)
        return plan * alms[i]
    end
end

end
