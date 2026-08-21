"""
    PartialFSHTMPIExt

MPI point decomposition of one transform: each rank holds a local `Plan` for its
subset of points and replicated coefficients. Synthesis is local; the adjoint is
`Allreduce`d. Loaded by `using MPI`.
"""
module PartialFSHTMPIExt

using PartialFSHT: PartialFSHT
using ComputationalBackends: ComputationalBackends
using LinearAlgebra: LinearAlgebra
using MPI: MPI

@inline _comm(backend::ComputationalBackends.AbstractMPIBackend) =
    something(backend.comm, MPI.COMM_WORLD)

function PartialFSHT._mul_plan_mpi!(f, plan::PartialFSHT.Plan, alm,
                                    ::ComputationalBackends.AbstractMPIBackend)
    return PartialFSHT._synthesize!(f, alm, plan.leaf, plan)
end

function PartialFSHT._mul_adj_mpi!(alm, plan::PartialFSHT.Plan, f,
                                   backend::ComputationalBackends.AbstractMPIBackend)
    PartialFSHT._adjoint_apply!(alm, f, plan.leaf, plan)
    MPI.Allreduce!(alm, +, _comm(backend))
    return alm
end

end
