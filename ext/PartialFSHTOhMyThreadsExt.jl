"""
    PartialFSHTOhMyThreadsExt

Threaded apply over independent plans, or over clusters of one [`ClusterLeaf`](@ref).
Selected by `ComputationalBackends.ThreadedBackend`. Loaded by `using OhMyThreads`.

Direct, stretch, and cluster leaves do not call FastTransforms, so they do not need
the OpenMP-in-task bracket NUFSHT uses. A NUFSHT leaf's apply goes through NUFSHT,
which already brackets FastTransforms when farmed.
"""
module PartialFSHTOhMyThreadsExt

using PartialFSHT: PartialFSHT
using ComputationalBackends: ComputationalBackends
using LinearAlgebra: LinearAlgebra
using OhMyThreads: OhMyThreads

function PartialFSHT._mul_plan_threaded!(f, plan::PartialFSHT.Plan, alm,
                                         ::ComputationalBackends.AbstractThreadedBackend)
    return _threaded_synthesize!(f, alm, plan.leaf, plan)
end

function PartialFSHT._mul_adj_threaded!(alm, plan::PartialFSHT.Plan, f,
                                        ::ComputationalBackends.AbstractThreadedBackend)
    return _threaded_adjoint!(alm, f, plan.leaf, plan)
end

function _threaded_synthesize!(f, alm, leaf::PartialFSHT.MultiBoxLeaf, plan)
    B = plan.ntrans
    PartialFSHT._assert_field(f, length(leaf.θ), B)
    fill!(f, zero(eltype(f)))
    OhMyThreads.@tasks for i in eachindex(leaf.leaves)
        g = leaf.groups[i]
        sub = leaf.leaves[i]
        fg = leaf.fgs[i]
        fill!(fg, zero(eltype(fg)))
        PartialFSHT.synthesize_stretch!(fg, alm, sub)
        @inbounds if ndims(f) == 1
            for (j, k) in enumerate(g)
                f[k] = fg[j, 1]
            end
        else
            for b in 1:B, (j, k) in enumerate(g)
                f[k, b] = fg[j, b]
            end
        end
    end
    return f
end

_threaded_synthesize!(f, alm, leaf, plan) = PartialFSHT._synthesize!(f, alm, leaf, plan)

function _threaded_adjoint!(alm, f, leaf::PartialFSHT.MultiBoxLeaf, plan)
    B = plan.ntrans
    fill!(alm, zero(eltype(alm)))
    accs = leaf.accs
    sums = leaf.sums
    nacc = length(accs)
    for s in sums
        fill!(s, zero(eltype(s)))
    end
    # One accumulator slot per task, and each slot is owned by exactly one task: boxes are
    # chunked rather than round-robined, so no two tasks ever touch the same buffer.
    nbox = length(leaf.leaves)
    OhMyThreads.@tasks for t in 1:nacc
        acc = accs[t]
        sum_t = sums[t]
        for i in t:nacc:nbox
            g = leaf.groups[i]
            sub = leaf.leaves[i]
            fg = leaf.fgs[i]
            @inbounds if ndims(f) == 1
                for (j, k) in enumerate(g)
                    fg[j, 1] = f[k]
                end
            else
                for b in 1:B, (j, k) in enumerate(g)
                    fg[j, b] = f[k, b]
                end
            end
            PartialFSHT.adjoint_stretch!(acc, fg, sub)
            sum_t .+= acc
        end
    end
    for s in sums
        PartialFSHT._accumulate_coeffs!(alm, s, B)
    end
    return alm
end

_threaded_adjoint!(alm, f, leaf, plan) = PartialFSHT._adjoint_apply!(alm, f, leaf, plan)

function LinearAlgebra.mul!(fs::AbstractVector, plans::AbstractVector{<:PartialFSHT.Plan},
                            alms::AbstractVector, ::ComputationalBackends.AbstractThreadedBackend)
    length(fs) == length(plans) == length(alms) || throw(DimensionMismatch(
        "farmed mul! needs equal-length fs, plans, alms"))
    OhMyThreads.@tasks for i in eachindex(plans)
        LinearAlgebra.mul!(fs[i], plans[i], alms[i])
    end
    return fs
end

end
