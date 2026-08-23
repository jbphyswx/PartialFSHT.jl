"""
    PartialFSHTNUFSHTExt

Full-sphere leaf: `NUFSHTAlgorithm` calls NUFSHT's public plans. Scalar coefficients
are converted to FastTransforms packing; spin uses NUFSHT's dense `spin_coeff_index`
layout (the same as PartialFSHT Direct). Loaded by `using NUFSHT`.
"""
module PartialFSHTNUFSHTExt

using PartialFSHT: PartialFSHT
using NUFSHT: NUFSHT

struct NUFSHTLeaf{P, A}
    nplan::P
    cbuf::A
end

PartialFSHT.algorithm(::NUFSHTLeaf) = PartialFSHT.NUFSHTAlgorithm()
PartialFSHT.ring_work(::NUFSHTLeaf) = 0

function PartialFSHT._build_leaf(::PartialFSHT.NUFSHTAlgorithm, θ, φ, lmax, ntrans, spin,
                                 tol, nufft_b, opts)
    # Element type follows the node set, as it does on every other leaf. Pinning ComplexF64 here
    # would make a Float32 plan silently widen on this leaf alone.
    CT = Complex{float(eltype(θ))}
    if Int(spin) == 0
        nplan = NUFSHT.make_plan(CT, θ, φ, lmax; tol = tol, ntrans = ntrans)
        nθ, nφ = PartialFSHT.coeff_size(lmax)
        cbuf = ntrans == 1 ?
            PartialFSHT._zeros_like(θ, CT, nθ, nφ) :
            PartialFSHT._zeros_like(θ, CT, nθ, nφ, ntrans)
        return NUFSHTLeaf(nplan, cbuf)
    end
    nplan = NUFSHT.make_spin_plan(CT, θ, φ, lmax, Int(spin); tol = tol, ntrans = ntrans)
    return NUFSHTLeaf(nplan, nothing)
end

function PartialFSHT._synthesize!(f, alm, leaf::NUFSHTLeaf, plan::PartialFSHT.Plan)
    if plan.spin == 0
        PartialFSHT.complex_to_fasttransforms!(leaf.cbuf, alm, plan.lmax)
        NUFSHT.nusht_type2!(f, leaf.cbuf, leaf.nplan)
    else
        NUFSHT.nusht_type2_spin!(f, alm, leaf.nplan)
    end
    return f
end

function PartialFSHT._adjoint_apply!(alm, f, leaf::NUFSHTLeaf, plan::PartialFSHT.Plan)
    if plan.spin == 0
        fill!(leaf.cbuf, 0)
        NUFSHT._nusht_true_adjoint!(leaf.cbuf, f, leaf.nplan)
        PartialFSHT.fasttransforms_to_complex!(alm, leaf.cbuf, plan.lmax)
    else
        NUFSHT.nusht_type1_spin!(alm, f, leaf.nplan)
    end
    return alm
end

end
