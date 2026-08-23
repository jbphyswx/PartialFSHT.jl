"""
Several boxes, one transform.

For a mask with genuinely disjoint components — two basins, a fragmented paleo-ocean, a set of
satellite swaths — each component gets its own stretched-interval leaf and the results are a
disjoint union in synthesis, a sum in the adjoint.

The accumulator count is bounded by the thread count rather than the box count. The previous
design held one full `(lmax+1)×(2lmax+1)×ntrans` coefficient buffer *per cluster*, which is
134 MB each at `lmax = 2048`; thirty-two of those is 4.3 GB of buffers for a transform whose
working set should be a few hundred megabytes.
"""

struct MultiBoxLeaf{T, LF, VT, A2, A3}
    groups::Vector{Vector{Int}}
    leaves::Vector{LF}
    fgs::Vector{A2}
    accs::Vector{A3}   # per-task scratch: `adjoint_stretch!` overwrites its output
    sums::Vector{A3}   # per-task running total, reduced once at the end
    θ::VT
    φ::VT
    lmax::Int
    ntrans::Int
end

function MultiBoxLeaf(θ::AbstractVector, φ::AbstractVector, lmax::Integer, ntrans::Integer,
                      tol::Real; nufft_backend = DirectSumBackend(), spin::Integer = 0,
                      groups = nothing, maxboxes::Integer = 8)
    T = float(eltype(θ))
    CT = Complex{T}
    L = Int(lmax)
    B = Int(ntrans)
    gs = groups === nothing ?
        partition_boxes(θ, φ, (b, np) -> cost_stretch(b, np, L, B, tol);
                        maxboxes = maxboxes) :
        [collect(g) for g in groups]
    length(gs) >= 1 || throw(ArgumentError("partition produced no boxes"))

    θh, φh = _host(θ), _host(φ)
    leaves = [StretchLeaf(view(θh, g), view(φh, g), L, B, tol;
                          nufft_backend = nufft_backend, spin = Int(spin)) for g in gs]
    fgs = [_zeros_like(θ, CT, length(g), B) for g in gs]
    nacc = min(length(gs), max(1, Threads.nthreads()))
    nθc, nφc = coeff_size(L)
    accs = [_zeros_like(θ, CT, nθc, nφc, B) for _ in 1:nacc]
    sums = [_zeros_like(θ, CT, nθc, nφc, B) for _ in 1:nacc]

    return MultiBoxLeaf{T, eltype(leaves), typeof(θ), eltype(fgs), eltype(accs)}(
        gs, leaves, fgs, accs, sums, θ, φ, L, B)
end

"""Number of independent accumulator slots, i.e. the adjoint's usable task count."""
naccumulators(leaf::MultiBoxLeaf) = length(leaf.accs)

algorithm(::MultiBoxLeaf) = MultiBoxAlgorithm()

ring_work(leaf::MultiBoxLeaf) = sum(ring_work, leaf.leaves)

"""
    boxes(leaf) -> Vector{SphereBox}
"""
boxes(leaf::MultiBoxLeaf) = [l.box for l in leaf.leaves]

function synthesize_multibox!(f, alm, leaf::MultiBoxLeaf)
    B = leaf.ntrans
    _assert_field(f, length(leaf.θ), B)
    fill!(f, zero(eltype(f)))
    @inbounds for i in eachindex(leaf.leaves)
        g = leaf.groups[i]
        fg = leaf.fgs[i]
        synthesize_stretch!(fg, alm, leaf.leaves[i])
        _scatter_field!(f, fg, g, B)
    end
    return f
end

function adjoint_multibox!(alm, f, leaf::MultiBoxLeaf)
    B = leaf.ntrans
    _assert_coeffs(alm, leaf.lmax, B)
    _assert_field(f, length(leaf.θ), B)
    fill!(alm, zero(eltype(alm)))
    acc = leaf.accs[1]
    @inbounds for i in eachindex(leaf.leaves)
        g = leaf.groups[i]
        fg = leaf.fgs[i]
        _gather_field!(fg, f, g, B)
        adjoint_stretch!(acc, fg, leaf.leaves[i])
        _accumulate_coeffs!(alm, acc, B)
    end
    return alm
end

@inline function _scatter_field!(f, fg, g::Vector{Int}, B::Int)
    if ndims(f) == 1
        @inbounds for (j, i) in enumerate(g)
            f[i] = fg[j, 1]
        end
    else
        @inbounds for b in 1:B, (j, i) in enumerate(g)
            f[i, b] = fg[j, b]
        end
    end
    return f
end

@inline function _gather_field!(fg, f, g::Vector{Int}, B::Int)
    if ndims(f) == 1
        @inbounds for (j, i) in enumerate(g)
            fg[j, 1] = f[i]
        end
    else
        @inbounds for b in 1:B, (j, i) in enumerate(g)
            fg[j, b] = f[i, b]
        end
    end
    return fg
end

@inline function _accumulate_coeffs!(alm, acc, B::Int)
    if ndims(alm) == 3
        alm .+= acc
    else
        @inbounds @views alm .+= acc[:, :, 1]
    end
    return alm
end
