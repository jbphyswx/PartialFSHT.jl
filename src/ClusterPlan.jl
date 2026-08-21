"""
Clustered-cap leaf: partition the point set, one stretch (or direct) plan per cluster.
Synthesis is the disjoint union; the adjoint sums cluster adjoints into `alm`.
"""

struct ClusterLeaf{T, L, VT, FG, AB}
    groups::Vector{Vector{Int}}
    leaves::Vector{L}
    θ::VT
    φ::VT
    lmax::Int
    ntrans::Int
    fgs::FG
    almbufs::AB
end

function ClusterLeaf(θ::AbstractVector, φ::AbstractVector, lmax::Integer, ntrans::Integer,
                     tol::Real; nufft_backend = DirectSumBackend(),
                     target_θstar = nothing, k = nothing, spin::Integer = 0)
    T = float(eltype(θ))
    CT = Complex{T}
    groups = cluster_points(θ, φ; k = k, target_θstar = target_θstar)
    leaves = [StretchLeaf(view(θ, g), view(φ, g), lmax, ntrans, tol;
                          nufft_backend = nufft_backend, spin = Int(spin)) for g in groups]
    B = Int(ntrans)
    nθ, nφ = coeff_size(lmax)
    fgs = [_zeros_like(θ, CT, length(g), B) for g in groups]
    almbufs = [_zeros_like(θ, CT, nθ, nφ, B) for _ in groups]
    Lleaf = eltype(leaves)
    return ClusterLeaf{T, Lleaf, typeof(θ), typeof(fgs), typeof(almbufs)}(
        groups, leaves, θ, φ, Int(lmax), B, fgs, almbufs)
end

function synthesize_cluster!(f, alm, leaf::ClusterLeaf)
    B = leaf.ntrans
    _assert_field(f, length(leaf.θ), B)
    fill!(f, zero(eltype(f)))
    for (g, sub, fg) in zip(leaf.groups, leaf.leaves, leaf.fgs)
        fill!(fg, zero(eltype(fg)))
        synthesize_stretch!(fg, alm, sub)
        @inbounds if ndims(f) == 1
            for (j, i) in enumerate(g)
                f[i] = fg[j, 1]
            end
        else
            for b in 1:B, (j, i) in enumerate(g)
                f[i, b] = fg[j, b]
            end
        end
    end
    return f
end

function adjoint_cluster!(alm, f, leaf::ClusterLeaf)
    B = leaf.ntrans
    fill!(alm, zero(eltype(alm)))
    for (g, sub, fg, acc) in zip(leaf.groups, leaf.leaves, leaf.fgs, leaf.almbufs)
        @inbounds if ndims(f) == 1
            for (j, i) in enumerate(g)
                fg[j, 1] = f[i]
            end
        else
            for b in 1:B, (j, i) in enumerate(g)
                fg[j, b] = f[i, b]
            end
        end
        fill!(acc, zero(eltype(acc)))
        adjoint_stretch!(acc, fg, sub)
        alm .+= acc
    end
    return alm
end
