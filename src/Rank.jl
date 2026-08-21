"""
Rank-laboratory helpers for experiments A–D (dense SVD / polar-optimization ranks).
Not the production apply path.
"""

"""
    numerical_rank(s, ε) -> Int

Count singular values `> ε * s[1]` (relative).
"""
function numerical_rank(s::AbstractVector, ε::Real)
    isempty(s) && return 0
    s1 = s[1]
    s1 == 0 && return 0
    return count(>(ε * s1), s)
end

"""
    per_degree_ranks(θ, φ, lmax, ε) -> Vector{Int}

Numerical rank of the block `A^{(ℓ)}_{i,m} = Y_{ℓm}(Ω_i)` vs `ℓ`.
"""
function per_degree_ranks(θ::AbstractVector, φ::AbstractVector, lmax::Integer, ε::Real)
    θh, φh = _host(θ), _host(φ)
    n = length(θh)
    ranks = Vector{Int}(undef, Int(lmax) + 1)
    for ℓ in 0:Int(lmax)
        A = Matrix{Complex{float(eltype(θ))}}(undef, n, 2ℓ + 1)
        @inbounds for (k, m) in enumerate(-ℓ:ℓ), i in 1:n
            A[i, k] = ylm(ℓ, m, θh[i], φh[i])
        end
        s = LinearAlgebra.svdvals(A)
        ranks[ℓ + 1] = numerical_rank(s, ε)
    end
    return ranks
end

"""
    alf_block(θ, m, lmax) -> Matrix

Associated-Legendre block `P_{i,ℓ} = P̄_ℓ^m(cos θ_i)` for `ℓ = |m|:lmax`.
"""
function alf_block(θ::AbstractVector, m::Integer, lmax::Integer)
    θh = _host(θ)
    T = float(eltype(θ))
    am = abs(Int(m))
    n = length(θh)
    ncols = Int(lmax) - am + 1
    P = Matrix{T}(undef, n, max(ncols, 0))
    @inbounds for i in 1:n
        s, c = sin(T(θh[i])), cos(T(θh[i]))
        for (k, ℓ) in enumerate(am:Int(lmax))
            P[i, k] = _alf_value(T, ℓ, Int(m), c, s)
        end
    end
    return P
end

"""
    complementary_block_ranks(P, nbox, ε) -> Matrix{Int}

ID/RSVD ranks of dyadic `θ`–`ℓ` blocks of an associated-Legendre matrix `P`.
"""
function complementary_block_ranks(P::AbstractMatrix, nbox::Integer, ε::Real)
    n, p = size(P)
    nbox = max(Int(nbox), 1)
    R = zeros(Int, nbox, nbox)
    for bi in 1:nbox, bj in 1:nbox
        i1 = round(Int, (bi - 1) * n / nbox) + 1
        i2 = round(Int, bi * n / nbox)
        j1 = round(Int, (bj - 1) * p / nbox) + 1
        j2 = round(Int, bj * p / nbox)
        i2 >= i1 && j2 >= j1 || continue
        blk = P[i1:i2, j1:j2]
        s = LinearAlgebra.svdvals(blk)
        R[bi, bj] = numerical_rank(s, ε)
    end
    return R
end
