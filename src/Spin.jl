"""
Spin-weighted transforms.

    ₛY_{ℓm}(θ,φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ},      N_ℓ = √((2ℓ+1)/4π)

The Wigner-`d` here is *not* a rotation — it is the spin analogue of an associated Legendre
function, a three-term recurrence in degree at fixed `(m, −s)`. Treating it as a rotation is
what made the previous spin path `O(L³)` per apply with a `cis` in the innermost loop and a
`(2L+1)²` scratch array (268 MB at `lmax = 2048`).

The degree recurrence is

    d^{ℓ+1}_{m'm} = (2ℓ+1)[ℓ(ℓ+1)cosβ − m'm] / (ℓ √((ℓ+1)²−m'²) √((ℓ+1)²−m²)) · d^ℓ_{m'm}
                  − (ℓ+1)√(ℓ²−m'²)√(ℓ²−m²) / (ℓ √((ℓ+1)²−m'²) √((ℓ+1)²−m²)) · d^{ℓ−1}_{m'm}

seeded at `ℓ₀ = max(|m'|,|m|)`, where the `d^{ℓ₀−1}` coefficient vanishes identically, from the
extremal closed form. Summing the Edmonds/Wikipedia series at the extremal index leaves a
single term and gives

    d^ℓ_{m,ℓ}(β)  = √C(2ℓ, ℓ−m) cos^{ℓ+m}(β/2) sin^{ℓ−m}(β/2)
    d^ℓ_{m,−ℓ}(β) = (−1)^{ℓ−m} √C(2ℓ, ℓ−m) cos^{ℓ−m}(β/2) sin^{ℓ+m}(β/2)

(checked against the closed `ℓ = 1` values). Both seeds are formed in log space and the
recurrence is block-scaled exactly as in [`alf`](@ref), so `sin^{ℓ+m}(β/2)` never underflows and
the result is correct at any degree the exponent range admits — the previous factorial sum was
silently wrong above `ℓ ≈ 40`.
"""

@inline _Nℓ(::Type{T}, ℓ::Integer) where {T} = sqrt(T(2 * Int(ℓ) + 1) / (4 * T(π)))

"""
Log of `√C(2ℓ, ℓ−a)` without forming the binomial.
"""
@inline function _half_lbinom(::Type{T}, ℓ::Int, a::Int) where {T}
    return (SpecialFunctions.loggamma(T(2ℓ + 1)) - SpecialFunctions.loggamma(T(ℓ - a + 1)) -
            SpecialFunctions.loggamma(T(ℓ + a + 1))) / 2
end

"""
    _d_extremal(T, ℓ, a, b, β) -> (log|d|, sign)

`d^ℓ_{a,b}(β)` at an extremal second index (`|b| = ℓ`), returned in log-magnitude/sign form so
the seed cannot underflow. `β ∈ [0, π]`, so `cos(β/2)` and `sin(β/2)` are both nonnegative and
the only sign is the `(−1)^{ℓ−a}` of the `b = −ℓ` branch.
"""
function _d_extremal(::Type{T}, ℓ::Int, a::Int, b::Int, β::T) where {T}
    c = cos(β / 2)
    s = sin(β / 2)
    lb = _half_lbinom(T, ℓ, a)
    if b == ℓ
        p_c, p_s, sgn = ℓ + a, ℓ - a, one(T)
    else
        p_c, p_s, sgn = ℓ - a, ℓ + a, (iseven(ℓ - a) ? one(T) : -one(T))
    end
    lc = p_c == 0 ? zero(T) : (c > 0 ? T(p_c) * log(c) : T(-Inf))
    ls = p_s == 0 ? zero(T) : (s > 0 ? T(p_s) * log(s) : T(-Inf))
    return lb + lc + ls, sgn
end

"""
    wigner_d(ℓ, m, n, β) -> Real

Wigner small-`d` element `d^ℓ_{mn}(β)` by the block-scaled degree recurrence. Correct at any
degree; `β` is taken in `[0, π]`.
"""
wigner_d(ℓ::Integer, m::Integer, n::Integer, β::Real) =
    wigner_d(float(typeof(float(β))), ℓ, m, n, β)

function wigner_d(::Type{T}, ℓ::Integer, m::Integer, n::Integer, β::Real) where {T}
    l = Int(ℓ)
    mm, nn = Int(m), Int(n)
    (abs(mm) > l || abs(nn) > l) && return zero(T)
    b = T(β)
    tiny = sqrt(eps(T))
    # Endpoints are exact and would otherwise take a log of zero.
    b <= tiny && return mm == nn ? one(T) : zero(T)
    b >= T(π) - tiny && return mm == -nn ? (iseven(l - nn) ? one(T) : -one(T)) : zero(T)

    l0 = max(abs(mm), abs(nn))
    # Seed with whichever index is extremal, using d^j_{a,b} = (-1)^{a-b} d^j_{b,a} to move an
    # extremal first index into second place.
    if abs(nn) == l0
        lg, sg = _d_extremal(T, l0, mm, nn == l0 ? l0 : -l0, b)
    else
        lg, sg = _d_extremal(T, l0, nn, mm == l0 ? l0 : -l0, b)
        isodd(nn - mm) && (sg = -sg)
    end
    isfinite(lg) || return zero(T)

    E = ceil(lg / log(T(2)))          # carry the seed as mantissa × 2^E
    d1 = sg * exp(lg - E * log(T(2)))
    l == l0 && return d1 * exp2(T(E))

    cb = cos(b)
    d0 = zero(T)   # at ℓ₀ = max(|m|,|n|) > 0 the ℓ₀-1 coefficient vanishes identically
    Lc = l0
    if l0 == 0
        # m = n = 0: the recurrence's 1/ℓ is singular at ℓ = 0, so step d⁰ = 1 → d¹ = cos β
        # explicitly and start recurring from ℓ = 1.
        d0 = d1
        d1 = d1 * cb
        Lc = 1
        l == 1 && return d1 * exp2(T(E))
    end
    steps = 0
    while Lc < l
        L1 = Lc + 1
        sm1 = sqrt(max(zero(T), T(L1 * L1 - mm * mm)))
        sn1 = sqrt(max(zero(T), T(L1 * L1 - nn * nn)))
        den = T(Lc) * sm1 * sn1
        if den <= 0
            d0, d1 = d1, zero(T)
            Lc = L1
            continue
        end
        sm0 = sqrt(max(zero(T), T(Lc * Lc - mm * mm)))
        sn0 = sqrt(max(zero(T), T(Lc * Lc - nn * nn)))
        c1 = T(2 * Lc + 1) * (T(Lc) * T(L1) * cb - T(mm) * T(nn)) / den
        c0 = T(L1) * sm0 * sn0 / den
        dn = c1 * d1 - c0 * d0
        d0, d1 = d1, dn
        Lc = L1
        steps += 1
        if steps == _RESCALE_STRIDE
            steps = 0
            mx = max(abs(d1), abs(d0))
            if mx > 0 && isfinite(mx)
                sh = ceil(log2(mx))
                if sh != 0
                    f = exp2(-sh)
                    d1 *= f
                    d0 *= f
                    E += sh
                end
            end
        end
    end
    return d1 * exp2(T(E))
end

"""
    sYlm(s, ℓ, m, θ, φ) -> Complex

Spin-weighted spherical harmonic `ₛY_{ℓm}(θ,φ)`.
"""
function sYlm(s::Integer, ℓ::Integer, m::Integer, θ::Real, φ::Real)
    T = float(typeof(float(θ)))
    abs(Int(s)) > Int(ℓ) && return zero(Complex{T})
    return Complex{T}(_Nℓ(T, ℓ) * wigner_d(T, ℓ, m, -Int(s), θ)) * cis(T(m) * T(φ))
end

# ---------------------------------------------------------------------------------------
# Spin ring engine: d^ℓ_{m,-s}(θ_j) contracted against spin coefficients
# ---------------------------------------------------------------------------------------

"""Recurrence state and partial sums for one ring chunk of the spin walk; one per task slot."""
struct SpinScratch{T<:AbstractFloat, CT<:Complex, VT<:AbstractVector{T}, MT<:AbstractMatrix{CT}}
    d1::VT
    d0::VT
    acc::MT
end

"""
    SpinRings{T,CT}

Ring-oriented spin transform: the same chunked, block-scaled walk as [`RingLegendre`](@ref)
with the degree recurrence for `d^ℓ_{m,−s}` in place of the associated-Legendre one. Polar
optimisation still applies, but never below `|m| = |s|`, since `d^ℓ_{m,−s}` is not small there.
"""
struct SpinRings{T<:AbstractFloat, CT<:Complex, VT<:AbstractVector{T}, MT<:AbstractMatrix{CT}}
    lmax::Int
    spin::Int
    ntrans::Int
    ε::T
    θ::VT
    cosθ::VT
    mlim::Vector{Int}
    jlo::Vector{Int}
    jhi::Vector{Int}
    chunks::Vector{UnitRange{Int}}
    slots::Vector{SpinScratch{T, CT, VT, MT}}
end

nslots(eng::SpinRings) = length(eng.slots)

function SpinRings(θ::AbstractVector, lmax::Integer, spin::Integer, ε::Real;
                   ntrans::Integer = 1, CT::Type = Complex{float(eltype(θ))},
                   slots::Integer = max(1, Threads.nthreads()))
    T = float(eltype(θ))
    L = Int(lmax)
    s = Int(spin)
    B = Int(ntrans)
    issorted(θ) || throw(ArgumentError("ring colatitudes must be sorted ascending"))
    nθ = length(θ)
    mlim = [max(polar_mmax(T(θ[j]), L, ε), abs(s)) for j in 1:nθ]
    jlo = fill(1, 2L + 1)
    jhi = fill(0, 2L + 1)
    @inbounds for (idx, m) in enumerate(-L:L)
        lo, hi = 0, -1
        for j in 1:nθ
            if mlim[j] >= abs(m)
                lo == 0 && (lo = j)
                hi = j
            end
        end
        jlo[idx] = lo == 0 ? 1 : lo
        jhi[idx] = hi
    end
    chunks = UnitRange{Int}[]
    i = 1
    while i <= nθ
        stop = min(i + _CHUNK_MAX - 1, nθ)
        push!(chunks, i:stop)
        i = stop + 1
    end
    cw = maximum(length, chunks)
    cosθ = similar(θ, T)
    @inbounds for j in 1:nθ
        cosθ[j] = cos(T(θ[j]))
    end
    VT = typeof(cosθ)
    MT = typeof(similar(θ, CT, cw, B))
    sl = [SpinScratch{T, CT, VT, MT}(similar(θ, T, cw), similar(θ, T, cw),
                                     similar(θ, CT, cw, B)) for _ in 1:max(1, Int(slots))]
    return SpinRings{T, CT, VT, MT}(
        L, s, B, T(ε), θ, cosθ, mlim, jlo, jhi, chunks, sl)
end

"""
    spin_rings_from_alm!(G, alm, eng) -> G

`G[j, m+lmax+1, b] = Σ_ℓ ₛa_{ℓm} N_ℓ d^ℓ_{m,−s}(θ_j)`.
"""
function spin_rings_from_alm!(G, alm, eng::SpinRings,
                              backend = ComputationalBackends.SerialBackend())
    fill!(G, zero(eltype(G)))
    _spin_walk!(_Fwd(), G, alm, eng, backend)
    return G
end

"""
    spin_alm_from_rings!(alm, G, eng) -> alm

Exact transpose of [`spin_rings_from_alm!`](@ref).
"""
function spin_alm_from_rings!(alm, G, eng::SpinRings,
                              backend = ComputationalBackends.SerialBackend())
    fill!(alm, zero(eltype(alm)))
    _spin_walk!(_Adj(), G, alm, eng, backend)
    return alm
end

function _spin_walk!(dir, G, alm, eng::SpinRings, backend)
    _spin_walk_orders!(dir, G, alm, eng, (-eng.lmax):eng.lmax, eng.slots[1])
    return nothing
end

"""Walk a subset of orders on one scratch slot; orders are independent."""
function _spin_walk_orders!(dir, G, alm, eng::SpinRings, ms, sc::SpinScratch)
    L = eng.lmax
    @inbounds for m in ms
        idx = m + L + 1
        lo, hi = eng.jlo[idx], eng.jhi[idx]
        hi >= lo || continue
        for chunk in eng.chunks
            r = max(chunk.start, lo):min(chunk.stop, hi)
            isempty(r) && continue
            _spin_chunk!(dir, G, alm, eng, m, r, sc)
        end
    end
    return nothing
end

function _spin_chunk!(dir, G, alm, eng::SpinRings{T, CT}, m::Int,
                      r::UnitRange{Int}, sc::SpinScratch) where {T, CT}
    L = eng.lmax
    B = eng.ntrans
    off = L + 1
    n = -eng.spin
    nr = length(r)
    l0 = max(abs(m), abs(n))
    l0 > L && return nothing
    d1 = view(sc.d1, 1:nr)
    d0 = view(sc.d0, 1:nr)
    acc = view(sc.acc, 1:nr, 1:B)

    # Shared seed exponent across the chunk. `_spin_seed_log` returns a natural log; the scale
    # is a power of two, so convert before rounding.
    ln2 = log(T(2))
    E = T(-Inf)
    @inbounds for j in r
        lg, _ = _spin_seed_log(T, l0, m, n, T(eng.θ[j]))
        E = max(E, lg)
    end
    E = isfinite(E) ? ceil(E / ln2) : zero(T)
    @inbounds for (k, j) in enumerate(r)
        lg, sg = _spin_seed_log(T, l0, m, n, T(eng.θ[j]))
        d1[k] = isfinite(lg) ? sg * exp(lg - E * ln2) : zero(T)
        d0[k] = zero(T)
    end

    if dir isa _Fwd
        fill!(acc, zero(CT))
    else
        @inbounds for b in 1:B, k in 1:nr
            acc[k, b] = G[r[k], m + off, b]
        end
    end

    _spin_touch!(dir, alm, acc, d1, l0, m, L, B, nr, E)

    steps = 0
    Lc = l0
    while Lc < L
        L1 = Lc + 1
        sm1 = sqrt(max(zero(T), T(L1 * L1 - m * m)))
        sn1 = sqrt(max(zero(T), T(L1 * L1 - n * n)))
        den = T(Lc) * sm1 * sn1
        if den <= 0
            @inbounds @simd for k in 1:nr
                d0[k] = d1[k]
                d1[k] = zero(T)
            end
            Lc = L1
            continue
        end
        sm0 = sqrt(max(zero(T), T(Lc * Lc - m * m)))
        sn0 = sqrt(max(zero(T), T(Lc * Lc - n * n)))
        a1 = T(2 * Lc + 1) * T(Lc) * T(L1) / den
        a2 = T(2 * Lc + 1) * T(m) * T(n) / den
        a0 = T(L1) * sm0 * sn0 / den
        cs = view(eng.cosθ, r)
        # Same fusion as the scalar ring pass: consume the new `d` in the register that produced
        # it. Worth ~1.25× there, and the recurrence here has the same shape.
        if dir isa _Fwd && B == 1 && abs(m) <= L1
            av = _spin_coeff(alm, L1, m, L, 1) * _Nℓ(T, L1)
            if iszero(av)
                @inbounds @simd for k in 1:nr
                    dn = (a1 * cs[k] - a2) * d1[k] - a0 * d0[k]
                    d0[k] = d1[k]
                    d1[k] = dn
                end
            else
                @inbounds @simd for k in 1:nr
                    dn = (a1 * cs[k] - a2) * d1[k] - a0 * d0[k]
                    d0[k] = d1[k]
                    d1[k] = dn
                    acc[k, 1] += av * dn
                end
            end
            Lc = L1
        elseif dir isa _Adj && B == 1 && abs(m) <= L1
            s = zero(eltype(alm))
            @inbounds @simd for k in 1:nr
                dn = (a1 * cs[k] - a2) * d1[k] - a0 * d0[k]
                d0[k] = d1[k]
                d1[k] = dn
                s += dn * acc[k, 1]
            end
            Lc = L1
            _spin_coeff_add!(alm, L1, m, L, 1, s * exp2(E) * _Nℓ(T, L1))
        else
            @inbounds @simd for k in 1:nr
                dn = (a1 * cs[k] - a2) * d1[k] - a0 * d0[k]
                d0[k] = d1[k]
                d1[k] = dn
            end
            Lc = L1
            _spin_touch!(dir, alm, acc, d1, Lc, m, L, B, nr, E)
        end
        steps += 1
        if steps == _RESCALE_STRIDE && Lc < L
            steps = 0
            mx = zero(T)
            @inbounds for k in 1:nr
                mx = max(mx, abs(d1[k]), abs(d0[k]))
            end
            if mx > 0 && isfinite(mx)
                sh = ceil(log2(mx))
                if sh != 0
                    f = exp2(-sh)
                    @inbounds @simd for k in 1:nr
                        d1[k] *= f
                        d0[k] *= f
                    end
                    if dir isa _Fwd
                        @inbounds for b in 1:B, k in 1:nr
                            acc[k, b] *= f
                        end
                    end
                    E += sh
                end
            end
        end
    end

    if dir isa _Fwd
        f = exp2(E)
        @inbounds for b in 1:B, k in 1:nr
            G[r[k], m + off, b] += acc[k, b] * f
        end
    end
    return nothing
end

function _spin_seed_log(::Type{T}, l0::Int, m::Int, n::Int, β::T) where {T}
    tiny = sqrt(eps(T))
    if β <= tiny
        return m == n ? (zero(T), one(T)) : (T(-Inf), one(T))
    elseif β >= T(π) - tiny
        return m == -n ? (zero(T), iseven(l0 - n) ? one(T) : -one(T)) : (T(-Inf), one(T))
    end
    if abs(n) == l0
        return _d_extremal(T, l0, m, n == l0 ? l0 : -l0, β)
    else
        lg, sg = _d_extremal(T, l0, n, m == l0 ? l0 : -l0, β)
        isodd(n - m) && (sg = -sg)
        return lg, sg
    end
end

@inline function _spin_touch!(::_Fwd, alm, acc, d, ℓ::Int, m::Int, L::Int, B::Int, nr::Int, E)
    abs(m) > ℓ && return nothing
    N = _Nℓ(real(eltype(acc)), ℓ)
    @inbounds for b in 1:B
        a = _spin_coeff(alm, ℓ, m, L, b) * N
        iszero(a) && continue
        @simd for k in 1:nr
            acc[k, b] += a * d[k]
        end
    end
    return nothing
end

@inline function _spin_touch!(::_Adj, alm, acc, d, ℓ::Int, m::Int, L::Int, B::Int, nr::Int, E)
    abs(m) > ℓ && return nothing
    N = _Nℓ(real(eltype(acc)), ℓ)
    f = exp2(E) * N
    @inbounds for b in 1:B
        s = zero(eltype(alm))
        @simd for k in 1:nr
            s += d[k] * acc[k, b]
        end
        _spin_coeff_add!(alm, ℓ, m, L, b, s * f)
    end
    return nothing
end

# ---------------------------------------------------------------------------------------
# Spin oracle
# ---------------------------------------------------------------------------------------

"""
    SpinDirectLeaf

Direct spin evaluation, `O(M L²)`: the correctness oracle for every spin fast path. Built on
[`SpinRings`](@ref) with polar optimisation off, so there is one Wigner-`d` implementation.
"""
struct SpinDirectLeaf{T, CT, EN, A3, VT, VC, VI}
    lmax::Int
    ntrans::Int
    spin::Int
    eng::EN
    G::A3
    perm::VI
    θs::VT
    eiφ::VC
end

function SpinDirectLeaf(θ::AbstractVector, φ::AbstractVector, lmax::Integer,
                        ntrans::Integer, spin::Integer)
    T = float(eltype(θ))
    CT = Complex{T}
    L = Int(lmax)
    B = Int(ntrans)
    θh = T.(_host(θ))
    φh = T.(_host(φ))
    perm = sortperm(θh)
    θs_h = θh[perm]
    eng = SpinRings(θs_h, L, Int(spin), zero(T); ntrans = B, CT = CT)
    n = length(θh)
    G = _zeros_like(θ, CT, n, 2L + 1, B)
    θs = _to_like(θ, θs_h)
    eiφ = _to_like(θ, CT[cis(φh[perm][i]) for i in 1:n])
    return SpinDirectLeaf{T, CT, typeof(eng), typeof(G), typeof(θs), typeof(eiφ),
                          typeof(perm)}(L, B, Int(spin), eng, G, perm, θs, eiφ)
end

algorithm(::SpinDirectLeaf) = DirectAlgorithm()
ring_work(::SpinDirectLeaf) = 0

function synthesize_spin_direct!(f, alm, leaf::SpinDirectLeaf{T, CT}) where {T, CT}
    B = leaf.ntrans
    L = leaf.lmax
    spin_rings_from_alm!(leaf.G, alm, leaf.eng)
    n = length(leaf.θs)
    off = L + 1
    fill!(f, zero(eltype(f)))
    @inbounds for b in 1:B, i in 1:n
        acc = leaf.G[i, off, b]
        e = leaf.eiφ[i]
        ph = one(CT)
        for m in 1:L
            ph *= e
            acc += leaf.G[i, m + off, b] * ph + leaf.G[i, -m + off, b] * conj(ph)
        end
        _field_set!(f, leaf.perm[i], b, acc)
    end
    return f
end

function adjoint_spin_direct!(alm, f, leaf::SpinDirectLeaf{T, CT}) where {T, CT}
    B = leaf.ntrans
    L = leaf.lmax
    n = length(leaf.θs)
    off = L + 1
    fill!(leaf.G, zero(CT))
    @inbounds for b in 1:B, i in 1:n
        fi = _field_get(f, leaf.perm[i], b)
        iszero(fi) && continue
        e = leaf.eiφ[i]
        leaf.G[i, off, b] = fi
        ph = one(CT)
        for m in 1:L
            ph *= e
            leaf.G[i, m + off, b] = fi * conj(ph)
            leaf.G[i, -m + off, b] = fi * ph
        end
    end
    spin_alm_from_rings!(alm, leaf.G, leaf.eng)
    return alm
end
