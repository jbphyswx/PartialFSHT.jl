"""
Scaled associated-Legendre evaluation on iso-latitude rings.

Fully normalised ALFs `P̄_ℓ^m(cosθ)` with the Condon–Shortley sectorial sign satisfy

    Y_ℓm(θ,φ) = P̄_ℓ^m(cosθ) e^{imφ},      P̄_ℓ^{-m} = (-1)^m P̄_ℓ^m

and follow the fixed-order, increasing-degree recurrence

    P̄_m^m     = -√((2m+1)/(2m)) sinθ P̄_{m-1}^{m-1},      P̄_0^0 = 1/√(4π)
    P̄_{m+1}^m = √(2m+3) cosθ P̄_m^m
    P̄_ℓ^m     = a_ℓ^m cosθ P̄_{ℓ-1}^m - b_ℓ^m P̄_{ℓ-2}^m.

The sectorial seed carries `sin^m θ`, which flushes to zero in IEEE-754 long before it stops
mattering: on a 20° cap at `lmax = 4096` it vanishes at `m = 801` while orders out to
`m = 1636` still hold 67% of the outermost ring's Legendre energy. The recurrence therefore
runs in *block-scaled* form — one shared power-of-two exponent per ring chunk, renormalised on
a fixed stride — so the usable range is the exponent's rather than the mantissa's. Partial sums
are carried in the same scale, so the only terms dropped are ones already negligible against
the running maximum.

Polar optimisation (Reinecke & Seljebotn, A&A 554 A112, 2013) confines each order to the
contiguous ring range where it is not negligible, as a reduction in work rather than a mask on
the accumulation; see [`polar_mmax`](@ref).
"""

# The recurrence grows by at most ~2x per degree while evanescent, so 64 degrees is ~10^19 of
# growth — well inside the mantissa range, and cheap to amortise.
const _RESCALE_STRIDE = 64

# Binary places of headroom required between the largest and smallest ring of a chunk, so
# renormalising against the chunk maximum cannot flush the smallest ring to zero.
const _CHUNK_HEADROOM_BITS = 800

# Longest ring chunk, chosen for cache locality rather than correctness.
const _CHUNK_MAX = 256

"""
    polar_pad_factor() -> Float64

Leading constant `½·3^{2/3}` in the polar-optimisation transition width, isolated so the
calibration sweep can confirm it against `BigFloat` instead of assuming it.
"""
polar_pad_factor() = 0.5 * cbrt(9.0)

"""
    polar_mmax(θ, lmax, ε; pad_factor=polar_pad_factor()) -> Int

Largest order that must be retained on a ring at colatitude `θ` for a degree-`lmax` transform
at relative tolerance `ε`.

`P̄_ℓ^m(cosθ)` turns from oscillatory to evanescent at `m ≈ ν sinθ` with `ν = ℓ+½`. Past the
turning point the Bessel/Airy form gives `|P| ~ exp(-m(2δ)^{3/2}/3)` with `δ = m/(ν sinθ) - 1`,
so reaching a relative level `ε` needs

    m - ν sinθ  ≳  ½·3^{2/3} (ν sinθ)^{1/3} ln(1/ε)^{2/3}.

The transition width grows like `ℓ^{1/3} ln(1/ε)^{2/3}` — it is not a constant. At `ℓ = 4096`,
`θ = 0.394`, `ε = 10⁻⁸` it is 85 orders, so a fixed pad silently over-truncates at high degree.
"""
function polar_mmax(θ::Real, lmax::Integer, ε::Real; pad_factor::Real = polar_pad_factor())
    T = float(typeof(float(θ)))
    L = Int(lmax)
    L <= 0 && return 0
    s = abs(sin(T(θ)))
    z = (T(L) + T(0.5)) * s
    lnε = abs(log(max(T(ε), floatmin(T))))
    pad = T(pad_factor) * cbrt(max(z, one(T))) * cbrt(lnε * lnε) + 2
    return clamp(Int(ceil(z + pad)), 0, L)
end

"""
    RingLegendre{T,CT}

Ring geometry, polar-optimisation ranges, chunking, and apply scratch for the scaled ALF
recurrence. Built once per ring set; `θ` must be sorted ascending.

- `mlim[j]` — largest order retained on ring `j`
- `jlo[m+1]`, `jhi[m+1]` — contiguous ring range retained for order `m` (empty when `jhi < jlo`)
- `chunks` — ring ranges sharing one scale exponent

The retained ring set for an order is contiguous because `{θ : sinθ ≥ c}` is an interval,
which holds whether the rings lie north of the equator, south of it, or straddle it.
"""
struct RingLegendre{T<:AbstractFloat, CT<:Complex, VT<:AbstractVector{T}, MT<:AbstractMatrix{CT}}
    lmax::Int
    ntrans::Int
    ε::T
    θ::VT
    cosθ::VT
    sinθ::VT
    log2sinθ::VT
    log2c::VT
    mlim::Vector{Int}
    jlo::Vector{Int}
    jhi::Vector{Int}
    chunks::Vector{UnitRange{Int}}
    p1::VT
    p2::VT
    accp::MT
    accm::MT
end

function RingLegendre(θ::AbstractVector, lmax::Integer, ε::Real;
                      ntrans::Integer = 1, CT::Type = Complex{float(eltype(θ))})
    T = float(eltype(θ))
    L = Int(lmax)
    L >= 0 || throw(ArgumentError("lmax must be nonnegative"))
    B = Int(ntrans)
    B >= 1 || throw(ArgumentError("ntrans must be ≥ 1"))
    nθ = length(θ)
    nθ >= 1 || throw(ArgumentError("need at least one ring"))
    issorted(θ) || throw(ArgumentError("ring colatitudes must be sorted ascending"))

    cosθ = similar(θ, T)
    sinθ = similar(θ, T)
    log2sinθ = similar(θ, T)
    @inbounds for j in 1:nθ
        s, c = sincos(T(θ[j]))
        as = abs(s)
        sinθ[j] = as
        cosθ[j] = c
        log2sinθ[j] = as > 0 ? log2(as) : T(-Inf)
    end

    mlim = [polar_mmax(T(θ[j]), L, ε) for j in 1:nθ]

    jlo = fill(1, L + 1)
    jhi = fill(0, L + 1)
    @inbounds for m in 0:L
        lo, hi = 0, -1
        for j in 1:nθ
            if mlim[j] >= m
                lo == 0 && (lo = j)
                hi = j
            end
        end
        jlo[m + 1] = lo == 0 ? 1 : lo
        jhi[m + 1] = hi
    end

    # log2 of the sectorial prefactor c_m = (1/√(4π)) Π_{k=1}^m √((2k+1)/(2k)), accumulated in
    # log space so it cannot under/overflow. The (-1)^m sign is applied at use.
    log2c = similar(θ, T, L + 1)
    acc = -log2(sqrt(4 * T(π)))
    log2c[1] = acc
    @inbounds for m in 1:L
        acc += log2(sqrt(T(2m + 1) / T(2m)))
        log2c[m + 1] = acc
    end

    chunks = _ring_chunks(sinθ, mlim)
    cw = min(_CHUNK_MAX, maximum(length, chunks))

    p1 = similar(θ, T, cw)
    p2 = similar(θ, T, cw)
    accp = similar(θ, CT, cw, B)
    accm = similar(θ, CT, cw, B)

    return RingLegendre{T, CT, typeof(cosθ), typeof(accp)}(
        L, B, T(ε), θ, cosθ, sinθ, log2sinθ, log2c, mlim, jlo, jhi, chunks,
        p1, p2, accp, accm)
end

"""
Split rings into chunks sharing one scale exponent. A chunk may grow while
`m_max · |log2(sinθ_hi / sinθ_lo)| ≤ _CHUNK_HEADROOM_BITS`, which is exactly the condition for
renormalising against the chunk maximum not to flush its smallest ring to zero.
"""
function _ring_chunks(sinθ::AbstractVector{T}, mlim::Vector{Int}) where {T}
    n = length(sinθ)
    chunks = UnitRange{Int}[]
    i = 1
    @inbounds while i <= n
        stop = i
        mmax = mlim[i]
        while stop < n
            nxt = stop + 1
            nxt - i + 1 > _CHUNK_MAX && break
            slo, shi = sinθ[i], sinθ[nxt]
            span = (slo > 0 && shi > 0) ? abs(log2(shi / slo)) : T(Inf)
            mm = max(mmax, mlim[nxt])
            mm * span > _CHUNK_HEADROOM_BITS && break
            mmax = mm
            stop = nxt
        end
        push!(chunks, i:stop)
        i = stop + 1
    end
    return chunks
end

"""Number of retained `(ℓ, m, ring)` triples — the engine's work count, for the cost model."""
function ring_work(eng::RingLegendre)
    w = 0
    @inbounds for m in 0:eng.lmax
        nj = eng.jhi[m + 1] - eng.jlo[m + 1] + 1
        nj > 0 && (w += nj * (eng.lmax - m + 1))
    end
    return w
end

# ---------------------------------------------------------------------------------------
# Forward and adjoint share one recurrence walk, selected by a type parameter so neither
# pays a branch in the inner loop.
# ---------------------------------------------------------------------------------------

struct _Fwd end
struct _Adj end

"""
    rings_from_alm!(G, alm, eng) -> G

`G[j, m+lmax+1, b] = Σ_ℓ a_{ℓm} P̄_ℓ^m(cosθ_j)` over the retained `(m, j)` set, both signs of
`m`, all batch members. Entries outside the retained set are zero. Bang-first; allocation-free
after construction.
"""
function rings_from_alm!(G, alm, eng::RingLegendre)
    L = eng.lmax
    B = _ntrans(alm)
    B == eng.ntrans || throw(DimensionMismatch("alm carries $B transforms, engine built for $(eng.ntrans)"))
    _assert_coeffs(alm, L, B)
    size(G, 1) == length(eng.θ) ||
        throw(DimensionMismatch("G has $(size(G,1)) rings, expected $(length(eng.θ))"))
    size(G, 2) == 2L + 1 ||
        throw(DimensionMismatch("G has $(size(G,2)) orders, expected $(2L+1)"))
    fill!(G, zero(eltype(G)))
    _walk_rings!(_Fwd(), G, alm, eng)
    return G
end

"""
    alm_from_rings!(alm, G, eng) -> alm

Exact transpose of [`rings_from_alm!`](@ref):
`a_{ℓm} = Σ_j P̄_ℓ^m(cosθ_j) G[j, m+lmax+1, b]` over the same retained set, so the pair
satisfies the adjoint identity to round-off.
"""
function alm_from_rings!(alm, G, eng::RingLegendre)
    L = eng.lmax
    B = size(G, 3)
    B == eng.ntrans || throw(DimensionMismatch("G carries $B transforms, engine built for $(eng.ntrans)"))
    _assert_coeffs(alm, L, B)
    fill!(alm, zero(eltype(alm)))
    _walk_rings!(_Adj(), G, alm, eng)
    return alm
end

function _walk_rings!(dir, G, alm, eng::RingLegendre)
    @inbounds for m in 0:eng.lmax
        lo, hi = eng.jlo[m + 1], eng.jhi[m + 1]
        hi >= lo || continue
        for chunk in eng.chunks
            r = max(chunk.start, lo):min(chunk.stop, hi)
            isempty(r) && continue
            _ring_chunk!(dir, G, alm, eng, m, r)
        end
    end
    return nothing
end

function _ring_chunk!(dir, G, alm, eng::RingLegendre{T, CT}, m::Int,
                      r::UnitRange{Int}) where {T, CT}
    L = eng.lmax
    B = eng.ntrans
    off = L + 1
    nr = length(r)
    sgn = iseven(m) ? one(T) : -one(T)
    p1 = view(eng.p1, 1:nr)
    p2 = view(eng.p2, 1:nr)
    accp = view(eng.accp, 1:nr, 1:B)
    accm = view(eng.accm, 1:nr, 1:B)

    E = _seed_exponent(eng, m, r)
    @inbounds for (k, j) in enumerate(r)
        p2[k] = sgn * exp2(T(m) * eng.log2sinθ[j] + eng.log2c[m + 1] - E)
    end

    if dir isa _Fwd
        fill!(accp, zero(CT))
        fill!(accm, zero(CT))
    else
        @inbounds for b in 1:B, k in 1:nr
            j = r[k]
            accp[k, b] = G[j, m + off, b]
            accm[k, b] = m == 0 ? zero(CT) : G[j, -m + off, b]
        end
    end

    _touch!(dir, alm, accp, accm, p2, m, m, B, sgn, nr, E)

    if m < L
        @inbounds for (k, j) in enumerate(r)
            p1[k] = sqrt(T(2m + 3)) * eng.cosθ[j] * p2[k]
        end
        _touch!(dir, alm, accp, accm, p1, m + 1, m, B, sgn, nr, E)

        steps = 0
        for ℓ in (m + 2):L
            a = sqrt(T(2ℓ - 1) * T(2ℓ + 1) / (T(ℓ - m) * T(ℓ + m)))
            b = sqrt(T(2ℓ + 1) / T(2ℓ - 3) * T(ℓ - m - 1) / T(ℓ - m) *
                     T(ℓ + m - 1) / T(ℓ + m))
            @inbounds @simd for k in 1:nr
                p = a * eng.cosθ[r[k]] * p1[k] - b * p2[k]
                p2[k] = p1[k]
                p1[k] = p
            end
            _touch!(dir, alm, accp, accm, p1, ℓ, m, B, sgn, nr, E)
            steps += 1
            if steps == _RESCALE_STRIDE && ℓ < L
                steps = 0
                s = _renorm!(dir, p1, p2, accp, accm, nr, B)
                E += s
            end
        end
    end

    if dir isa _Fwd
        f = exp2(E)
        @inbounds for b in 1:B, k in 1:nr
            j = r[k]
            G[j, m + off, b] += accp[k, b] * f
            m > 0 && (G[j, -m + off, b] += accm[k, b] * f)
        end
    end
    return nothing
end

"""Shared seed exponent: the largest `log2|P̄_m^m|` in the chunk, so every mantissa starts ≤ 1."""
function _seed_exponent(eng::RingLegendre{T}, m::Int, r::UnitRange{Int}) where {T}
    e = T(-Inf)
    @inbounds for j in r
        e = max(e, T(m) * eng.log2sinθ[j] + eng.log2c[m + 1])
    end
    return isfinite(e) ? ceil(e) : zero(T)
end

# Forward: accumulate a_{ℓm} P̄ into the ring accumulators.
@inline function _touch!(::_Fwd, alm, accp, accm, P, ℓ::Int, m::Int, B::Int, sgn, nr::Int, E)
    @inbounds for b in 1:B
        ap = _coeff(alm, ℓ, m, b)
        am = m == 0 ? zero(ap) : _coeff(alm, ℓ, -m, b)
        (iszero(ap) & iszero(am)) && continue
        ams = am * sgn
        @simd for k in 1:nr
            accp[k, b] += ap * P[k]
            accm[k, b] += ams * P[k]
        end
    end
    return nothing
end

# Adjoint: contract P̄ against the ring data straight into the coefficient array. The running
# scale multiplies the output here rather than the input, which is what makes this the exact
# transpose of the forward walk.
@inline function _touch!(::_Adj, alm, accp, accm, P, ℓ::Int, m::Int, B::Int, sgn, nr::Int, E)
    f = exp2(E)
    @inbounds for b in 1:B
        sp = zero(eltype(alm))
        sm = zero(eltype(alm))
        @simd for k in 1:nr
            sp += P[k] * accp[k, b]
            sm += P[k] * accm[k, b]
        end
        _coeff_add!(alm, ℓ, m, b, sp * f)
        m > 0 && _coeff_add!(alm, ℓ, -m, b, sm * sgn * f)
    end
    return nothing
end

"""
Renormalise the two-term state against the chunk maximum, returning the binary places
absorbed into the shared exponent. The forward direction carries its partial sums in the same
scale, so they are rescaled too; the adjoint holds fixed input and needs no rescaling.
"""
function _renorm!(dir, p1, p2, accp, accm, nr::Int, B::Int)
    T = eltype(p1)
    mx = zero(T)
    @inbounds for k in 1:nr
        mx = max(mx, abs(p1[k]), abs(p2[k]))
    end
    (mx == 0 || !isfinite(mx)) && return zero(T)
    s = ceil(log2(mx))
    s == 0 && return zero(T)
    f = exp2(-s)
    @inbounds @simd for k in 1:nr
        p1[k] *= f
        p2[k] *= f
    end
    if dir isa _Fwd
        @inbounds for b in 1:B, k in 1:nr
            accp[k, b] *= f
            accm[k, b] *= f
        end
    end
    return s
end

# ---------------------------------------------------------------------------------------
# Single-value evaluation (oracle, seeding, tests)
# ---------------------------------------------------------------------------------------

"""
    alf(T, ℓ, m, θ) -> T

`P̄_ℓ^m(cosθ)` by the same block-scaled recurrence at one degree and order. Carried as
mantissa × 2^E throughout, so the sectorial `sin^m θ` never underflows and the value is
correct wherever the exponent range reaches — verified against `BigFloat` to `ℓ = 8192`.
"""
function alf(::Type{T}, ℓ::Integer, m::Integer, θ::Real) where {T<:AbstractFloat}
    l = Int(ℓ)
    am = abs(Int(m))
    am > l && return zero(T)
    s, x = sincos(T(θ))
    s = abs(s)
    log2c = -log2(sqrt(4 * T(π)))
    for k in 1:am
        log2c += log2(sqrt(T(2k + 1) / T(2k)))
    end
    e0 = am == 0 ? log2c : T(am) * (s > 0 ? log2(s) : T(-Inf)) + log2c
    isfinite(e0) || return zero(T)
    E = ceil(e0)
    p2 = (iseven(am) ? one(T) : -one(T)) * exp2(e0 - E)
    flip = (Int(m) < 0 && isodd(am)) ? -one(T) : one(T)
    l == am && return flip * _ldexp2(p2, E)
    p1 = sqrt(T(2am + 3)) * x * p2
    steps = 0
    for el in (am + 2):l
        a = sqrt(T(2el - 1) * T(2el + 1) / (T(el - am) * T(el + am)))
        b = sqrt(T(2el + 1) / T(2el - 3) * T(el - am - 1) / T(el - am) *
                 T(el + am - 1) / T(el + am))
        p = a * x * p1 - b * p2
        p2 = p1
        p1 = p
        steps += 1
        if steps == _RESCALE_STRIDE
            steps = 0
            mx = max(abs(p1), abs(p2))
            if mx > 0 && isfinite(mx)
                sh = ceil(log2(mx))
                if sh != 0
                    f = exp2(-sh)
                    p1 *= f
                    p2 *= f
                    E += sh
                end
            end
        end
    end
    return flip * _ldexp2(p1, E)
end

alf(ℓ::Integer, m::Integer, θ::Real) = alf(Float64, ℓ, m, θ)

# `E` is an integral-valued float; keep the scaling exact and avoid Int overflow at BigFloat.
@inline _ldexp2(v::T, E) where {T} = v * exp2(T(E))

"""
    ylm(ℓ, m, θ, φ) -> Complex

Complex orthonormal Condon–Shortley spherical harmonic `Y_ℓm(θ,φ)`.
"""
function ylm(ℓ::Integer, m::Integer, θ::Real, φ::Real)
    T = float(typeof(float(θ)))
    return Complex{T}(alf(T, ℓ, m, θ)) * cis(T(m) * T(φ))
end
