# Scaled associated-Legendre evaluation on iso-latitude rings — the one ALF implementation in the
# package, used by the oracle as well with polar optimisation switched off.
#
#     Y_ℓm(θ,φ) = P̄_ℓ^m(cosθ) e^{imφ},      P̄_ℓ^{-m} = (-1)^m P̄_ℓ^m
#     P̄_m^m     = -√((2m+1)/(2m)) sinθ P̄_{m-1}^{m-1},      P̄_0^0 = 1/√(4π)
#     P̄_{m+1}^m = √(2m+3) cosθ P̄_m^m
#     P̄_ℓ^m     = a_ℓ^m cosθ P̄_{ℓ-1}^m - b_ℓ^m P̄_{ℓ-2}^m
#
# The recurrence runs block-scaled (mantissa × 2^E, one exponent per ring chunk) because the
# sectorial seed's sin^m θ underflows long before it stops mattering. Derivations are in the manual
# under Theory; the engineering constraints — chunk headroom, the nested retained ranges, the fused
# inner loop and the kernel shape that lost — are under Implementation notes.

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
    sectorial_log2c(T, lmax) -> Vector{T}

`log₂` of the sectorial prefactor `c_m = (1/√(4π)) Π_{k=1}^m √((2k+1)/(2k))`, accumulated in log
space so it can neither under- nor overflow. The `(-1)^m` Condon–Shortley sign is applied at use.
"""
function sectorial_log2c(::Type{T}, lmax::Integer) where {T}
    L = Int(lmax)
    v = Vector{T}(undef, L + 1)
    acc = -log2(sqrt(4 * T(π)))
    v[1] = acc
    @inbounds for m in 1:L
        acc += log2(sqrt(T(2m + 1) / T(2m)))
        v[m + 1] = acc
    end
    return v
end

"""
    _nested_ranges(mlim, lmax, symmetric) -> (jlo, jhi)

Contiguous retained ring range per order: `{j : mlim[j] ≥ |m|}`, indexed by `m ∈ 0:lmax` when
`symmetric` is `false` and by `m ∈ -lmax:lmax` when it is `true`.

`mlim` is unimodal in `j` — it tracks `sinθ`, which rises then falls over ascending colatitude —
so the retained sets are *nested* as the order grows, and all of them come out of one outward walk
from the peak in `O(nθ + lmax)`. Scanning every ring for every order instead is `O(lmax · nθ)`,
which at `lmax = 2048` is half a million comparisons inside a cost model that gets consulted
several times per plan.
"""
function _nested_ranges(mlim::Vector{Int}, lmax::Integer, symmetric::Bool)
    L = Int(lmax)
    n = length(mlim)
    n >= 1 || throw(ArgumentError("need at least one ring"))
    # Unimodality is what makes one outward walk sufficient; check it rather than assume it.
    peak = argmax(mlim)
    @inbounds for j in 2:peak
        mlim[j] >= mlim[j - 1] ||
            throw(ArgumentError("retained-order limits are not unimodal in colatitude " *
                                "(fell at ring $j, before the peak at $peak)"))
    end
    @inbounds for j in (peak + 1):n
        mlim[j] <= mlim[j - 1] ||
            throw(ArgumentError("retained-order limits are not unimodal in colatitude " *
                                "(rose at ring $j, after the peak at $peak)"))
    end

    nm = symmetric ? 2L + 1 : L + 1
    jlo = fill(1, nm)
    jhi = fill(0, nm)
    idx = (m) -> symmetric ? m + L + 1 : m + 1
    lo = hi = peak
    mtop = min(mlim[peak], L)
    @inbounds for m in mtop:-1:0
        while lo > 1 && mlim[lo - 1] >= m
            lo -= 1
        end
        while hi < n && mlim[hi + 1] >= m
            hi += 1
        end
        jlo[idx(m)] = lo
        jhi[idx(m)] = hi
        if symmetric && m > 0
            jlo[idx(-m)] = lo
            jhi[idx(-m)] = hi
        end
    end
    return jlo, jhi
end

"""
    retained_triples(mlim, lmax) -> Int

Retained `(ℓ, m, ring)` triples implied by per-ring order limits, in `O(nθ + lmax)` and without
allocating. Both the engines and the cost model count the same way, so a planner decision and the
work it predicts cannot drift apart.
"""
function retained_triples(mlim::Vector{Int}, lmax::Integer)
    L = Int(lmax)
    jlo, jhi = _nested_ranges(mlim, L, false)
    w = 0
    @inbounds for m in 0:L
        nj = jhi[m + 1] - jlo[m + 1] + 1
        nj > 0 && (w += nj * (L - m + 1))
    end
    return w
end

"""
Recurrence state and partial sums for one ring chunk, one per task slot.

The order loop is embarrassingly parallel — distinct orders touch distinct `G` columns going
forward and distinct coefficient entries coming back — so a per-slot scratch is the only thing
threading it requires.
"""
struct RingScratch{T<:AbstractFloat, CT<:Complex, VT<:AbstractVector{T}, MT<:AbstractMatrix{CT}}
    p1::VT
    p2::VT
    accp::MT
    accm::MT
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
struct RingLegendre{T<:AbstractFloat, CT<:Complex, VT<:AbstractVector{T},
                    VI<:AbstractVector{Int}, SC}
    lmax::Int
    ntrans::Int
    ε::T
    θ::VT
    cosθ::VT
    log2sinθ::VT
    log2c::VT
    mlim::VI              # follows the node array type, so a device kernel can read it
    mmax::Int
    jlo::Vector{Int}      # host-side loop bounds
    jhi::Vector{Int}
    chunks::Vector{UnitRange{Int}}
    slots::Vector{SC}
end

"""Number of independent task slots the ring pass can use."""
nslots(eng::RingLegendre) = length(eng.slots)

function RingLegendre(θ::AbstractVector, lmax::Integer, ε::Real;
                      ntrans::Integer = 1, CT::Type = Complex{float(eltype(θ))},
                      slots::Integer = max(1, Threads.nthreads()),
                      reflect::Bool = false)
    T = float(eltype(θ))
    L = Int(lmax)
    L >= 0 || throw(ArgumentError("lmax must be nonnegative"))
    B = Int(ntrans)
    B >= 1 || throw(ArgumentError("ntrans must be ≥ 1"))
    nθ = length(θ)
    nθ >= 1 || throw(ArgumentError("need at least one ring"))

    # Everything derived from the ring set is built on the host and transferred once. The
    # geometry is a plan-time O(nθ + L) computation, so doing it with scalar indexing is free —
    # and doing it *on* a device array would be scalar iteration on the GPU. That includes the
    # sortedness check, which is why it comes after the host copy.
    θh = T.(_host(θ))
    issorted(θh) || throw(ArgumentError("ring colatitudes must be sorted ascending"))
    cosθ_h = similar(θh)
    sinθ_h = similar(θh)
    log2sinθ_h = similar(θh)
    # `reflect` means the physical colatitudes are `π .- θ`, with `θ` the ascending array given.
    # Since `sin(π-θ) = sin θ`, the seed, the polar-optimisation limits and the chunking are all
    # unchanged and the entire reflection is one sign on `cosθ`: the recurrence
    # `P̄_ℓ^m = a cosθ P̄_{ℓ-1}^m - b P̄_{ℓ-2}^m` with `cosθ → -cosθ` reproduces `(-1)^{ℓ+m} P̄_ℓ^m`
    # by induction from a seed that does not involve `cosθ` at all. That is exactly the reflection
    # identity `Y_ℓm(π-θ,φ) = (-1)^{ℓ+m} Y_ℓm(θ,φ)`, so it needs no pass over the coefficients.
    csgn = reflect ? -one(T) : one(T)
    @inbounds for j in 1:nθ
        s, c = sincos(θh[j])
        as = abs(s)
        sinθ_h[j] = as
        cosθ_h[j] = csgn * c
        log2sinθ_h[j] = as > 0 ? log2(as) : T(-Inf)
    end
    cosθ = _to_like(θ, cosθ_h)
    log2sinθ = _to_like(θ, log2sinθ_h)

    mlim = Vector{Int}(undef, nθ)
    @inbounds for j in 1:nθ
        mlim[j] = polar_mmax(θh[j], L, ε)
    end
    jlo, jhi = _nested_ranges(mlim, L, false)

    log2c_h = sectorial_log2c(T, L)
    log2c = _to_like(θ, log2c_h)

    chunks = _ring_chunks(sinθ_h, mlim)
    cw = min(_CHUNK_MAX, maximum(length, chunks))

    ns = max(1, Int(slots))
    VT = typeof(cosθ)
    MT = typeof(similar(θ, CT, cw, B))
    sl = [RingScratch{T, CT, VT, MT}(similar(θ, T, cw), similar(θ, T, cw),
                                     similar(θ, CT, cw, B), similar(θ, CT, cw, B))
          for _ in 1:ns]
    mlim_dev = _to_like(θ, mlim)

    return RingLegendre{T, CT, VT, typeof(mlim_dev), eltype(sl)}(
        L, B, T(ε), _to_like(θ, θh), cosθ, log2sinθ, log2c, mlim_dev, maximum(mlim),
        jlo, jhi, chunks, sl)
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
function rings_from_alm!(G, alm, eng::RingLegendre,
                         backend = ComputationalBackends.SerialBackend())
    L = eng.lmax
    B = _ntrans(alm)
    B == eng.ntrans || throw(DimensionMismatch("alm carries $B transforms, engine built for $(eng.ntrans)"))
    _assert_coeffs(alm, L, B)
    size(G, 1) == length(eng.θ) ||
        throw(DimensionMismatch("G has $(size(G,1)) rings, expected $(length(eng.θ))"))
    size(G, 2) == 2L + 1 ||
        throw(DimensionMismatch("G has $(size(G,2)) orders, expected $(2L+1)"))
    fill!(G, zero(eltype(G)))
    _walk_rings!(_Fwd(), G, alm, eng, backend)
    return G
end

"""
    alm_from_rings!(alm, G, eng) -> alm

Exact transpose of [`rings_from_alm!`](@ref):
`a_{ℓm} = Σ_j P̄_ℓ^m(cosθ_j) G[j, m+lmax+1, b]` over the same retained set, so the pair
satisfies the adjoint identity to round-off.
"""
function alm_from_rings!(alm, G, eng::RingLegendre,
                         backend = ComputationalBackends.SerialBackend())
    L = eng.lmax
    B = size(G, 3)
    B == eng.ntrans || throw(DimensionMismatch("G carries $B transforms, engine built for $(eng.ntrans)"))
    _assert_coeffs(alm, L, B)
    fill!(alm, zero(eltype(alm)))
    _walk_rings!(_Adj(), G, alm, eng, backend)
    return alm
end

# Serial walk; the OhMyThreads extension overrides this for a threaded backend.
function _walk_rings!(dir, G, alm, eng::RingLegendre, backend)
    _walk_orders!(dir, G, alm, eng, 0:eng.lmax, eng.slots[1])
    return nothing
end

"""
Walk a subset of orders on one scratch slot. Orders are independent, so a caller may hand
disjoint `ms` to different tasks; interleaving (`t:nslots:lmax`) balances better than contiguous
blocks because the work at order `m` falls off like `lmax - m`.
"""
function _walk_orders!(dir, G, alm, eng::RingLegendre, ms, sc::RingScratch)
    @inbounds for m in ms
        lo, hi = eng.jlo[m + 1], eng.jhi[m + 1]
        hi >= lo || continue
        for chunk in eng.chunks
            r = max(chunk.start, lo):min(chunk.stop, hi)
            isempty(r) && continue
            _ring_chunk!(dir, G, alm, eng, m, r, sc)
        end
    end
    return nothing
end

function _ring_chunk!(dir, G, alm, eng::RingLegendre{T, CT}, m::Int,
                      r::UnitRange{Int}, sc::RingScratch) where {T, CT}
    L = eng.lmax
    B = eng.ntrans
    off = L + 1
    nr = length(r)
    sgn = iseven(m) ? one(T) : -one(T)
    p1 = view(sc.p1, 1:nr)
    p2 = view(sc.p2, 1:nr)
    accp = view(sc.accp, 1:nr, 1:B)
    accm = view(sc.accm, 1:nr, 1:B)

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

        cs = view(eng.cosθ, r)
        steps = 0
        for ℓ in (m + 2):L
            a = sqrt(T(2ℓ - 1) * T(2ℓ + 1) / (T(ℓ - m) * T(ℓ + m)))
            b = sqrt(T(2ℓ + 1) / T(2ℓ - 3) * T(ℓ - m - 1) / T(ℓ - m) *
                     T(ℓ + m - 1) / T(ℓ + m))
            # Single transform: fuse the recurrence with its accumulation so the new `P̄` value is
            # consumed in the register that produced it. Measured ~10% over the two-pass form; the
            # state stays L1-resident either way, so this is the whole win available from
            # restructuring — a register-resident kernel with the degree loop innermost measured
            # *2× worse*, because it caps SIMD at the unroll width while this loop vectorises
            # across the entire ring chunk.
            #
            # Batched transforms deliberately do NOT fuse. Fusing would put the batch loop
            # innermost, and `accp[k, bb]` strides by `nr` in `bb` — measured 0.64x at
            # `ntrans = 2` against the two-pass form, because the strided inner loop cannot
            # vectorise and the outer one no longer can either. The two-pass form below keeps every
            # accumulation contiguous in the ring index, which is the axis worth vectorising.
            if dir isa _Fwd && B == 1
                ap = _coeff(alm, ℓ, m, 1)
                ams = (m == 0 ? zero(ap) : _coeff(alm, ℓ, -m, 1)) * sgn
                @inbounds @simd for k in 1:nr
                    p = a * cs[k] * p1[k] - b * p2[k]
                    p2[k] = p1[k]
                    p1[k] = p
                    accp[k, 1] += ap * p
                    accm[k, 1] += ams * p
                end
            elseif dir isa _Adj && B == 1
                # `accp`/`accm` hold the ring data and are read-only here; the running scale
                # multiplies the output, which is what makes this the exact transpose.
                f = exp2(E)
                sp = zero(eltype(alm))
                sm = zero(eltype(alm))
                @inbounds @simd for k in 1:nr
                    p = a * cs[k] * p1[k] - b * p2[k]
                    p2[k] = p1[k]
                    p1[k] = p
                    sp += p * accp[k, 1]
                    sm += p * accm[k, 1]
                end
                _coeff_add!(alm, ℓ, m, 1, sp * f)
                m > 0 && _coeff_add!(alm, ℓ, -m, 1, sm * sgn * f)
            else
                @inbounds @simd for k in 1:nr
                    p = a * cs[k] * p1[k] - b * p2[k]
                    p2[k] = p1[k]
                    p1[k] = p
                end
                _touch!(dir, alm, accp, accm, p1, ℓ, m, B, sgn, nr, E)
            end
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

# Element type follows the argument, so a `BigFloat` colatitude is evaluated in `BigFloat` rather
# than silently rounded to Float64 — which matters, since this is the reference the fast paths and
# the `BigFloat` comparisons are gated against.
alf(ℓ::Integer, m::Integer, θ::Real) = alf(float(typeof(float(θ))), ℓ, m, θ)

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
