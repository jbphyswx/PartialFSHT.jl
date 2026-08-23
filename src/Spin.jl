# Spin-weighted transforms.
#
#     ₛY_{ℓm}(θ,φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ},      N_ℓ = √((2ℓ+1)/4π)
#
# The Wigner-d here is NOT a rotation. It is the spin analogue of an associated Legendre function:
# a three-term recurrence in *degree* at fixed (m, −s),
#
#     d^{ℓ+1}_{m'm} = (2ℓ+1)[ℓ(ℓ+1)cosβ − m'm] / (ℓ √((ℓ+1)²−m'²) √((ℓ+1)²−m²)) · d^ℓ_{m'm}
#                   − (ℓ+1)√(ℓ²−m'²)√(ℓ²−m²) / (ℓ √((ℓ+1)²−m'²) √((ℓ+1)²−m²)) · d^{ℓ−1}_{m'm}
#
# seeded at ℓ₀ = max(|m'|,|m|), where the d^{ℓ₀−1} coefficient vanishes identically, from the
# extremal closed form
#
#     d^ℓ_{m,ℓ}(β)  = √C(2ℓ, ℓ−m) cos^{ℓ+m}(β/2) sin^{ℓ−m}(β/2)
#     d^ℓ_{m,−ℓ}(β) = (−1)^{ℓ−m} √C(2ℓ, ℓ−m) cos^{ℓ−m}(β/2) sin^{ℓ+m}(β/2)
#
# and block-scaled exactly as `alf`, so sin^{ℓ+m}(β/2) never underflows. Both the log-binomial and
# the two exponents depend only on the order and the two logarithms only on the ring, so the whole
# seed is tabulated at construction (see `SpinRings`) — the closed form costs three `loggamma` calls
# and two `log`s, which is not something an apply should pay per (order, ring).

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

"""
Recurrence state and partial sums for one ring chunk of the spin walk; one per task slot.
"""
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

The extremal seed is `√C(2ℓ₀, ℓ₀−a) cos^{p_c}(θ/2) sin^{p_s}(θ/2)` with `ℓ₀ = max(|m|,|s|)`. Both
the log-binomial and the two exponents depend only on the order, and the two logarithms only on
the ring, so all of it is tabulated here: the apply then forms a seed with two multiply-adds
instead of the three `loggamma` calls and two `log`s the closed form needs.
"""
struct SpinRings{T<:AbstractFloat, CT<:Complex, VT<:AbstractVector{T}, VI<:AbstractVector{Int},
                 SC}
    lmax::Int
    spin::Int
    ntrans::Int
    ε::T
    θ::VT
    cosθ::VT
    log2cos_h::VT         # log2 cos(θ_j/2)
    log2sin_h::VT         # log2 sin(θ_j/2)
    Nℓ::VT                # √((2ℓ+1)/4π), ℓ = 0:lmax
    seed_lb2::VT          # ½ log2 C(2ℓ₀, ℓ₀−a) per order
    seed_pc::VI           # cos(θ/2) exponent per order
    seed_ps::VI           # sin(θ/2) exponent per order
    seed_sgn::VT
    mlim::VI              # follows the node array type, so a device kernel can read it
    jlo::Vector{Int}      # host-side loop bounds
    jhi::Vector{Int}
    chunks::Vector{UnitRange{Int}}
    slots::Vector{SC}
end

nslots(eng::SpinRings) = length(eng.slots)

function SpinRings(θ::AbstractVector, lmax::Integer, spin::Integer, ε::Real;
                   ntrans::Integer = 1, CT::Type = Complex{float(eltype(θ))},
                   slots::Integer = max(1, Threads.nthreads()),
                   reflect::Bool = false)
    T = float(eltype(θ))
    L = Int(lmax)
    s = Int(spin)
    B = Int(ntrans)
    B >= 1 || throw(ArgumentError("ntrans must be ≥ 1"))
    nθ = length(θ)
    nθ >= 1 || throw(ArgumentError("need at least one ring"))

    # Geometry is host work — plan-time and O(nθ + L) — then transferred once. The sortedness
    # check runs on the host copy for the same reason.
    θh = T.(_host(θ))
    issorted(θh) || throw(ArgumentError("ring colatitudes must be sorted ascending"))
    cosθ_h = similar(θh)
    log2cos_h = similar(θh)
    log2sin_h = similar(θh)
    # `reflect` means the physical colatitudes are `π .- θ` for the ascending `θ` given. Then
    # `cos θ → -cos θ`, and the two half-angle factors swap: `cos((π-θ)/2) = sin(θ/2)`. Nothing
    # else moves — `sin(π-θ) = sin θ` leaves the polar-optimisation limits alone — so the
    # reflection identity `ₛY_{ℓm}(π-θ,φ) = (-1)^{ℓ+m} ₋ₛY_{ℓm}(θ,φ)` is realised here at zero
    # cost, with the *original* spin weight and no pass over the coefficients.
    csgn = reflect ? -one(T) : one(T)
    @inbounds for j in 1:nθ
        t = θh[j]
        cosθ_h[j] = csgn * cos(t)
        ch, sh = cos(t / 2), abs(sin(t / 2))
        reflect && ((ch, sh) = (sh, ch))
        log2cos_h[j] = ch > 0 ? log2(ch) : T(-Inf)
        log2sin_h[j] = sh > 0 ? log2(sh) : T(-Inf)
    end

    mlim = Vector{Int}(undef, nθ)
    @inbounds for j in 1:nθ
        mlim[j] = max(polar_mmax(θh[j], L, ε), abs(s))
    end
    jlo, jhi = _nested_ranges(mlim, L, true)

    # Seed tables. ℓ₀ = max(|m|,|s|); whichever index is extremal moves into second place via
    # d^ℓ_{a,b} = (−1)^{a−b} d^ℓ_{b,a}, so the surviving free index `a` is m when |s| ≥ |m| and
    # −s otherwise. Both branches are a function of the order alone.
    n = -s
    inv_ln2 = one(T) / log(T(2))
    lb2 = Vector{T}(undef, 2L + 1)
    pc = Vector{Int}(undef, 2L + 1)
    ps = Vector{Int}(undef, 2L + 1)
    sg = Vector{T}(undef, 2L + 1)
    @inbounds for (idx, m) in enumerate(-L:L)
        l0 = max(abs(m), abs(n))
        if l0 > L
            lb2[idx] = T(-Inf); pc[idx] = 0; ps[idx] = 0; sg[idx] = one(T)
            continue
        end
        # (a, b): b is the extremal second index, a the free first index.
        if abs(n) == l0
            a, b, extra = m, (n == l0 ? l0 : -l0), one(T)
        else
            a, b, extra = n, (m == l0 ? l0 : -l0), (isodd(n - m) ? -one(T) : one(T))
        end
        if b == l0
            pc[idx], ps[idx], sgn = l0 + a, l0 - a, one(T)
        else
            pc[idx], ps[idx], sgn = l0 - a, l0 + a, (iseven(l0 - a) ? one(T) : -one(T))
        end
        lb2[idx] = _half_lbinom(T, l0, a) * inv_ln2
        sg[idx] = sgn * extra
    end

    chunks = _spin_ring_chunks(log2cos_h, log2sin_h, mlim, abs(s))
    cw = min(_CHUNK_MAX, maximum(length, chunks))

    Nℓ_h = Vector{T}(undef, L + 1)
    @inbounds for ℓ in 0:L
        Nℓ_h[ℓ + 1] = _Nℓ(T, ℓ)
    end

    cosθ = _to_like(θ, cosθ_h)
    VT = typeof(cosθ)
    MT = typeof(similar(θ, CT, cw, B))
    sl = [SpinScratch{T, CT, VT, MT}(similar(θ, T, cw), similar(θ, T, cw),
                                     similar(θ, CT, cw, B))
          for _ in 1:max(1, Int(slots))]
    mlim_dev = _to_like(θ, mlim)
    return SpinRings{T, CT, VT, typeof(mlim_dev), eltype(sl)}(
        L, s, B, T(ε), _to_like(θ, θh), cosθ, _to_like(θ, log2cos_h), _to_like(θ, log2sin_h),
        _to_like(θ, Nℓ_h), _to_like(θ, lb2), _to_like(θ, pc), _to_like(θ, ps), _to_like(θ, sg),
        mlim_dev, jlo, jhi, chunks, sl)
end

"""
Split spin rings into chunks sharing one scale exponent.

The seed's ring dependence is `p_c log₂cos(θ/2) + p_s log₂sin(θ/2)` with `p_c + p_s = 2ℓ₀`, so the
spread across a chunk is bounded by `2ℓ₀` times the larger of the two log ratios. Splitting on that
bound is what keeps renormalising against the chunk maximum from flushing the smallest ring to
zero — with a fixed block size instead, a ring whose true `d^ℓ_{m,−s}` is well inside tolerance
comes back as exactly `0.0` (measured at `lmax = 128`, `m = 96`).
"""
function _spin_ring_chunks(log2cos::AbstractVector{T}, log2sin::AbstractVector{T},
                           mlim::Vector{Int}, absspin::Int) where {T}
    n = length(log2cos)
    chunks = UnitRange{Int}[]
    i = 1
    @inbounds while i <= n
        stop = i
        l0max = max(mlim[i], absspin)
        while stop < n
            nxt = stop + 1
            nxt - i + 1 > _CHUNK_MAX && break
            spanc = abs(log2cos[nxt] - log2cos[i])
            spans = abs(log2sin[nxt] - log2sin[i])
            span = max(isfinite(spanc) ? spanc : T(Inf), isfinite(spans) ? spans : T(Inf))
            l0 = max(l0max, mlim[nxt], absspin)
            2 * l0 * span > _CHUNK_HEADROOM_BITS && break
            l0max = l0
            stop = nxt
        end
        push!(chunks, i:stop)
        i = stop + 1
    end
    return chunks
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

"""
Seed exponent `log₂|d^{ℓ₀}_{m,−s}(θ_j)|` from the tabulated log-binomial and the two tabulated
per-ring logarithms. The `p == 0` guards are what make the poles exact: at `θ = 0` the only
surviving order has `p_s = 0` *and* a vanishing log-binomial, so the seed is exactly one and the
rest are exactly zero, with no endpoint special case.
"""
@inline function _spin_seed_log2(eng::SpinRings{T}, j::Int, lb2::T, pc::Int, ps::Int) where {T}
    e = lb2
    @inbounds pc != 0 && (e += T(pc) * eng.log2cos_h[j])
    @inbounds ps != 0 && (e += T(ps) * eng.log2sin_h[j])
    return e
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

    mi = m + L + 1
    @inbounds lb2, pc, ps, sg = eng.seed_lb2[mi], eng.seed_pc[mi], eng.seed_ps[mi], eng.seed_sgn[mi]
    E = T(-Inf)
    @inbounds for j in r
        E = max(E, _spin_seed_log2(eng, j, lb2, pc, ps))
    end
    E = isfinite(E) ? ceil(E) : zero(T)
    @inbounds for (k, j) in enumerate(r)
        e = _spin_seed_log2(eng, j, lb2, pc, ps)
        d1[k] = isfinite(e) ? sg * exp2(e - E) : zero(T)
        d0[k] = zero(T)
    end

    if dir isa _Fwd
        fill!(acc, zero(CT))
    else
        @inbounds for b in 1:B, k in 1:nr
            acc[k, b] = G[r[k], m + off, b]
        end
    end

    _spin_touch!(dir, alm, acc, d1, l0, m, L, B, nr, E, eng)

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
        @inbounds N1 = eng.Nℓ[L1 + 1]
        # Single transform: fuse the recurrence with its accumulation so the new `d` is consumed in
        # the register that produced it. Batched transforms do NOT fuse — that would put the batch
        # loop innermost, where `acc[k, b]` strides by `nr`, and the strided inner loop costs more
        # than the fusion saves (measured 0.64x at `ntrans = 2` in the scalar pass, same shape).
        if dir isa _Fwd && B == 1
            @inbounds av = _spin_coeff(alm, L1, m, L, 1) * N1
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
        elseif dir isa _Adj && B == 1
            s = zero(CT)
            @inbounds @simd for k in 1:nr
                dn = (a1 * cs[k] - a2) * d1[k] - a0 * d0[k]
                d0[k] = d1[k]
                d1[k] = dn
                s += dn * acc[k, 1]
            end
            Lc = L1
            @inbounds _spin_coeff_add!(alm, L1, m, L, 1, s * (exp2(E) * N1))
        else
            @inbounds @simd for k in 1:nr
                dn = (a1 * cs[k] - a2) * d1[k] - a0 * d0[k]
                d0[k] = d1[k]
                d1[k] = dn
            end
            Lc = L1
            _spin_touch!(dir, alm, acc, d1, Lc, m, L, B, nr, E, eng)
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

@inline function _spin_touch!(::_Fwd, alm, acc, d, ℓ::Int, m::Int, L::Int, B::Int, nr::Int, E,
                              eng::SpinRings)
    abs(m) > ℓ && return nothing
    @inbounds N = eng.Nℓ[ℓ + 1]
    @inbounds for b in 1:B
        a = _spin_coeff(alm, ℓ, m, L, b) * N
        iszero(a) && continue
        @simd for k in 1:nr
            acc[k, b] += a * d[k]
        end
    end
    return nothing
end

@inline function _spin_touch!(::_Adj, alm, acc, d, ℓ::Int, m::Int, L::Int, B::Int, nr::Int, E,
                              eng::SpinRings)
    abs(m) > ℓ && return nothing
    @inbounds f = exp2(E) * eng.Nℓ[ℓ + 1]
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
struct SpinDirectLeaf{T, CT, EN, A3, VT, VI}
    lmax::Int
    ntrans::Int
    spin::Int
    eng::EN
    G::A3
    perm::VI
    θs::VT
    φs::VT
end

function SpinDirectLeaf(θ::AbstractVector, φ::AbstractVector, lmax::Integer,
                        ntrans::Integer, spin::Integer)
    T = float(promote_type(float(eltype(θ)), float(eltype(φ))))
    CT = Complex{T}
    L = Int(lmax)
    B = Int(ntrans)
    length(θ) == length(φ) || throw(DimensionMismatch("θ and φ must have the same length"))
    θh = _host(θ)
    φh = _host(φ)
    n = length(θh)
    perm = sortperm(θh)
    θs_h = Vector{T}(undef, n)
    φs_h = Vector{T}(undef, n)
    @inbounds for i in 1:n
        p = perm[i]
        θs_h[i] = T(θh[p])
        φs_h[i] = T(φh[p])
    end
    eng = SpinRings(θs_h, L, Int(spin), zero(T); ntrans = B, CT = CT)
    G = _zeros_like(θ, CT, n, 2L + 1, B)
    θs = _to_like(θ, θs_h)
    φs = _to_like(θ, φs_h)
    return SpinDirectLeaf{T, CT, typeof(eng), typeof(G), typeof(θs),
                          typeof(perm)}(L, B, Int(spin), eng, G, perm, θs, φs)
end

algorithm(::SpinDirectLeaf) = DirectAlgorithm()
ring_work(::SpinDirectLeaf) = 0

function synthesize_spin_direct!(f, alm, leaf::SpinDirectLeaf{T, CT}) where {T, CT}
    B = leaf.ntrans
    L = leaf.lmax
    _assert_coeffs(alm, L, B)
    _assert_field(f, length(leaf.θs), B)
    spin_rings_from_alm!(leaf.G, alm, leaf.eng)
    n = length(leaf.θs)
    off = L + 1
    fill!(f, zero(eltype(f)))
    @inbounds for b in 1:B, i in 1:n
        acc = leaf.G[i, off, b]
        φi = leaf.φs[i]
        for m in 1:L
            ph = cis(T(m) * φi)
            acc += leaf.G[i, m + off, b] * ph + leaf.G[i, -m + off, b] * conj(ph)
        end
        _field_set!(f, leaf.perm[i], b, acc)
    end
    return f
end

function adjoint_spin_direct!(alm, f, leaf::SpinDirectLeaf{T, CT}) where {T, CT}
    B = leaf.ntrans
    L = leaf.lmax
    _assert_coeffs(alm, L, B)
    _assert_field(f, length(leaf.θs), B)
    n = length(leaf.θs)
    off = L + 1
    fill!(leaf.G, zero(CT))
    @inbounds for b in 1:B, i in 1:n
        fi = _field_get(f, leaf.perm[i], b)
        iszero(fi) && continue
        φi = leaf.φs[i]
        leaf.G[i, off, b] = fi
        for m in 1:L
            ph = cis(T(m) * φi)
            leaf.G[i, m + off, b] = fi * conj(ph)
            leaf.G[i, -m + off, b] = fi * ph
        end
    end
    spin_alm_from_rings!(alm, leaf.G, leaf.eng)
    return alm
end
