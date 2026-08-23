"""
Stretched-interval leaf: the workhorse partial-sky transform.

Evaluate the degree-`lmax` field on the few rings that intersect the data's colatitude
interval, apodise, periodise, and interpolate to the sample points with one small NUFFT. This
is Carron & Reinecke's construction (arXiv:2603.17166) with the coefficient rotation removed —
they never rotate, they *assume* a frame with the patch at a pole, and a per-apply Wigner
rotation costs eight times the ring work it saves.

    alm ──ring ALF (polar-optimised)──▶ G[ring, m]
        ──apodise in θ──▶ periodise (DFS at a pole, two-sided otherwise)
        ──2-D FFT──▶ reduced Fourier torus ──type-2 NUFFT──▶ f(Ωᵢ)

The adjoint is the same pipeline transposed, step by step, so `⟨𝒴a, f⟩ = ⟨a, 𝒴*f⟩` holds to
round-off rather than to the tolerance of a quadrature.

Longitude is reduced by polar optimisation (`nφ = 2 m_max + 1`), which is the whole reduction
when the interval touches a pole and a partial one otherwise. Stretching the longitude *arc*
as well is a further saving for mid-latitude patches; it needs a zoom/chirp-z per ring and is
gated on the calibration measurement rather than assumed.
"""

"""
    next_smooth_even(n) -> Int

Smallest even `5`-smooth integer `≥ n`. The DFS reflection shifts longitude by `Nφ÷2` grid
cells to realise `φ → φ+π`, which is only a half-turn when `Nφ` is even. An odd grid makes it a
shift of `π(1 - 1/Nφ)` instead, and the resulting seam caps accuracy at ~1e-5 regardless of the
requested tolerance — `Nφ = 135` shifts by 3.118 rad rather than 3.1416.
"""
next_smooth_even(n::Integer) = 2 * next_smooth(cld(Int(n), 2))

"""
    next_smooth(n) -> Int

Smallest `5`-smooth integer `≥ n`. Grid sizes are free to grow — oversampling only improves the
trigonometric interpolation — and every size the previous planner chose had a large prime
factor (`nφ = 17·19, 3·181, 5·191, 5·349, 3·1091` at `lmax = 256…4096`), which puts FFTW on a
Rader/Bluestein path for nothing.
"""
function next_smooth(n::Integer)
    n <= 1 && return 1
    best = typemax(Int)
    p2 = 1
    while p2 < 2n + 2
        p3 = p2
        while p3 < 2n + 2
            p5 = p3
            while p5 < 2n + 2
                p5 >= n && (best = min(best, p5))
                p5 > typemax(Int) ÷ 5 && break
                p5 *= 5
            end
            p3 > typemax(Int) ÷ 3 && break
            p3 *= 3
        end
        p2 > typemax(Int) ÷ 2 && break
        p2 *= 2
    end
    return best == typemax(Int) ? Int(n) : best
end

# Spare colatitude modes above the derived bandlimit `K`. The derivation puts the ES window's
# spectral support at `≲ β/σ`, which is leading order, so a couple of modes of slack costs
# almost nothing against a ring count of tens to hundreds and covers the tail.
const _K_MARGIN = 2

"""
    StretchLeaf

One box, one stretched-interval transform. Buffers and plans are built once; an apply
allocates nothing.
"""
struct StretchLeaf{T, CT, VT, VC, PT, A3, A2, EN, RB, RF, FP, IP, N2, N1, NB}
    box::SphereBox{T}
    geomθ::StretchGeometry{T}
    lmax::Int
    ntrans::Int
    spin::Int
    tol::T
    flip_south::Bool
    nθ_half::Int
    nθ::Int
    nφ::Int
    mmax::Int
    eng::EN
    θ_rings::VT
    Θ_rings::VT
    wθ::VT
    Θ_pts::VT
    φ_pts::VT
    parity::PT              # (-1)^{ℓ+m} coefficient-shaped signs, empty unless flip_south
    almwork::A3
    G::A3
    Fφ::A3
    F::A3
    F̃::A3
    Fhat::A3
    fbuf::A2
    phase::VC
    phase_conj::VC
    ring_bfft::RB
    ring_fft::RF
    fft_plan::FP
    ifft_plan::IP
    nufft2::N2
    nufft1::N1
    nufft_backend::NB
end

function StretchLeaf(θ::AbstractVector, φ::AbstractVector, lmax::Integer, ntrans::Integer,
                     tol::Real; nufft_backend = DirectSumBackend(), spin::Integer = 0,
                     box = nothing, εapo_max::Real = _EPS_APO_MAX,
                     oversample_θ::Real = 1, oversample_φ::Real = 1)
    T = float(eltype(θ))
    CT = Complex{T}
    L = Int(lmax)
    B = Int(ntrans)
    bx = box === nothing ? bounding_box(θ, φ) : box

    # A south-pole periodisation maps increasing Θ to *decreasing* θ. The ring engine walks
    # ascending θ, and the grid's θ-FFT needs rows in ascending Θ; those two orders are then
    # reversed relative to each other, which silently transposes the grid and makes the result
    # O(1) wrong. So whenever the cheapest periodisation is the southern one, reflect the whole
    # problem to the north instead: Y_ℓm(π-θ,φ) = (-1)^{ℓ+m} Y_ℓm(θ,φ), so one sign pass on the
    # coefficients replaces all of the descending-order bookkeeping.
    #
    # The decision has to come from the *chosen* geometry rather than from whether the box
    # literally touches θ = π: a patch at θ ≈ π - 0.3 touches nothing yet is still cheapest as a
    # south-pole periodisation.
    tiny = sqrt(eps(T))
    geom0 = stretch_geometry(bx.θlo, bx.θhi, L, tol; εapo_max = εapo_max)
    flip = geom0.kind === :pole && geom0.θhi >= T(π) - tiny && geom0.θlo > tiny
    θw = flip ? [T(π) - T(t) for t in _host(θ)] : T.(_host(θ))
    bxw = flip ? bounding_box(θw, _host(φ)) : bx
    geomθ = flip ? stretch_geometry(bxw.θlo, bxw.θhi, L, tol; εapo_max = εapo_max) : geom0

    # A pole/full periodisation is DFS-doubled, so the full grid must be exactly twice the
    # half-period ring count: round the *half* to 5-smooth and double it (2·smooth is smooth).
    # Rounding the doubled size instead can land on an odd 5-smooth value — 125 for 122 — and
    # then the doubling no longer matches the buffer.
    Kθ = Int(ceil((geomθ.K + _K_MARGIN) * oversample_θ))
    if geomθ.kind === :interior
        nθ = next_smooth(2 * Kθ + 1)
        nθ_half = nθ
    else
        nθ_half = next_smooth(Kθ + 1)
        nθ = 2 * nθ_half
    end

    θ_rings_h, Θ_rings_h = stretch_rings(geomθ, nθ_half)
    # Ring order must be ascending in both θ (the engine's requirement) and Θ (the grid's), and
    # after the northward reflection above it is. Assert rather than sort: a silent reorder here
    # is exactly what reversed the grid.
    issorted(θ_rings_h) || throw(AssertionError(
        "ring colatitudes are not ascending for a $(geomθ.kind) periodisation on " *
        "θ ∈ [$(geomθ.θlo), $(geomθ.θhi)] — the grid and the ring engine would disagree"))

    # A south-touching interval is reflected to the north. For spin the reflection also flips
    # the spin weight, since d^ℓ_{m,-s}(π-θ) = (-1)^{ℓ+m} d^ℓ_{m,s}(θ), i.e.
    # ₛY_{ℓm}(π-θ,φ) = (-1)^{ℓ+m} ₋ₛY_{ℓm}(θ,φ) — the same coefficient sign pass, opposite spin.
    spin_eff = flip ? -Int(spin) : Int(spin)
    eng = spin_eff == 0 ? RingLegendre(θ_rings_h, L, tol; ntrans = B, CT = CT) :
                          SpinRings(θ_rings_h, L, spin_eff, tol; ntrans = B, CT = CT)
    mmax = maximum(eng.mlim)
    nφ = next_smooth_even(Int(ceil((2 * max(mmax, abs(Int(spin))) + 1) * oversample_φ)))

    wθ_h = similar(Θ_rings_h)
    stretch_window!(wθ_h, geomθ, Θ_rings_h)

    θ_rings = _to_like(θ, θ_rings_h)
    Θ_rings = _to_like(θ, Θ_rings_h)
    wθ = _to_like(θ, wθ_h)

    n = length(θ)
    Θ_pts_h = T[theta_to_stretch(geomθ, t) for t in θw]
    φ_pts_h = T[mod(T(p), 2 * T(π)) for p in _host(φ)]
    Θ_pts = _to_like(θ, Θ_pts_h)
    φ_pts = _to_like(θ, φ_pts_h)

    parity = _to_like(θ, flip ? _parity_signs(T, L) : zeros(T, 0, 0))

    nθc, nφc = coeff_size(L)
    almwork = _zeros_like(θ, CT, nθc, nφc, B)
    G = _zeros_like(θ, CT, nθ_half, 2L + 1, B)
    Fφ = _zeros_like(θ, CT, nθ_half, nφ, B)
    F = _zeros_like(θ, CT, nθ_half, nφ, B)
    F̃ = _zeros_like(θ, CT, nθ, nφ, B)
    Fhat = _zeros_like(θ, CT, nθ, nφ, B)
    fbuf = _zeros_like(θ, CT, n, B)

    ring_bfft = AbstractFFTs.plan_bfft(Fφ, 2)
    ring_fft = AbstractFFTs.plan_fft(Fφ, 2)
    fft_plan = AbstractFFTs.plan_fft(F̃, (1, 2))
    ifft_plan = AbstractFFTs.plan_bfft(Fhat, (1, 2))

    # Half-cell offset in Θ only where the ring grid is a midpoint grid (pole / full sphere);
    # the interior grid starts at Θ = 0 and needs no shift.
    shift = geomθ.kind === :interior ? zero(T) : T(π) / T(nθ)
    N = T(nθ * nφ)
    phase_h = CT[cis(-T(_mode_k(i, nθ, 1)) * shift) / N for i in 1:nθ]
    phase = _to_like(θ, phase_h)
    phase_conj = _to_like(θ, conj.(phase_h))

    nufft2 = _nufft_makeplan(nufft_backend, Θ_pts, φ_pts, 2, (nθ, nφ), +1, B, float(tol);
                             dtype = T, modeord = 1)
    nufft1 = _nufft_makeplan(nufft_backend, Θ_pts, φ_pts, 1, (nθ, nφ), -1, B, float(tol);
                             dtype = T, modeord = 1)

    return StretchLeaf{T, CT, typeof(θ_rings), typeof(phase), typeof(parity), typeof(G),
                       typeof(fbuf), typeof(eng), typeof(ring_bfft), typeof(ring_fft),
                       typeof(fft_plan), typeof(ifft_plan), typeof(nufft2), typeof(nufft1),
                       typeof(nufft_backend)}(
        bx, geomθ, L, B, Int(spin), T(tol), flip, nθ_half, nθ, nφ, mmax, eng,
        θ_rings, Θ_rings, wθ, Θ_pts, φ_pts, parity, almwork, G, Fφ, F, F̃, Fhat, fbuf,
        phase, phase_conj, ring_bfft, ring_fft, fft_plan, ifft_plan, nufft2, nufft1,
        nufft_backend)
end

"""`(-1)^{ℓ+m}` laid out in the coefficient array, for the north/south reflection."""
function _parity_signs(::Type{T}, lmax::Integer) where {T}
    nθc, nφc = coeff_size(lmax)
    s = zeros(T, nθc, nφc)
    @inbounds for ℓ in 0:Int(lmax), m in -ℓ:ℓ
        s[sph_mode(ℓ, m)] = iseven(ℓ + m) ? one(T) : -one(T)
    end
    return s
end

algorithm(::StretchLeaf) = StretchAlgorithm()

"""
    ring_work(leaf) -> Int

Retained `(ℓ, m, ring)` triples in this leaf's ALF pass — the counted work the cost model uses,
so complexity claims are gated by counting rather than timing.
"""
ring_work(leaf::StretchLeaf) = ring_work(leaf.eng)

# ---------------------------------------------------------------------------------------

function synthesize_stretch!(f, alm, leaf::StretchLeaf)
    B = leaf.ntrans
    _assert_coeffs(alm, leaf.lmax, B)
    _assert_field(f, length(leaf.Θ_pts), B)

    src = alm
    if leaf.flip_south
        _apply_parity!(leaf.almwork, alm, leaf.parity, B)
        src = leaf.almwork
    end

    _fill_rings!(leaf.G, src, leaf.eng)
    _rings_to_grid!(leaf.F, leaf.G, leaf.lmax, leaf.ring_bfft, leaf.Fφ)
    leaf.F .*= leaf.wθ
    _periodize!(leaf.F̃, leaf.F, leaf.geomθ, leaf.spin)
    LinearAlgebra.mul!(leaf.Fhat, leaf.fft_plan, leaf.F̃)
    leaf.Fhat .*= leaf.phase
    _nufft_exec!(leaf.nufft2, leaf.Fhat, leaf.fbuf)
    _copy_field_out!(f, leaf.fbuf, B)
    return f
end

function adjoint_stretch!(alm, f, leaf::StretchLeaf)
    B = leaf.ntrans
    _assert_coeffs(alm, leaf.lmax, B)
    _assert_field(f, length(leaf.Θ_pts), B)

    _copy_field_in!(leaf.fbuf, f, B)
    _nufft_exec!(leaf.nufft1, leaf.fbuf, leaf.Fhat)
    leaf.Fhat .*= leaf.phase_conj
    LinearAlgebra.mul!(leaf.F̃, leaf.ifft_plan, leaf.Fhat)
    _periodize_adjoint!(leaf.F, leaf.F̃, leaf.geomθ, leaf.spin)
    leaf.F .*= leaf.wθ
    _grid_to_rings!(leaf.G, leaf.F, leaf.lmax, leaf.ring_fft, leaf.Fφ)

    dst = leaf.flip_south ? leaf.almwork : alm
    _drain_rings!(dst, leaf.G, leaf.eng)
    if leaf.flip_south
        _apply_parity!(alm, dst, leaf.parity, B)
    end
    return alm
end

# One ring engine per spin weight; the scalar and spin recurrences differ, the surrounding
# pipeline does not.
_fill_rings!(G, alm, eng::RingLegendre) = rings_from_alm!(G, alm, eng)
_fill_rings!(G, alm, eng::SpinRings) = spin_rings_from_alm!(G, alm, eng)
_drain_rings!(alm, G, eng::RingLegendre) = alm_from_rings!(alm, G, eng)
_drain_rings!(alm, G, eng::SpinRings) = spin_alm_from_rings!(alm, G, eng)

function _apply_parity!(out, src, parity, B::Int)
    if ndims(out) == 3 && ndims(src) == 3
        @inbounds for b in 1:B
            @views out[:, :, b] .= src[:, :, b] .* parity
        end
    elseif ndims(out) == 3
        @inbounds @views out[:, :, 1] .= src .* parity
    elseif ndims(src) == 3
        @inbounds @views out .= src[:, :, 1] .* parity
    else
        out .= src .* parity
    end
    return out
end

# ---------------------------------------------------------------------------------------
# Longitude: m-coefficients on rings ↔ samples on the reduced φ grid
# ---------------------------------------------------------------------------------------

"""
`F[j,k] = Σ_m G[j,m] e^{imφ_k}` on `φ_k = 2πk/nφ`, by one planned inverse FFT along longitude.
Orders beyond the grid's Nyquist are structurally absent — `nφ ≥ 2 m_max + 1` by construction —
so the pack is a permutation, not an aliasing.
"""
function _rings_to_grid!(F, G, lmax::Integer, plan, scratch)
    _pack_orders!(scratch, G, lmax)
    LinearAlgebra.mul!(F, plan, scratch)
    return F
end

"""Adjoint of [`_rings_to_grid!`](@ref)."""
function _grid_to_rings!(G, F, lmax::Integer, plan, scratch)
    LinearAlgebra.mul!(scratch, plan, F)
    return _unpack_orders!(G, scratch, lmax)
end

function _pack_orders!(F, G, lmax::Integer)
    fill!(F, zero(eltype(F)))
    nφ = size(F, 2)
    B = size(F, 3)
    off = Int(lmax) + 1
    mkeep = min(Int(lmax), (nφ - 1) ÷ 2)
    @inbounds for b in 1:B, m in -mkeep:mkeep
        idx = mod(m, nφ) + 1
        @views F[:, idx, b] .+= G[:, m + off, b]
    end
    return F
end

function _unpack_orders!(G, F, lmax::Integer)
    fill!(G, zero(eltype(G)))
    nφ = size(F, 2)
    B = size(F, 3)
    off = Int(lmax) + 1
    mkeep = min(Int(lmax), (nφ - 1) ÷ 2)
    @inbounds for b in 1:B, m in -mkeep:mkeep
        idx = mod(m, nφ) + 1
        @views G[:, m + off, b] .= F[:, idx, b]
    end
    return G
end

# ---------------------------------------------------------------------------------------
# Colatitude periodisation
# ---------------------------------------------------------------------------------------

"""
Extend the ring stack over the full period in `Θ`. A pole-touching or full-sphere interval uses
the DFS reflection `f(-θ, φ) = (-1)^s f(θ, φ+π)`, which is the analytic continuation of the
field through the pole and hence exact. An interior interval already covers its whole period —
the window closes it at both ends — so the stack passes through unchanged.
"""
function _periodize!(F̃, F, geom::StretchGeometry, spin::Integer)
    if geom.kind === :interior
        copyto!(F̃, F)
        return F̃
    end
    return dfs_double!(F̃, F, spin)
end

"""Adjoint of [`_periodize!`](@ref)."""
function _periodize_adjoint!(F, F̃, geom::StretchGeometry, spin::Integer)
    if geom.kind === :interior
        copyto!(F, F̃)
        return F
    end
    return dfs_fold!(F, F̃, spin)
end

function _copy_field_out!(f, fbuf, B::Int)
    if ndims(f) == 1
        @views copyto!(f, fbuf[:, 1])
    else
        copyto!(f, fbuf)
    end
    return f
end

function _copy_field_in!(fbuf, f, B::Int)
    if ndims(f) == 1
        @views copyto!(fbuf[:, 1], f)
        B > 1 && (@views fbuf[:, 2:B] .= 0)
    else
        copyto!(fbuf, f)
    end
    return fbuf
end
