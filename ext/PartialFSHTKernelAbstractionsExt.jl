"""
    PartialFSHTKernelAbstractionsExt

Device-generic kernels: DFS doubling/folding, and the polar-optimised associated-Legendre ring
pass. Written once against KernelAbstractions so they run on any backend, and dispatched on
`GPUArraysCore.AbstractGPUArray`. Loaded by `using KernelAbstractions`.

The ring pass maps better onto a GPU than onto a CPU. Parallelising over `(ring, order)` puts the
whole recurrence state — the two-term window *and* the block-scaling exponent — in registers, so
there is no shared scratch to contend over, and in the forward direction no two threads write the
same output. That is the mirror image of the host path, where a single shared scratch buffer was
the only thing keeping the order loop serial.
"""
module PartialFSHTKernelAbstractionsExt

using PartialFSHT: PartialFSHT
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
using GPUArraysCore: GPUArraysCore

@inline function _sync(backend)
    applicable(KernelAbstractions.synchronize, backend) && KernelAbstractions.synchronize(backend)
    return nothing
end

# Coefficient index arithmetic, inlined. `FastSphericalHarmonics.sph_mode` is the same
# expression, but it carries `@assert`s, and a kernel should not contain a throw path.
@inline function _cidx(ℓ::Int, m::Int)
    am = abs(m)
    return (ℓ - am + 1, 2 * am + (m >= 0 ? 1 : 0))
end

# ---------------------------------------------------------------------------------------
# DFS periodisation
# ---------------------------------------------------------------------------------------

@kernel function _dfs_double_kern!(F̃, @Const(F), Nθ, Nφ, half, sgn)
    i, j, b = @index(Global, NTuple)
    if i <= Nθ
        @inbounds F̃[i, j, b] = F[i, j, b]
    else
        ii = i - Nθ
        js = mod1(j + half, Nφ)
        @inbounds F̃[i, j, b] = sgn * F[Nθ + 1 - ii, js, b]
    end
end

PartialFSHT.dfs_double!(F̃::GPUArraysCore.AbstractGPUArray, F::GPUArraysCore.AbstractGPUArray) =
    PartialFSHT.dfs_double!(F̃, F, 0)

function PartialFSHT.dfs_double!(F̃::GPUArraysCore.AbstractGPUArray,
                                 F::GPUArraysCore.AbstractGPUArray, s::Integer)
    Nθ, Nφ, B = size(F, 1), size(F, 2), size(F, 3)
    @assert size(F̃, 1) == 2Nθ && size(F̃, 2) == Nφ && size(F̃, 3) == B
    backend = KernelAbstractions.get_backend(F̃)
    sgn = iseven(Int(s)) ? one(eltype(F)) : -one(eltype(F))
    _dfs_double_kern!(backend)(F̃, F, Nθ, Nφ, Nφ ÷ 2, sgn; ndrange = (2Nθ, Nφ, B))
    _sync(backend)
    return F̃
end

@kernel function _dfs_fold_kern!(F, @Const(F̃), Nθ, Nφ, half, sgn)
    i, j, b = @index(Global, NTuple)
    im = 2Nθ + 1 - i
    js = mod1(j - half, Nφ)
    @inbounds F[i, j, b] = F̃[i, j, b] + sgn * F̃[im, js, b]
end

PartialFSHT.dfs_fold!(F::GPUArraysCore.AbstractGPUArray, F̃::GPUArraysCore.AbstractGPUArray) =
    PartialFSHT.dfs_fold!(F, F̃, 0)

function PartialFSHT.dfs_fold!(F::GPUArraysCore.AbstractGPUArray,
                               F̃::GPUArraysCore.AbstractGPUArray, s::Integer)
    Nθ, Nφ, B = size(F, 1), size(F, 2), size(F, 3)
    @assert size(F̃, 1) == 2Nθ && size(F̃, 2) == Nφ && size(F̃, 3) == B
    backend = KernelAbstractions.get_backend(F)
    sgn = iseven(Int(s)) ? one(eltype(F)) : -one(eltype(F))
    _dfs_fold_kern!(backend)(F, F̃, Nθ, Nφ, Nφ ÷ 2, sgn; ndrange = (Nθ, Nφ, B))
    _sync(backend)
    return F
end

# ---------------------------------------------------------------------------------------
# Ring pass: one thread per (ring, order)
# ---------------------------------------------------------------------------------------

@kernel function _rings_fwd_kern!(G, @Const(alm), @Const(cosθ), @Const(log2sinθ),
                                  @Const(log2c), @Const(mlim), L::Int, B::Int, stride::Int)
    j, mi = @index(Global, NTuple)
    m = mi - 1
    @inbounds if mlim[j] >= m
        T = eltype(cosθ)
        CT = eltype(G)
        off = L + 1
        x = cosθ[j]
        sgn = iseven(m) ? one(T) : -one(T)
        e0 = m == 0 ? log2c[1] : T(m) * log2sinθ[j] + log2c[m + 1]
        if isfinite(e0)
            Eseed = ceil(e0)
            pseed = sgn * exp2(e0 - Eseed)
            for b in 1:B
                q2 = pseed
                E = Eseed
                ip, jp = _cidx(m, m)
                im_, jm_ = _cidx(m, -m)
                ap = alm[ip, jp, b]
                am = m == 0 ? zero(CT) : alm[im_, jm_, b]
                accp = ap * q2
                accm = am * sgn * q2
                q1 = q2
                if m < L
                    q1 = sqrt(T(2m + 3)) * x * q2
                    ip, jp = _cidx(m + 1, m)
                    im_, jm_ = _cidx(m + 1, -m)
                    ap = alm[ip, jp, b]
                    am = m == 0 ? zero(CT) : alm[im_, jm_, b]
                    accp += ap * q1
                    accm += am * sgn * q1
                    steps = 0
                    for ℓ in (m + 2):L
                        a = sqrt(T(2ℓ - 1) * T(2ℓ + 1) / (T(ℓ - m) * T(ℓ + m)))
                        bb = sqrt(T(2ℓ + 1) / T(2ℓ - 3) * T(ℓ - m - 1) / T(ℓ - m) *
                                  T(ℓ + m - 1) / T(ℓ + m))
                        p = a * x * q1 - bb * q2
                        q2 = q1
                        q1 = p
                        ip, jp = _cidx(ℓ, m)
                        im_, jm_ = _cidx(ℓ, -m)
                        ap = alm[ip, jp, b]
                        am = m == 0 ? zero(CT) : alm[im_, jm_, b]
                        accp += ap * q1
                        accm += am * sgn * q1
                        steps += 1
                        if steps == stride && ℓ < L
                            steps = 0
                            mx = max(abs(q1), abs(q2))
                            if mx > 0 && isfinite(mx)
                                sh = ceil(log2(mx))
                                if sh != 0
                                    fsc = exp2(-sh)
                                    q1 *= fsc
                                    q2 *= fsc
                                    accp *= fsc
                                    accm *= fsc
                                    E += sh
                                end
                            end
                        end
                    end
                end
                fac = exp2(E)
                G[j, m + off, b] = accp * fac
                if m > 0
                    G[j, -m + off, b] = accm * fac
                end
            end
        end
    end
end

# Adjoint: same walk, but every (ring, order) thread scatters into shared coefficient slots, so
# the accumulation is atomic (`KernelAbstractions.@atomic`, which is Atomix's, re-exported).
# `almre` is the coefficient array reinterpreted as `(2, …)` reals: there is no atomic add for a
# `Complex` element, and two independent real adds are equivalent because the real and imaginary
# parts never interact.
#
# Contention is up to one thread per retained ring on the same (ℓ, m) slot. That is the price of
# keeping the forward direction race-free and register-resident; the alternative — parallelising
# the adjoint over order alone — removes the atomics but hands one thread the whole
# rings × degrees walk, which is both long-running and badly load-imbalanced.
@kernel function _rings_adj_kern!(almre, @Const(G), @Const(cosθ), @Const(log2sinθ),
                                  @Const(log2c), @Const(mlim), L::Int, B::Int, stride::Int)
    j, mi = @index(Global, NTuple)
    m = mi - 1
    @inbounds if mlim[j] >= m
        T = eltype(almre)
        off = L + 1
        x = cosθ[j]
        sgn = iseven(m) ? one(T) : -one(T)
        e0 = m == 0 ? log2c[1] : T(m) * log2sinθ[j] + log2c[m + 1]
        if isfinite(e0)
            Eseed = ceil(e0)
            pseed = sgn * exp2(e0 - Eseed)
            for b in 1:B
                gp = G[j, m + off, b]
                gm = m == 0 ? zero(gp) : G[j, -m + off, b]
                q2 = pseed
                E = Eseed
                _atomic_add!(almre, m, m, b, (q2 * exp2(E)) * gp)
                if m > 0
                    _atomic_add!(almre, m, -m, b, (q2 * sgn * exp2(E)) * gm)
                end
                if m < L
                    q1 = sqrt(T(2m + 3)) * x * q2
                    _atomic_add!(almre, m + 1, m, b, (q1 * exp2(E)) * gp)
                    if m > 0
                        _atomic_add!(almre, m + 1, -m, b, (q1 * sgn * exp2(E)) * gm)
                    end
                    steps = 0
                    for ℓ in (m + 2):L
                        a = sqrt(T(2ℓ - 1) * T(2ℓ + 1) / (T(ℓ - m) * T(ℓ + m)))
                        bb = sqrt(T(2ℓ + 1) / T(2ℓ - 3) * T(ℓ - m - 1) / T(ℓ - m) *
                                  T(ℓ + m - 1) / T(ℓ + m))
                        p = a * x * q1 - bb * q2
                        q2 = q1
                        q1 = p
                        steps += 1
                        if steps == stride && ℓ < L
                            steps = 0
                            mx = max(abs(q1), abs(q2))
                            if mx > 0 && isfinite(mx)
                                sh = ceil(log2(mx))
                                if sh != 0
                                    fsc = exp2(-sh)
                                    q1 *= fsc
                                    q2 *= fsc
                                    E += sh
                                end
                            end
                        end
                        w = q1 * exp2(E)
                        _atomic_add!(almre, ℓ, m, b, w * gp)
                        if m > 0
                            _atomic_add!(almre, ℓ, -m, b, (w * sgn) * gm)
                        end
                    end
                end
            end
        end
    end
end

@inline function _atomic_add!(almre, ℓ::Int, m::Int, b::Int, v)
    i1, i2 = _cidx(ℓ, m)
    @inbounds KernelAbstractions.@atomic almre[1, i1, i2, b] += real(v)
    @inbounds KernelAbstractions.@atomic almre[2, i1, i2, b] += imag(v)
    return nothing
end

"""A coefficient array viewed with the trailing batch axis the kernels index with."""
@inline _batched(a) = ndims(a) == 3 ? a : reshape(a, size(a, 1), size(a, 2), 1)

function PartialFSHT._walk_rings!(dir, G::GPUArraysCore.AbstractGPUArray, alm,
                                  eng::PartialFSHT.RingLegendre, backend)
    L = eng.lmax
    B = eng.ntrans
    nθ = length(eng.θ)
    nm = min(eng.mmax, L) + 1
    be = KernelAbstractions.get_backend(G)
    stride = PartialFSHT._RESCALE_STRIDE
    _require_device(be, eng.cosθ, "ring geometry")
    if dir isa PartialFSHT._Fwd
        _rings_fwd_kern!(be)(G, _batched(alm), eng.cosθ, eng.log2sinθ, eng.log2c,
                             eng.mlim, L, B, stride; ndrange = (nθ, nm))
    else
        almre = reinterpret(reshape, real(eltype(alm)), _batched(alm))
        _rings_adj_kern!(be)(almre, G, eng.cosθ, eng.log2sinθ, eng.log2c,
                            eng.mlim, L, B, stride; ndrange = (nθ, nm))
    end
    _sync(be)
    return nothing
end

# A device apply whose engine tables live on the host is the one failure mode that reads as working
# on a CPU-backed KA backend and dies on real hardware, so it is checked rather than left to the
# kernel launch.
@inline function _require_device(be, a, what)
    KernelAbstractions.get_backend(a) === be || throw(ArgumentError(
        "$what is on $(KernelAbstractions.get_backend(a)) but the transform is running on $be — " *
        "build the plan from a node set of the intended array type"))
    return nothing
end

# ---------------------------------------------------------------------------------------
# Spin ring pass: one thread per (ring, order), the seed read from the engine's tables
# ---------------------------------------------------------------------------------------

@inline function _spin_seed(::Type{T}, lb2, pc::Int, ps::Int, l2c::T, l2s::T) where {T}
    e = lb2
    pc != 0 && (e += T(pc) * l2c)
    ps != 0 && (e += T(ps) * l2s)
    return e
end

@kernel function _spin_fwd_kern!(G, @Const(alm), @Const(cosθ), @Const(log2cos), @Const(log2sin),
                                 @Const(Nℓ), @Const(lb2), @Const(pcs), @Const(pss), @Const(sgns),
                                 @Const(mlim), L::Int, B::Int, spin::Int, stride::Int)
    j, mi = @index(Global, NTuple)
    m = mi - L - 1
    @inbounds if mlim[j] >= abs(m)
        T = eltype(cosθ)
        CT = eltype(G)
        off = L + 1
        nn = -spin
        l0 = max(abs(m), abs(nn))
        if l0 <= L
            x = cosθ[j]
            e0 = _spin_seed(T, lb2[mi], pcs[mi], pss[mi], log2cos[j], log2sin[j])
            if isfinite(e0)
                Eseed = ceil(e0)
                dseed = sgns[mi] * exp2(e0 - Eseed)
                for b in 1:B
                    E = Eseed
                    d1 = dseed
                    d0 = zero(T)
                    # Grouped as `(coefficient × Nℓ) × d`, exactly as the host walk does, so the
                    # forward direction is bit-identical rather than merely equal to round-off.
                    # Complex multiplication is not associative in floating point.
                    acc = (alm[l0 + 1, m + off, b] * Nℓ[l0 + 1]) * d1
                    Lc = l0
                    steps = 0
                    while Lc < L
                        L1 = Lc + 1
                        sm1 = sqrt(max(zero(T), T(L1 * L1 - m * m)))
                        sn1 = sqrt(max(zero(T), T(L1 * L1 - nn * nn)))
                        den = T(Lc) * sm1 * sn1
                        if den <= 0
                            d0 = d1
                            d1 = zero(T)
                            Lc = L1
                            continue
                        end
                        sm0 = sqrt(max(zero(T), T(Lc * Lc - m * m)))
                        sn0 = sqrt(max(zero(T), T(Lc * Lc - nn * nn)))
                        a1 = T(2 * Lc + 1) * T(Lc) * T(L1) / den
                        a2 = T(2 * Lc + 1) * T(m) * T(nn) / den
                        a0 = T(L1) * sm0 * sn0 / den
                        dn = (a1 * x - a2) * d1 - a0 * d0
                        d0 = d1
                        d1 = dn
                        Lc = L1
                        acc += (alm[L1 + 1, m + off, b] * Nℓ[L1 + 1]) * d1
                        steps += 1
                        if steps == stride && Lc < L
                            steps = 0
                            mx = max(abs(d1), abs(d0))
                            if mx > 0 && isfinite(mx)
                                sh = ceil(log2(mx))
                                if sh != 0
                                    fsc = exp2(-sh)
                                    d1 *= fsc
                                    d0 *= fsc
                                    acc *= fsc
                                    E += sh
                                end
                            end
                        end
                    end
                    G[j, m + off, b] = acc * exp2(E)
                end
            end
        end
    end
end

@kernel function _spin_adj_kern!(almre, @Const(G), @Const(cosθ), @Const(log2cos), @Const(log2sin),
                                 @Const(Nℓ), @Const(lb2), @Const(pcs), @Const(pss), @Const(sgns),
                                 @Const(mlim), L::Int, B::Int, spin::Int, stride::Int)
    j, mi = @index(Global, NTuple)
    m = mi - L - 1
    @inbounds if mlim[j] >= abs(m)
        T = eltype(almre)
        off = L + 1
        nn = -spin
        l0 = max(abs(m), abs(nn))
        if l0 <= L
            x = cosθ[j]
            e0 = _spin_seed(T, lb2[mi], pcs[mi], pss[mi], log2cos[j], log2sin[j])
            if isfinite(e0)
                Eseed = ceil(e0)
                dseed = sgns[mi] * exp2(e0 - Eseed)
                for b in 1:B
                    g = G[j, m + off, b]
                    E = Eseed
                    d1 = dseed
                    d0 = zero(T)
                    _atomic_add_spin!(almre, l0, m, off, b, (d1 * exp2(E) * Nℓ[l0 + 1]) * g)
                    Lc = l0
                    steps = 0
                    while Lc < L
                        L1 = Lc + 1
                        sm1 = sqrt(max(zero(T), T(L1 * L1 - m * m)))
                        sn1 = sqrt(max(zero(T), T(L1 * L1 - nn * nn)))
                        den = T(Lc) * sm1 * sn1
                        if den <= 0
                            d0 = d1
                            d1 = zero(T)
                            Lc = L1
                            continue
                        end
                        sm0 = sqrt(max(zero(T), T(Lc * Lc - m * m)))
                        sn0 = sqrt(max(zero(T), T(Lc * Lc - nn * nn)))
                        a1 = T(2 * Lc + 1) * T(Lc) * T(L1) / den
                        a2 = T(2 * Lc + 1) * T(m) * T(nn) / den
                        a0 = T(L1) * sm0 * sn0 / den
                        dn = (a1 * x - a2) * d1 - a0 * d0
                        d0 = d1
                        d1 = dn
                        Lc = L1
                        steps += 1
                        if steps == stride && Lc < L
                            steps = 0
                            mx = max(abs(d1), abs(d0))
                            if mx > 0 && isfinite(mx)
                                sh = ceil(log2(mx))
                                if sh != 0
                                    fsc = exp2(-sh)
                                    d1 *= fsc
                                    d0 *= fsc
                                    E += sh
                                end
                            end
                        end
                        _atomic_add_spin!(almre, Lc, m, off, b, (d1 * exp2(E) * Nℓ[Lc + 1]) * g)
                    end
                end
            end
        end
    end
end

# Spin coefficients use the dense `(ℓ+1, m+lmax+1)` layout, so the index needs no `_cidx`.
@inline function _atomic_add_spin!(almre, ℓ::Int, m::Int, off::Int, b::Int, v)
    @inbounds KernelAbstractions.@atomic almre[1, ℓ + 1, m + off, b] += real(v)
    @inbounds KernelAbstractions.@atomic almre[2, ℓ + 1, m + off, b] += imag(v)
    return nothing
end

function PartialFSHT._spin_walk!(dir, G::GPUArraysCore.AbstractGPUArray, alm,
                                 eng::PartialFSHT.SpinRings, backend)
    L = eng.lmax
    B = eng.ntrans
    nθ = length(eng.θ)
    be = KernelAbstractions.get_backend(G)
    stride = PartialFSHT._RESCALE_STRIDE
    _require_device(be, eng.cosθ, "spin ring geometry")
    if dir isa PartialFSHT._Fwd
        _spin_fwd_kern!(be)(G, _batched(alm), eng.cosθ, eng.log2cos_h, eng.log2sin_h,
                            eng.Nℓ, eng.seed_lb2, eng.seed_pc, eng.seed_ps, eng.seed_sgn,
                            eng.mlim, L, B, eng.spin, stride; ndrange = (nθ, 2L + 1))
    else
        almre = reinterpret(reshape, real(eltype(alm)), _batched(alm))
        _spin_adj_kern!(be)(almre, G, eng.cosθ, eng.log2cos_h, eng.log2sin_h,
                            eng.Nℓ, eng.seed_lb2, eng.seed_pc, eng.seed_ps, eng.seed_sgn,
                            eng.mlim, L, B, eng.spin, stride; ndrange = (nθ, 2L + 1))
    end
    _sync(be)
    return nothing
end

end
