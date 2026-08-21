"""
Selected-θ associated-Legendre contraction onto iso-latitude rings, with polar
optimization. The ring SHT is still degree `lmax`, but only `nθ` colatitudes.
"""

struct RingScratch{T, CT, DS, AT3, VI}
    direct::DS
    G::AT3          # (nθ, 2lmax+1, B) Fourier-in-φ coefficients, CMCL m in columns
    mlims::VI
end

function RingScratch(θ_rings::AbstractVector, lmax::Integer, ntrans::Integer, ε::Real;
                     spin::Integer = 0)
    nθ = length(θ_rings)
    T = float(eltype(θ_rings))
    φdummy = _zeros_like(θ_rings, T, nθ)
    ds = DirectScratch(θ_rings, φdummy)
    CT = Complex{eltype(ds.cosθ)}
    G = _zeros_like(θ_rings, CT, nθ, 2Int(lmax) + 1, Int(ntrans))
    θh = _host(θ_rings)
    sabs = abs(Int(spin))
    mlims_h = [max(polar_mmax_lmax(T(θh[j]), lmax, ε), sabs) for j in 1:nθ]
    mlims = _to_like(θ_rings, mlims_h)
    return RingScratch{eltype(ds.cosθ), CT, typeof(ds), typeof(G), typeof(mlims)}(ds, G, mlims)
end

"""
    rings_from_alm!(G, alm, scratch, lmax, ε) -> G

`G[j, m+lmax+1, b] = Σ_ℓ a_{ℓm} P̄_ℓ^m(cos θ_j)` for `|m| ≤ polar_mmax(θ_j, lmax, ε)`.
"""
function rings_from_alm!(G, alm, scratch::RingScratch{T}, lmax::Integer, ε::Real) where {T}
    nθ = length(scratch.direct.cosθ)
    B = _ntrans(alm)
    alm = _coeff_host(alm)
    fill!(G, zero(eltype(G)))
    ds = scratch.direct
    x = ds.cosθ
    s = ds.sinθ
    Pmm = ds.Pmm
    P0 = ds.P0
    P1 = ds.P1
    P2 = ds.P2
    inv4π = _invsqrt4π(T)
    off = Int(lmax) + 1
    mlims = scratch.mlims
    bc = _use_broadcast(x)

    fill!(Pmm, inv4π)
    _ring_accum0!(G, alm, Pmm, nθ, B, 0, off)
    if lmax >= 1
        _scale_legendre!(P1, x, Pmm, sqrt(T(3)), bc)
        _ring_accum0!(G, alm, P1, nθ, B, 1, off)
        copyto!(P0, Pmm)
        pa, pb, pc = P0, P1, P2
        for ℓ in 2:Int(lmax)
            a = sqrt(T(2ℓ - 1) * T(2ℓ + 1)) / T(ℓ)
            b = sqrt(T(2ℓ + 1) / T(2ℓ - 3)) * T(ℓ - 1) / T(ℓ)
            _legendre_step!(pc, x, pb, pa, a, b, bc)
            _ring_accum0!(G, alm, pc, nθ, B, ℓ, off)
            pa, pb, pc = pb, pc, pa
        end
    end

    fill!(Pmm, inv4π)
    @inbounds for m in 1:Int(lmax)
        fac = -sqrt(T(2m + 1) / T(2m))
        _scale_inplace!(Pmm, s, fac, bc)
        _ring_accum_pm!(G, alm, Pmm, nθ, B, m, m, off, mlims)
        if m < lmax
            _scale_legendre!(P1, x, Pmm, sqrt(T(2m + 3)), bc)
            _ring_accum_pm!(G, alm, P1, nθ, B, m + 1, m, off, mlims)
            copyto!(P0, Pmm)
            pa, pb, pc = P0, P1, P2
            for ℓ in (m + 2):Int(lmax)
                a = sqrt(T(2ℓ - 1) * T(2ℓ + 1) / (T(ℓ - m) * T(ℓ + m)))
                b = sqrt(T(2ℓ + 1) / T(2ℓ - 3) * T(ℓ - m - 1) / T(ℓ - m) *
                         T(ℓ + m - 1) / T(ℓ + m))
                _legendre_step!(pc, x, pb, pa, a, b, bc)
                _ring_accum_pm!(G, alm, pc, nθ, B, ℓ, m, off, mlims)
                pa, pb, pc = pb, pc, pa
            end
        end
    end
    return G
end

function _ring_accum0!(G, alm, P, nθ, B, ℓ, off)
    if _use_broadcast(P)
        for b in 1:B
            a = _coeff(alm, ℓ, 0, b)
            iszero(a) && continue
            Gb = view(G, :, off, b)
            @. Gb += a * P
        end
        return nothing
    end
    @inbounds for b in 1:B
        a = _coeff(alm, ℓ, 0, b)
        iszero(a) && continue
        for j in 1:nθ
            G[j, off, b] += a * P[j]
        end
    end
    return nothing
end

function _ring_accum_pm!(G, alm, P, nθ, B, ℓ, m, off, mlims)
    sgn = iseven(m) ? one(eltype(P)) : -one(eltype(P))
    if _use_broadcast(P)
        for b in 1:B
            ap = _coeff(alm, ℓ, m, b)
            amv = _coeff(alm, ℓ, -m, b)
            (iszero(ap) & iszero(amv)) && continue
            Gp = view(G, :, m + off, b)
            Gm = view(G, :, -m + off, b)
            @. Gp += ifelse(mlims >= m, ap * P, zero(ap))
            @. Gm += ifelse(mlims >= m, amv * sgn * P, zero(amv))
        end
        return nothing
    end
    @inbounds for b in 1:B
        ap = _coeff(alm, ℓ, m, b)
        amv = _coeff(alm, ℓ, -m, b)
        (iszero(ap) & iszero(amv)) && continue
        for j in 1:nθ
            mlims[j] < m && continue
            G[j, m + off, b] += ap * P[j]
            G[j, -m + off, b] += amv * sgn * P[j]
        end
    end
    return nothing
end

function _ring_adj0!(alm, G, P, nθ, B, ℓ, off)
    if _use_broadcast(P)
        for b in 1:B
            _coeff_add!(alm, ℓ, 0, b, LinearAlgebra.dot(P, view(G, :, off, b)))
        end
        return nothing
    end
    @inbounds for b in 1:B
        acc = zero(eltype(alm))
        for j in 1:nθ
            acc += P[j] * G[j, off, b]
        end
        _coeff_add!(alm, ℓ, 0, b, acc)
    end
    return nothing
end

function _ring_adj_pm!(alm, G, P, nθ, B, ℓ, m, off, mlims)
    sgn = iseven(m) ? one(eltype(P)) : -one(eltype(P))
    if _use_broadcast(P)
        for b in 1:B
            Gp = view(G, :, m + off, b)
            Gm = view(G, :, -m + off, b)
            mask = mlims .>= m
            _coeff_add!(alm, ℓ, m, b, LinearAlgebra.dot(P .* mask, Gp))
            _coeff_add!(alm, ℓ, -m, b, sgn * LinearAlgebra.dot(P .* mask, Gm))
        end
        return nothing
    end
    @inbounds for b in 1:B
        accp = zero(eltype(alm))
        accm = zero(eltype(alm))
        for j in 1:nθ
            mlims[j] < m && continue
            accp += P[j] * G[j, m + off, b]
            accm += sgn * P[j] * G[j, -m + off, b]
        end
        _coeff_add!(alm, ℓ, m, b, accp)
        _coeff_add!(alm, ℓ, -m, b, accm)
    end
    return nothing
end

"""
    alm_from_rings!(alm, G, scratch, lmax, ε) -> alm

Adjoint of [`rings_from_alm!`](@ref).
"""
function alm_from_rings!(alm, G, scratch::RingScratch{T}, lmax::Integer, ε::Real) where {T}
    alm_dev = alm
    alm = _coeff_host(alm)
    fill!(alm, zero(eltype(alm)))
    nθ = length(scratch.direct.cosθ)
    B = size(G, 3)
    ds = scratch.direct
    x = ds.cosθ
    s = ds.sinθ
    Pmm = ds.Pmm
    P0 = ds.P0
    P1 = ds.P1
    P2 = ds.P2
    inv4π = _invsqrt4π(T)
    off = Int(lmax) + 1
    mlims = scratch.mlims
    bc = _use_broadcast(x)

    fill!(Pmm, inv4π)
    _ring_adj0!(alm, G, Pmm, nθ, B, 0, off)
    if lmax >= 1
        _scale_legendre!(P1, x, Pmm, sqrt(T(3)), bc)
        _ring_adj0!(alm, G, P1, nθ, B, 1, off)
        copyto!(P0, Pmm)
        pa, pb, pc = P0, P1, P2
        for ℓ in 2:Int(lmax)
            a = sqrt(T(2ℓ - 1) * T(2ℓ + 1)) / T(ℓ)
            b = sqrt(T(2ℓ + 1) / T(2ℓ - 3)) * T(ℓ - 1) / T(ℓ)
            _legendre_step!(pc, x, pb, pa, a, b, bc)
            _ring_adj0!(alm, G, pc, nθ, B, ℓ, off)
            pa, pb, pc = pb, pc, pa
        end
    end

    fill!(Pmm, inv4π)
    @inbounds for m in 1:Int(lmax)
        fac = -sqrt(T(2m + 1) / T(2m))
        _scale_inplace!(Pmm, s, fac, bc)
        _ring_adj_pm!(alm, G, Pmm, nθ, B, m, m, off, mlims)
        if m < lmax
            _scale_legendre!(P1, x, Pmm, sqrt(T(2m + 3)), bc)
            _ring_adj_pm!(alm, G, P1, nθ, B, m + 1, m, off, mlims)
            copyto!(P0, Pmm)
            pa, pb, pc = P0, P1, P2
            for ℓ in (m + 2):Int(lmax)
                a = sqrt(T(2ℓ - 1) * T(2ℓ + 1) / (T(ℓ - m) * T(ℓ + m)))
                b = sqrt(T(2ℓ + 1) / T(2ℓ - 3) * T(ℓ - m - 1) / T(ℓ - m) *
                         T(ℓ + m - 1) / T(ℓ + m))
                _legendre_step!(pc, x, pb, pa, a, b, bc)
                _ring_adj_pm!(alm, G, pc, nθ, B, ℓ, m, off, mlims)
                pa, pb, pc = pb, pc, pa
            end
        end
    end
    if alm_dev !== alm
        copyto!(alm_dev, alm)
    end
    return alm_dev
end

"""
    fourier_to_rings!(F, G, lmax)
    fourier_to_rings!(F, G, lmax, plan, scratch)

Inverse DFT in φ: `F[j,k] = Σ_m G[j, m] exp(i m φ_k)` with `φ_k = 2π(k-1)/nφ`.
CMCL `m` packing of width `2lmax+1` is aliased onto `nφ` FFT bins when
`nφ < 2lmax+1`. The 5-argument form is `bfft` along longitude (planned).
"""
function fourier_to_rings!(F::AbstractArray, G::AbstractArray, lmax::Integer)
    scratch = similar(F)
    plan = AbstractFFTs.plan_bfft(scratch, 2)
    return fourier_to_rings!(F, G, lmax, plan, scratch)
end

function fourier_to_rings!(F::AbstractArray, G::AbstractArray, lmax::Integer, plan, scratch)
    _pack_cmcl_to_fftw!(scratch, G, lmax)
    LinearAlgebra.mul!(F, plan, scratch)
    return F
end

"""
    rings_to_fourier!(G, F, lmax)
    rings_to_fourier!(G, F, lmax, plan, scratch)

Adjoint of [`fourier_to_rings!`](@ref): `G_{j m} = Σ_k F_{jk} e^{-im φ_k}`.
"""
function rings_to_fourier!(G::AbstractArray, F::AbstractArray, lmax::Integer)
    scratch = similar(F)
    plan = AbstractFFTs.plan_fft(F, 2)
    return rings_to_fourier!(G, F, lmax, plan, scratch)
end

function rings_to_fourier!(G::AbstractArray, F::AbstractArray, lmax::Integer, plan, scratch)
    LinearAlgebra.mul!(scratch, plan, F)
    return _unpack_fftw_to_cmcl!(G, scratch, lmax)
end

function _pack_cmcl_to_fftw!(F, G, lmax::Integer)
    fill!(F, zero(eltype(F)))
    nφ = size(F, 2)
    B = size(F, 3)
    off = Int(lmax) + 1
    @inbounds for b in 1:B, m in -Int(lmax):Int(lmax)
        idx = mod(m, nφ) + 1
        Gm = view(G, :, m + off, b)
        Fv = view(F, :, idx, b)
        Fv .+= Gm
    end
    return F
end

function _unpack_fftw_to_cmcl!(G, Ffft, lmax::Integer)
    fill!(G, zero(eltype(G)))
    nφ = size(Ffft, 2)
    B = size(Ffft, 3)
    off = Int(lmax) + 1
    @inbounds for b in 1:B, m in -Int(lmax):Int(lmax)
        idx = mod(m, nφ) + 1
        copyto!(view(G, :, m + off, b), view(Ffft, :, idx, b))
    end
    return G
end

"""
    rings_from_alm_spin!(G, alm, scratch, lmax, spin, θ, dl, dlp, β) -> G

Spin ring SHT: `G[j, m+off] = Σ_ℓ ₛa_{ℓm} N_ℓ d^ℓ_{m,-s}(θ_j)` via Trapani–Navaza `Δ`,
not the log-gamma sum. Polar `mlims` still apply, but never drop `|m| < |s|`.
"""
function rings_from_alm_spin!(G, alm, scratch::RingScratch, lmax::Integer, spin::Integer,
                              θ, dl, dlp, β)
    if _use_broadcast(G) || _use_broadcast(alm) || _use_broadcast(θ)
        Gh = Array(G)
        _rings_from_alm_spin_cpu!(Gh, Array(alm), Array(scratch.mlims), lmax, spin,
                                  Array(θ), Array(dl), Array(dlp), Array(β))
        copyto!(G, Gh)
        return G
    end
    return _rings_from_alm_spin_cpu!(G, alm, scratch.mlims, lmax, spin, θ, dl, dlp, β)
end

function _rings_from_alm_spin_cpu!(G, alm, mlims, lmax::Integer, spin::Integer, θ, dl, dlp, β)
    fill!(G, zero(eltype(G)))
    fill!(β, zero(eltype(β)))
    T = float(eltype(θ))
    CT = eltype(G)
    nθ = length(θ)
    B = size(G, 3)
    s = Int(spin)
    n = -s
    off = Int(lmax) + 1
    _wigner_d_halfpi_step!(dl, dlp, 0, off)
    @inbounds for ℓ in 0:Int(lmax)
        if ℓ > 0
            dl, dlp = dlp, dl
            _wigner_d_halfpi_step!(dl, dlp, ℓ, off)
        end
        ℓ < abs(s) && continue
        Nfac = CT(_Nℓ(T, ℓ))
        for b in 1:B, m in -ℓ:ℓ
            a = _spin_coeff(alm, ℓ, m, lmax, b)
            iszero(a) && continue
            pref = a * Nfac * _im_pow(CT, m - n)
            for k in -ℓ:ℓ
                β[m + off, k + off, b] += pref * dl[k + off, m + off] * dl[k + off, n + off]
            end
        end
    end
    @inbounds for b in 1:B, j in 1:nθ, m in -Int(lmax):Int(lmax)
        abs(m) > mlims[j] && continue
        acc = zero(CT)
        for k in -Int(lmax):Int(lmax)
            acc += β[m + off, k + off, b] * cis(-T(k) * θ[j])
        end
        G[j, m + off, b] = acc
    end
    return G
end

"""
    alm_from_rings_spin!(alm, G, scratch, lmax, spin, θ, dl, dlp, γ) -> alm

Transpose adjoint of [`rings_from_alm_spin!`](@ref).
"""
function alm_from_rings_spin!(alm, G, scratch::RingScratch, lmax::Integer, spin::Integer,
                              θ, dl, dlp, γ)
    if _use_broadcast(alm) || _use_broadcast(G) || _use_broadcast(θ)
        almh = Array(alm)
        _alm_from_rings_spin_cpu!(almh, Array(G), Array(scratch.mlims), lmax, spin,
                                  Array(θ), Array(dl), Array(dlp), Array(γ))
        copyto!(alm, almh)
        return alm
    end
    return _alm_from_rings_spin_cpu!(alm, G, scratch.mlims, lmax, spin, θ, dl, dlp, γ)
end

function _alm_from_rings_spin_cpu!(alm, G, mlims, lmax::Integer, spin::Integer, θ, dl, dlp, γ)
    fill!(alm, zero(eltype(alm)))
    fill!(γ, zero(eltype(γ)))
    T = float(eltype(θ))
    CT = eltype(alm)
    nθ = length(θ)
    B = size(G, 3)
    s = Int(spin)
    n = -s
    off = Int(lmax) + 1
    @inbounds for b in 1:B, j in 1:nθ, m in -Int(lmax):Int(lmax)
        abs(m) > mlims[j] && continue
        g = G[j, m + off, b]
        iszero(g) && continue
        for k in -Int(lmax):Int(lmax)
            γ[m + off, k + off, b] += g * cis(-T(k) * θ[j])
        end
    end
    _wigner_d_halfpi_step!(dl, dlp, 0, off)
    @inbounds for ℓ in 0:Int(lmax)
        if ℓ > 0
            dl, dlp = dlp, dl
            _wigner_d_halfpi_step!(dl, dlp, ℓ, off)
        end
        ℓ < abs(s) && continue
        Nfac = CT(_Nℓ(T, ℓ))
        for b in 1:B, m in -ℓ:ℓ
            acc = zero(CT)
            for k in -ℓ:ℓ
                acc += dl[k + off, m + off] * dl[k + off, n + off] * γ[m + off, k + off, b]
            end
            _spin_coeff_add!(alm, ℓ, m, lmax, b, Nfac * _im_pow(CT, m - n) * acc)
        end
    end
    return alm
end
