"""
Carron–Reinecke polar-stretch leaf: rotate the bounding cap to the pole, evaluate
the degree-`lmax` field on `Lmax+1` polar-optimized rings, apodize, and interpolate
with a reduced Fourier sphere + NUFFT.
"""

struct StretchLeaf{T, CT, VT, CTV, AT3, AT2, MT, RS, FT, IT, N2, N1, DLM, RFT, RIT}
    cap::BoundingCap{T}
    lmax::Int
    Lmax::Int
    ntrans::Int
    spin::Int
    tol::T
    θmax::T
    α::T
    β::T
    γ::T
    mmax::Int
    nθ::Int
    nφ::Int
    θ_rings::VT
    Θ_rings::VT
    w::VT
    θ_rot::VT
    φ_rot::VT
    Θ_pts::VT
    alm_rot::AT3
    rings::RS
    F::AT3
    Fφ::AT3
    F̃::AT3
    Fhat::AT3
    fbuf::AT2
    phase_fwd::CTV
    phase_adj::CTV
    spin_phase::CTV
    spin_phase_adj::CTV
    fft_plan::FT
    ifft_plan::IT
    ring_bfft_plan::RFT
    ring_fft_plan::RIT
    nufft2::N2
    nufft1::N1
    dl::MT
    dlp::MT
    work_wigner::CTV
    wigner_src::CTV
    Δcache::Vector{Matrix{T}}
    dcache::WignerThinD{T}
    ring_β::AT3
    ring_γ::AT3
    nufft_backend::DLM
end

function StretchLeaf(θ::AbstractVector, φ::AbstractVector, lmax::Integer, ntrans::Integer,
                     tol::Real; nufft_backend = DirectSumBackend(), cap = nothing,
                     spin::Integer = 0)
    T = float(eltype(θ))
    CT = Complex{T}
    cap = cap === nothing ? bounding_cap(θ, φ) : cap
    sp = stretch_params(cap.θstar, lmax, tol)
    Lmax = sp.Lmax
    θmax = T(sp.θmax)
    nθ = Lmax + 1
    mmax = max(polar_mmax(θmax, lmax, tol), abs(Int(spin)))
    nφ = 2 * max(Lmax, mmax) + 1
    α, β, γ = euler_to_pole(cap)
    Θ_host = T[T(π) * (i - T(0.5)) / nθ for i in 1:nθ]
    θ_rings_host = Θ_host .* (θmax / T(π))
    w_host = stretch_window(Θ_host, cap.θstar, θmax, tol)
    θ_rings = _to_like(θ, θ_rings_host)
    Θ_rings = _to_like(θ, Θ_host)
    w = _to_like(θ, w_host)
    n = length(θ)
    θh, φh = _host(θ), _host(φ)
    θ_rot_h = similar(θh, T)
    φ_rot_h = similar(φh, T)
    rotate_to_pole!(θ_rot_h, φ_rot_h, θh, φh, cap)
    θ_rot = _to_like(θ, θ_rot_h)
    φ_rot = _to_like(θ, φ_rot_h)
    Θ_pts = similar(θ_rot, T)
    Θ_pts .= T(π) .* θ_rot ./ θmax
    B = Int(ntrans)
    nθc, nφc = coeff_size(lmax)
    alm_rot = _zeros_like(θ, CT, nθc, nφc, B)
    rings = RingScratch(θ_rings, lmax, B, tol; spin = Int(spin))
    F = _zeros_like(θ, CT, nθ, nφ, B)
    Fφ = _zeros_like(θ, CT, nθ, nφ, B)
    F̃ = _zeros_like(θ, CT, 2nθ, nφ, B)
    Fhat = _zeros_like(θ, CT, 2nθ, nφ, B)
    fbuf = _zeros_like(θ, CT, n, B)
    fft_plan = AbstractFFTs.plan_fft(F̃, (1, 2))
    ifft_plan = AbstractFFTs.plan_bfft(Fhat, (1, 2))
    ring_bfft_plan = AbstractFFTs.plan_bfft(Fφ, 2)
    ring_fft_plan = AbstractFFTs.plan_fft(Fφ, 2)
    nθ_dbl = 2nθ
    N = T(nθ_dbl * nφ)
    phase_host = CT[cis(-T(π) * T(_mode_k(i, nθ_dbl, 1)) / T(nθ_dbl)) / N for i in 1:nθ_dbl]
    phase_fwd = _to_like(θ, phase_host)
    phase_adj = _to_like(θ, conj.(phase_host))
    sph_h = Vector{CT}(undef, n)
    spin_basis_phase!(sph_h, θh, φh, θ_rot_h, φ_rot_h, cap, Int(spin))
    spin_phase = _to_like(θ, sph_h)
    spin_phase_adj = _to_like(θ, conj.(sph_h))
    nufft2 = _nufft_makeplan(nufft_backend, Θ_pts, φ_rot, 2, (nθ_dbl, nφ), +1, B, float(tol);
                             dtype = T, modeord = 1)
    nufft1 = _nufft_makeplan(nufft_backend, Θ_pts, φ_rot, 1, (nθ_dbl, nφ), -1, B, float(tol);
                             dtype = T, modeord = 1)
    Lw = 2Int(lmax) + 1
    dl = _zeros_like(θ, T, Lw, Lw)
    dlp = _zeros_like(θ, T, Lw, Lw)
    work = _zeros_like(θ, CT, Lw)
    wsrc = _zeros_like(θ, CT, Lw)
    Δcache = Matrix{T}[]
    dcache = wigner_thin_d_cache(T, Int(lmax), mmax, β)
    ring_β = _zeros_like(θ, CT, Lw, Lw, B)
    ring_γ = _zeros_like(θ, CT, Lw, Lw, B)
    return StretchLeaf{T, CT, typeof(θ_rings), typeof(phase_fwd), typeof(alm_rot), typeof(fbuf),
                       typeof(dl), typeof(rings), typeof(fft_plan), typeof(ifft_plan),
                       typeof(nufft2), typeof(nufft1), typeof(nufft_backend),
                       typeof(ring_bfft_plan), typeof(ring_fft_plan)}(
        cap, Int(lmax), Lmax, B, Int(spin), T(tol), θmax, T(α), T(β), T(γ), mmax, nθ, nφ,
        θ_rings, Θ_rings, w, θ_rot, φ_rot, Θ_pts, alm_rot, rings, F, Fφ, F̃, Fhat, fbuf,
        phase_fwd, phase_adj, spin_phase, spin_phase_adj, fft_plan, ifft_plan,
        ring_bfft_plan, ring_fft_plan, nufft2, nufft1, dl, dlp, work, wsrc, Δcache, dcache,
        ring_β, ring_γ, nufft_backend)
end

function synthesize_stretch!(f, alm, leaf::StretchLeaf)
    B = leaf.ntrans
    _assert_coeffs(alm, leaf.lmax, B)
    _assert_field(f, length(leaf.θ_rot), B)
    rotate_alm!(leaf.alm_rot, alm, leaf.α, leaf.β, leaf.γ, leaf.lmax, leaf.mmax,
                leaf.dl, leaf.dlp, leaf.work_wigner, leaf.spin, leaf.wigner_src,
                leaf.Δcache, leaf.dcache, false)
    if leaf.spin == 0
        rings_from_alm!(leaf.rings.G, leaf.alm_rot, leaf.rings, leaf.lmax, leaf.tol)
    else
        rings_from_alm_spin!(leaf.rings.G, leaf.alm_rot, leaf.rings, leaf.lmax, leaf.spin,
                             leaf.θ_rings, leaf.dl, leaf.dlp, leaf.ring_β)
    end
    fourier_to_rings!(leaf.F, leaf.rings.G, leaf.lmax, leaf.ring_bfft_plan, leaf.Fφ)
    leaf.F .*= leaf.w
    dfs_double!(leaf.F̃, leaf.F, leaf.spin)
    LinearAlgebra.mul!(leaf.Fhat, leaf.fft_plan, leaf.F̃)
    leaf.Fhat .*= leaf.phase_fwd
    _nufft_exec!(leaf.nufft2, leaf.Fhat, leaf.fbuf)
    if leaf.spin != 0
        leaf.fbuf .*= leaf.spin_phase
    end
    _copy_fbuf_out!(f, leaf.fbuf, B)
    return f
end

function adjoint_stretch!(alm, f, leaf::StretchLeaf)
    B = leaf.ntrans
    _assert_coeffs(alm, leaf.lmax, B)
    _assert_field(f, length(leaf.θ_rot), B)
    _copy_fbuf_in!(leaf.fbuf, f, B)
    if leaf.spin != 0
        leaf.fbuf .*= leaf.spin_phase_adj
    end
    _nufft_exec!(leaf.nufft1, leaf.fbuf, leaf.Fhat)
    leaf.Fhat .*= leaf.phase_adj
    LinearAlgebra.mul!(leaf.F̃, leaf.ifft_plan, leaf.Fhat)
    dfs_fold!(leaf.F, leaf.F̃, leaf.spin)
    leaf.F .*= leaf.w
    rings_to_fourier!(leaf.rings.G, leaf.F, leaf.lmax, leaf.ring_fft_plan, leaf.Fφ)
    if leaf.spin == 0
        alm_from_rings!(leaf.alm_rot, leaf.rings.G, leaf.rings, leaf.lmax, leaf.tol)
    else
        alm_from_rings_spin!(leaf.alm_rot, leaf.rings.G, leaf.rings, leaf.lmax, leaf.spin,
                             leaf.θ_rings, leaf.dl, leaf.dlp, leaf.ring_γ)
    end
    # D(α,β,γ)^{-1} = D(-γ, -β, -α)
    rotate_alm!(alm, leaf.alm_rot, -leaf.γ, -leaf.β, -leaf.α, leaf.lmax, leaf.lmax,
                leaf.dl, leaf.dlp, leaf.work_wigner, leaf.spin, leaf.wigner_src,
                leaf.Δcache, leaf.dcache, true)
    return alm
end

function _copy_fbuf_out!(f, fbuf, B)
    if ndims(f) == 1
        copyto!(f, view(fbuf, :, 1))
    else
        copyto!(f, fbuf)
    end
    return f
end

function _copy_fbuf_in!(fbuf, f, B)
    if ndims(f) == 1
        copyto!(view(fbuf, :, 1), f)
        if B > 1
            view(fbuf, :, 2:B) .= 0
        end
    else
        copyto!(fbuf, f)
    end
    return fbuf
end
