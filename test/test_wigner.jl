Test.@testset "Wigner-d ℓ=1 closed forms" begin
    β = 0.37
    Test.@test PartialFSHT.wigner_d(1, 0, 0, β) ≈ cos(β)
    Test.@test PartialFSHT.wigner_d(1, 1, 0, β) ≈ -sin(β) / sqrt(2)
    Test.@test PartialFSHT.wigner_d(1, 0, 1, β) ≈ sin(β) / sqrt(2)
    Test.@test PartialFSHT.wigner_d(1, 1, 1, β) ≈ (1 + cos(β)) / 2
    Test.@test PartialFSHT.wigner_d(1, 1, -1, β) ≈ (1 - cos(β)) / 2
    Test.@test PartialFSHT.wigner_d(1, -1, 0, β) ≈ sin(β) / sqrt(2)
    Test.@test PartialFSHT.wigner_d(1, 0, -1, β) ≈ -sin(β) / sqrt(2)
end

Test.@testset "Wigner rotation vs rotated points" begin
    Random.seed!(4)
    lmax = 5
    n = 30
    θ = 0.25 .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    cap = PartialFSHT.bounding_cap(θ, φ)
    α, β, γ = PartialFSHT.euler_to_pole(cap)
    θ′, φ′ = PartialFSHT.rotate_to_pole(θ, φ, cap)

    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    out = similar(alm)
    PartialFSHT.rotate_alm!(out, alm, α, β, γ, lmax, lmax)

    brute = similar(alm)
    fill!(brute, 0)
    for ℓ in 0:lmax, mp in -ℓ:ℓ
        acc = 0.0im
        for m in -ℓ:ℓ
            D = cis(-mp * α) * PartialFSHT.wigner_d(ℓ, mp, m, β) * cis(-m * γ)
            acc += D * PartialFSHT._coeff(alm, ℓ, m, 1)
        end
        PartialFSHT._coeff_set!(brute, ℓ, mp, 1, acc)
    end
    Test.@test maximum(abs.(out .- brute)) < 1e-12

    p0 = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
    p1 = PartialFSHT.Plan(θ′, φ′; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
    f0 = p0 * alm
    f1 = p1 * out
    Test.@test maximum(abs.(f0 .- f1)) < 1e-10
end

Test.@testset "spin Wigner rotation vs rotated points" begin
    Random.seed!(19)
    lmax = 5
    s = 1
    n = 30
    θ = 0.25 .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    cap = PartialFSHT.bounding_cap(θ, φ)
    α, β, γ = PartialFSHT.euler_to_pole(cap)
    θ′, φ′ = PartialFSHT.rotate_to_pole(θ, φ, cap)

    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    out = similar(alm)
    PartialFSHT.rotate_alm!(out, alm, α, β, γ, lmax, lmax, s)

    p0 = PartialFSHT.Plan(θ, φ; lmax = lmax, spin = s, method = PartialFSHT.DirectAlgorithm())
    p1 = PartialFSHT.Plan(θ′, φ′; lmax = lmax, spin = s, method = PartialFSHT.DirectAlgorithm())
    f0 = p0 * alm
    f1 = p1 * out
    χ = PartialFSHT.spin_frame_angle.(θ, φ, θ′, φ′, Ref(cap))
    f1 .*= cis.(s .* χ)
    Test.@test maximum(abs.(f0 .- f1)) < 1e-10
end

Test.@testset "cached Δ rotation matches Trapani" begin
    Random.seed!(22)
    lmax = 8
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    α, β, γ = 0.2, 0.5, -0.3
    out = similar(alm)
    outc = similar(alm)
    PartialFSHT.rotate_alm!(out, alm, α, β, γ, lmax, lmax)
    L = 2lmax + 1
    dl = zeros(Float64, L, L)
    dlp = zeros(Float64, L, L)
    work = zeros(ComplexF64, L)
    src = zeros(ComplexF64, L)
    cache = PartialFSHT.wigner_Δ_cache(Float64, lmax)
    Test.@test !isempty(cache)
    PartialFSHT.rotate_alm!(outc, alm, α, β, γ, lmax, lmax, dl, dlp, work, 0, src, cache)
    Test.@test maximum(abs.(outc .- out)) < 1e-12
end

Test.@testset "thin d(β) matches wigner_d and Trapani" begin
    Random.seed!(23)
    lmax = 6
    mmax = 3
    α, β, γ = 0.2, 0.5, -0.3
    dcache = PartialFSHT.wigner_thin_d_cache(Float64, lmax, mmax, β)
    Test.@test !isempty(dcache.planes)
    ℓ = 4
    P = dcache.planes[ℓ + 1]
    mlim = min(ℓ, mmax)
    for mp in -mlim:mlim, m in -ℓ:ℓ
        Test.@test P[mp + mlim + 1, m + ℓ + 1] ≈ PartialFSHT.wigner_d(ℓ, mp, m, β) atol = 1e-12
    end

    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    out_t = similar(alm)
    out_d = similar(alm)
    L = 2lmax + 1
    dl = zeros(Float64, L, L)
    dlp = zeros(Float64, L, L)
    work = zeros(ComplexF64, L)
    src = zeros(ComplexF64, L)
    PartialFSHT.rotate_alm!(out_t, alm, α, β, γ, lmax, mmax, dl, dlp, work, 0, src)
    PartialFSHT.rotate_alm!(out_d, alm, α, β, γ, lmax, mmax, dl, dlp, work, 0, src,
                            Matrix{Float64}[], dcache, false)
    Test.@test maximum(abs.(out_d .- out_t)) < 1e-11

    # Euclidean adjoint of the partial map: input |m'|≤mmax, output all m.
    f = similar(alm)
    copyto!(f, out_d)
    adj = similar(alm)
    PartialFSHT.rotate_alm!(adj, f, -γ, -β, -α, lmax, lmax, dl, dlp, work, 0, src,
                            Matrix{Float64}[], dcache, true)
    # <R a, R a> = <a, R* R a>
    Test.@test abs(LinearAlgebra.dot(out_d, out_d) - LinearAlgebra.dot(alm, adj)) < 1e-10
end

Test.@testset "ℓ-recurrence thin d matches packed cache" begin
    Random.seed!(24)
    lmax = 8
    mmax = 4
    α, β, γ = 0.2, 0.5, -0.3
    packed = PartialFSHT.wigner_thin_d_cache(Float64, lmax, mmax, β)
    Test.@test !isempty(packed.planes)
    for ℓ in 0:lmax
        P = packed.planes[ℓ + 1]
        mlim = min(ℓ, mmax)
        for mp in -mlim:mlim, m in -ℓ:ℓ
            Test.@test P[mp + mlim + 1, m + ℓ + 1] ≈ PartialFSHT.wigner_d(ℓ, mp, m, β) atol = 2e-11
        end
    end
    nr, nc = 2mmax + 1, 2lmax + 1
    stream = PartialFSHT.WignerThinD{Float64}(lmax, mmax, β, Matrix{Float64}[],
                                              zeros(nr, nc), zeros(nr, nc), zeros(nr, nc))
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    out_p = similar(alm)
    out_s = similar(alm)
    L = 2lmax + 1
    dl = zeros(Float64, L, L)
    dlp = zeros(Float64, L, L)
    work = zeros(ComplexF64, L)
    src = zeros(ComplexF64, L)
    PartialFSHT.rotate_alm!(out_p, alm, α, β, γ, lmax, mmax, dl, dlp, work, 0, src,
                            Matrix{Float64}[], packed, false)
    PartialFSHT.rotate_alm!(out_s, alm, α, β, γ, lmax, mmax, dl, dlp, work, 0, src,
                            Matrix{Float64}[], stream, false)
    Test.@test maximum(abs.(out_p .- out_s)) < 1e-10
end

Test.@testset "over-budget thin d allocates stream planes" begin
    L, M = 1024, 500
    Test.@test PartialFSHT._thin_d_cache_nbytes(L, M, Float64) > PartialFSHT._WIGNER_THIN_D_BYTES
    d = PartialFSHT.wigner_thin_d_cache(Float64, L, M, 0.4)
    Test.@test isempty(d.planes)
    Test.@test size(d.rec1) == (2M + 1, 2L + 1)
    Test.@test size(d.rec2) == size(d.rec1)
    Test.@test size(d.rec3) == size(d.rec1)
end
