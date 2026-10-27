#=
Device parity, on real hardware.

    julia --project=gpu -e 'using Pkg; Pkg.instantiate()'
    julia --project=gpu gpu/runtests.jl

The main suite tests the three KernelAbstractions kernels against the host on `JLArrays`, which is
enough to catch a wrong kernel but not enough to certify a device *transform*: `JLArrays` provides
no FFT, so the FFT and NUFFT stages of a stretched apply are never reached there. That is the gap
this file closes, and it is the only part of the package that cannot be validated without a device.

Nothing here is CUDA-specific beyond obtaining the array type. The ring passes and the DFS go
through KernelAbstractions, the FFTs through AbstractFFTs, and the NUFFT through cuFINUFFT — so a
device array needs no vendor code in `src/`, which is why this tests *parity* rather than a separate
GPU implementation.

The precondition is asserted rather than skipped: run without a device and this fails, instead of
reporting a pass that checked nothing.
=#

using Test: Test
using PartialFSHT: PartialFSHT
using ComputationalBackends: ComputationalBackends
using KernelAbstractions: KernelAbstractions
using GPUArraysCore: GPUArraysCore
using CUDA: CUDA
using FINUFFT: FINUFFT
using LinearAlgebra: LinearAlgebra
using Printf: Printf
using Random: Random

const PF = PartialFSHT
const CB = ComputationalBackends

Test.@testset "GPU parity" begin
    Test.@test CUDA.functional()
    CUDA.functional() || error("no functional CUDA device: this entry point exists to test one")
    Printf.@printf("device: %s\n", string(CUDA.name(CUDA.device())))

    Random.seed!(11)

    Test.@testset "scalar ring pass, bit-identical forward" begin
        for L in (128, 512)
            θh = sort(0.02 .+ 0.5 .* rand(64))
            almh = zeros(ComplexF64, PF.coeff_size(L)..., 1)
            for ℓ in 0:L, m in (-ℓ):ℓ
                almh[PF.sph_mode(ℓ, m), 1] = randn(ComplexF64)
            end
            engh = PF.RingLegendre(θh, L, 1e-8; ntrans = 1)
            Gh = zeros(ComplexF64, length(θh), 2L + 1, 1)
            PF.rings_from_alm!(Gh, almh, engh)

            engd = PF.RingLegendre(CUDA.cu(θh), L, 1e-8; ntrans = 1)
            Gd = CUDA.zeros(ComplexF64, length(θh), 2L + 1, 1)
            PF.rings_from_alm!(Gd, CUDA.cu(almh), engd)
            # one thread per (ring, order), each owning its output: no race, so exact
            Test.@test Array(Gd) == Gh

            # the adjoint scatters through atomic adds, so its summation order is not fixed
            ah = zeros(ComplexF64, PF.coeff_size(L)..., 1)
            PF.alm_from_rings!(ah, Gh, engh)
            ad = CUDA.zeros(ComplexF64, PF.coeff_size(L)..., 1)
            PF.alm_from_rings!(ad, CUDA.cu(Gh), engd)
            Test.@test maximum(abs.(Array(ad) .- ah)) <= 1e-12 * maximum(abs.(ah))
        end
    end

    Test.@testset "spin ring pass, bit-identical forward" begin
        for (L, s) in ((128, 2), (256, -1))
            θh = sort(0.05 .+ 0.5 .* rand(48))
            almh = zeros(ComplexF64, PF.coeff_size(L)..., 1)
            for ℓ in 0:L, m in (-ℓ):ℓ
                almh[PF.spin_coeff_index(ℓ, m, L), 1] = randn(ComplexF64)
            end
            engh = PF.SpinRings(θh, L, s, 1e-8; ntrans = 1)
            Gh = zeros(ComplexF64, length(θh), 2L + 1, 1)
            PF.spin_rings_from_alm!(Gh, almh, engh)

            engd = PF.SpinRings(CUDA.cu(θh), L, s, 1e-8; ntrans = 1)
            Gd = CUDA.zeros(ComplexF64, length(θh), 2L + 1, 1)
            PF.spin_rings_from_alm!(Gd, CUDA.cu(almh), engd)
            Test.@test Array(Gd) == Gh
        end
    end

    Test.@testset "end to end: the stages JLArrays cannot reach" begin
        # This is the whole reason the file exists. A stretched apply on a device array runs the
        # ring pass and DFS as kernels, the two FFTs through AbstractFFTs' CUDA plans, and the
        # interpolation through cuFINUFFT. None of those last three is exercised by the main suite.
        for L in (128, 512)
            θstar = 20 * π / 180
            M = 4 * round(Int, (L * θstar)^2)
            θh = θstar .* sqrt.(rand(M))
            φh = 2π .* rand(M)
            almh = PF.zeros_alm(L)
            for ℓ in 0:L, m in (-ℓ):ℓ
                almh[PF.sph_mode(ℓ, m)] = randn(ComplexF64) / (1 + ℓ)
            end

            ph = PF.Plan(θh, φh; lmax = L, tol = 1e-8, method = PF.StretchAlgorithm())
            fh = ph * almh

            pd = PF.Plan(CUDA.cu(θh), CUDA.cu(φh); lmax = L, tol = 1e-8,
                         method = PF.StretchAlgorithm())
            almd = CUDA.cu(almh)
            fd = pd * almd
            rel = maximum(abs.(Array(fd) .- fh)) / maximum(abs.(fh))
            Printf.@printf("  lmax %4d, M %8d: device vs host rel = %.3e\n", L, M, rel)
            # the FFT and NUFFT stages are separate implementations on device, so this is a
            # tolerance comparison rather than a bitwise one
            Test.@test rel < 1e-7

            # and the adjoint identity has to survive the round trip on device
            gd = CUDA.cu(randn(ComplexF64, M))
            ad = pd' * gd
            lhs = LinearAlgebra.dot(gd, fd)
            rhs = LinearAlgebra.dot(ad, almd)
            Test.@test abs(lhs - rhs) <= 1e-9 * abs(lhs)
        end
    end

    Test.@testset "Float32 on device" begin
        L = 256
        θstar = 20 * π / 180
        M = 4 * round(Int, (L * θstar)^2)
        θh = Float32.(θstar .* sqrt.(rand(M)))
        φh = Float32.(2π .* rand(M))
        almh = PF.zeros_alm(ComplexF32, L)
        for ℓ in 0:L, m in (-ℓ):ℓ
            almh[PF.sph_mode(ℓ, m)] = ComplexF32(randn(ComplexF64) / (1 + ℓ))
        end
        pd = PF.Plan(CUDA.cu(θh), CUDA.cu(φh); lmax = L, tol = 1e-5,
                     method = PF.StretchAlgorithm())
        fd = pd * CUDA.cu(almh)
        Test.@test eltype(fd) === ComplexF32
        ph = PF.Plan(θh, φh; lmax = L, tol = 1e-5, method = PF.StretchAlgorithm())
        fh = ph * almh
        Test.@test maximum(abs.(Array(fd) .- fh)) / maximum(abs.(fh)) < 1e-4
    end
end
