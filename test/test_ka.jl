Test.@testset "KernelAbstractions DFS (JLArray)" begin
    Test.@test JLArrays.JLArray <: GPUArraysCore.AbstractGPUArray
    Random.seed!(101)
    Nθ, Nφ, B = 7, 13, 2
    F = randn(ComplexF64, Nθ, Nφ, B)
    F̃_cpu = zeros(ComplexF64, 2Nθ, Nφ, B)
    PartialFSHT.dfs_double!(F̃_cpu, F)
    Fj = JLArrays.JLArray(F)
    F̃j = JLArrays.JLArray(zeros(ComplexF64, 2Nθ, Nφ, B))
    PartialFSHT.dfs_double!(F̃j, Fj)
    Test.@test Array(F̃j) ≈ F̃_cpu

    G_cpu = zeros(ComplexF64, Nθ, Nφ, B)
    PartialFSHT.dfs_fold!(G_cpu, F̃_cpu)
    Gj = JLArrays.JLArray(zeros(ComplexF64, Nθ, Nφ, B))
    PartialFSHT.dfs_fold!(Gj, F̃j)
    Test.@test Array(Gj) ≈ G_cpu
end

Test.@testset "KernelAbstractions DFS spin-1 (JLArray)" begin
    Random.seed!(103)
    Nθ, Nφ, B = 7, 13, 2
    F = randn(ComplexF64, Nθ, Nφ, B)
    F̃_cpu = zeros(ComplexF64, 2Nθ, Nφ, B)
    PartialFSHT.dfs_double!(F̃_cpu, F, 1)
    Fj = JLArrays.JLArray(F)
    F̃j = JLArrays.JLArray(zeros(ComplexF64, 2Nθ, Nφ, B))
    PartialFSHT.dfs_double!(F̃j, Fj, 1)
    Test.@test Array(F̃j) ≈ F̃_cpu
end

Test.@testset "Direct selected-θ on JLArray" begin
    Random.seed!(102)
    lmax = 4
    n = 18
    θ = 0.35 .* rand(n)
    φ = 2π .* rand(n)
    planC = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
    θj = JLArrays.JLArray(θ)
    φj = JLArrays.JLArray(φ)
    planG = PartialFSHT.Plan(θj, φj; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    fC = planC * alm
    fG = planG * alm
    Test.@test Array(fG) ≈ fC atol = 1e-12
    fR = randn(ComplexF64, n)
    aC = planC' * fR
    aG = planG' * JLArrays.JLArray(fR)
    Test.@test Array(aG) ≈ aC atol = 1e-11
end
