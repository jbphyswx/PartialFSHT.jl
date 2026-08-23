"""
MPI point-decomposition test. Run under a launcher with at least two ranks:

    mpiexec -n 2 julia --project=test test/mpi_ranks.jl

Not part of `runtests.jl`, because a single-rank run would assert nothing about the thing that
matters: that the adjoint sums coefficient contributions across ranks. It is a separate entry
point rather than a conditional skip, so a green suite is never mistaken for coverage.
"""

using MPI: MPI
using PartialFSHT: PartialFSHT
using ComputationalBackends: ComputationalBackends
using LinearAlgebra: LinearAlgebra
using Random: Random
using Test: Test

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)

NRANKS >= 2 || error("this test needs at least two ranks; got $NRANKS")

# Every rank draws the *same* global point set, then keeps its own contiguous slice. That way the
# reference (one serial plan over all the points) is computable on every rank.
Random.seed!(1234)
const L = 64
const M = 600
const θ_all = (20 * π / 180) .* sqrt.(rand(M))
const φ_all = 2π .* rand(M)

lo = div(M * RANK, NRANKS) + 1
hi = div(M * (RANK + 1), NRANKS)
local_idx = lo:hi

alm = PartialFSHT.zeros_alm(L)
Random.seed!(99)
for ℓ in 0:L, m in -ℓ:ℓ
    alm[PartialFSHT.sph_mode(ℓ, m)] = randn(ComplexF64)
end

be = ComputationalBackends.MPIBackend(; comm = COMM)
plan_local = PartialFSHT.Plan(θ_all[local_idx], φ_all[local_idx]; lmax = L,
                              method = PartialFSHT.DirectAlgorithm(), backend = be)
plan_ref = PartialFSHT.Plan(θ_all, φ_all; lmax = L,
                            method = PartialFSHT.DirectAlgorithm())

Test.@testset "MPI rank $RANK of $NRANKS" begin
    # Synthesis is local and needs no communication.
    f_local = plan_local * alm
    f_ref = plan_ref * alm
    Test.@test maximum(abs.(f_local .- f_ref[local_idx])) < 1e-12

    # The adjoint is Allreduced, so every rank must end up with the *global* adjoint.
    f_global = plan_ref * alm
    a_local = plan_local' * f_global[local_idx]
    a_ref = plan_ref' * f_global
    rel = maximum(abs.(a_local .- a_ref)) / maximum(abs.(a_ref))
    Test.@test rel < 1e-12

    # A rank must not get the global answer by accident: without the reduction its own slice
    # alone would differ, so check the partial really is partial.
    plan_serial_slice = PartialFSHT.Plan(θ_all[local_idx], φ_all[local_idx]; lmax = L,
                                         method = PartialFSHT.DirectAlgorithm())
    a_slice = plan_serial_slice' * f_global[local_idx]
    Test.@test maximum(abs.(a_slice .- a_ref)) > 1e-6 * maximum(abs.(a_ref))
end

MPI.Barrier(COMM)
RANK == 0 && println("MPI point decomposition OK across $NRANKS ranks")
MPI.Finalize()
