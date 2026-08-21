"""
    PartialFSHTFINUFFTExt

FINUFFT implementation of the 2-D NUFFT seam, selected by `nufft = FINUFFTBackend()`.
Loaded by `using FINUFFT`. Host coordinates; `CuArray` nodes use `PartialFSHTcuFINUFFTExt`.
"""
module PartialFSHTFINUFFTExt

using PartialFSHT: PartialFSHT
using FINUFFT: FINUFFT

function PartialFSHT._nufft_makeplan(::PartialFSHT.FINUFFTBackend, x::AbstractVector, y::AbstractVector,
                                     type, n_modes, iflag, ntrans, tol; kwargs...)
    p = FINUFFT.finufft_makeplan(type, collect(Int, n_modes), iflag, ntrans, tol; kwargs...)
    FINUFFT.finufft_setpts!(p, PartialFSHT._host(x), PartialFSHT._host(y))
    return PartialFSHT._nufft_finalize!(p)
end

PartialFSHT._nufft_setpts!(p::FINUFFT.finufft_plan, x, y) =
    FINUFFT.finufft_setpts!(p, PartialFSHT._host(x), PartialFSHT._host(y))

PartialFSHT._nufft_exec!(p::FINUFFT.finufft_plan, input, output) =
    FINUFFT.finufft_exec!(p, input, output)

PartialFSHT._nufft_destroy!(p::FINUFFT.finufft_plan) = FINUFFT.finufft_destroy!(p)
PartialFSHT._nufft_finalize!(p::FINUFFT.finufft_plan) = (finalizer(PartialFSHT._nufft_destroy!, p); p)

end
