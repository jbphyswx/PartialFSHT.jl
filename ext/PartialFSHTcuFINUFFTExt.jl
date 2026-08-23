"""
    PartialFSHTcuFINUFFTExt

cuFINUFFT binding for `CuArray` node sets. Other device steps (DFS, FFT) go through
KernelAbstractions / AbstractFFTs. Loaded by `using CUDA` together with `using FINUFFT`.
"""
module PartialFSHTcuFINUFFTExt

using PartialFSHT: PartialFSHT
using FINUFFT: FINUFFT
using CUDA: CUDA

struct CuNUFFTPlan{P}
    plan::P
end

Base.show(io::IO, ::CuNUFFTPlan) = print(io, "CuNUFFTPlan(cuFINUFFT)")

function PartialFSHT._nufft_makeplan(::PartialFSHT.FINUFFTBackend, x::CUDA.CuArray, y,
                                     type, n_modes, iflag, ntrans, tol; kwargs...)
    p = CuNUFFTPlan(FINUFFT.cufinufft_makeplan(type, collect(Int, n_modes), iflag, ntrans, tol; kwargs...))
    PartialFSHT._nufft_setpts!(p, x, y)
    return PartialFSHT._nufft_finalize!(p)
end

PartialFSHT._nufft_setpts!(p::CuNUFFTPlan, x, y) = (FINUFFT.cufinufft_setpts!(p.plan, x, y); p)
PartialFSHT._nufft_exec!(p::CuNUFFTPlan, input, output) = (FINUFFT.cufinufft_exec!(p.plan, input, output); output)
PartialFSHT._nufft_destroy!(p::CuNUFFTPlan) = FINUFFT.cufinufft_destroy!(p.plan)
PartialFSHT._nufft_finalize!(p::CuNUFFTPlan) = (finalizer(q -> FINUFFT.cufinufft_destroy!(q), p.plan); p)

end
