"""
    PartialFSHTNonuniformFFTsExt

NonuniformFFTs.jl implementation of the 2-D NUFFT seam, selected by
`nufft = NonuniformFFTsBackend()`. Loaded by `using NonuniformFFTs`.
"""
module PartialFSHTNonuniformFFTsExt

using PartialFSHT: PartialFSHT
using NonuniformFFTs: NonuniformFFTs

const _HALFSUPPORT_ERROR = (4.84e-4, 4.45e-6, 4.52e-8, 5.16e-10, 8.76e-12, 2.31e-13, 1.02e-14)

@inline function _halfsupport(tol)
    i = findfirst(<=(tol), _HALFSUPPORT_ERROR)
    return NonuniformFFTs.HalfSupport(isnothing(i) ? length(_HALFSUPPORT_ERROR) + 1 : i + 1)
end

struct NonuniformFFTsPlan{P, V}
    plan::P
    type::Int
    ntrans::Int
    scratch::V
end

function PartialFSHT._nufft_makeplan(::PartialFSHT.NonuniformFFTsBackend, x::AbstractVector, y::AbstractVector,
                                     type, n_modes, iflag, ntrans, tol; dtype = Float64, modeord = 1, kwargs...)
    Ns = (Int(n_modes[1]), Int(n_modes[2]))
    plan = NonuniformFFTs.PlanNUFFT(Complex{dtype}, Ns;
                                    ntransforms = Int(ntrans),
                                    m = _halfsupport(tol),
                                    fftshift = (modeord == 0))
    NonuniformFFTs.set_points!(plan, (x, y))
    scratch = similar(x, Complex{dtype}, length(x) * Int(ntrans))
    return NonuniformFFTsPlan{typeof(plan), typeof(scratch)}(plan, Int(type), Int(ntrans), scratch)
end

PartialFSHT._nufft_setpts!(p::NonuniformFFTsPlan, x, y) = (NonuniformFFTs.set_points!(p.plan, (x, y)); p)

@inline function _as_transforms(v, M, ntrans)
    return ntuple(b -> view(vec(v), ((b - 1) * M + 1):(b * M)), ntrans)
end
@inline function _as_modes(u, n1, n2, ntrans)
    return ntuple(b -> reshape(view(vec(u), ((b - 1) * n1 * n2 + 1):(b * n1 * n2)), n1, n2), ntrans)
end

function PartialFSHT._nufft_exec!(p::NonuniformFFTsPlan, input, output)
    n1, n2 = size(p.plan)
    if p.type == 2
        M = length(output) ÷ p.ntrans
        NonuniformFFTs.exec_type2!(_as_transforms(output, M, p.ntrans), p.plan,
                                   _as_modes(input, n1, n2, p.ntrans))
    else
        M = length(input) ÷ p.ntrans
        NonuniformFFTs.exec_type1!(_as_modes(output, n1, n2, p.ntrans), p.plan,
                                   _as_transforms(input, M, p.ntrans))
    end
    return output
end

PartialFSHT._nufft_destroy!(::NonuniformFFTsPlan) = nothing
PartialFSHT._nufft_finalize!(p::NonuniformFFTsPlan) = p

end
