# 2-D NUFFT seam. Direct summation is always available and is the reference the fast backends are
# tested against; FINUFFT and NonuniformFFTs arrive through package extensions.

abstract type AbstractNUFFTHandle end

struct DirectSumNUFFT{T, VT<:AbstractVector{T}} <: AbstractNUFFTHandle
    type::Int
    n1::Int
    n2::Int
    iflag::Int
    ntrans::Int
    modeord::Int
    x::VT
    y::VT
end

function _nufft_makeplan(backend, x, y, type, n_modes, iflag, ntrans, tol; kwargs...)
    return throw(ArgumentError(
        "$(nameof(typeof(backend))) is not available — load its extension (`using FINUFFT` " *
        "or `using NonuniformFFTs`), or omit `nufft` to use direct summation."))
end

function _nufft_setpts! end
function _nufft_exec! end
function _nufft_destroy! end

"""
    DirectSumBackend()

Evaluate the Fourier series by direct summation: `O(M · n₁n₂)`, exact, and the reference the
fast backends are tested against. Never selected by [`AutoNUFFT`](@ref).
"""
struct DirectSumBackend end

"""
    FINUFFTBackend()

FINUFFT guru plans. Needs `using FINUFFT`; `CuArray` nodes route to cuFINUFFT.
"""
struct FINUFFTBackend end

"""
    NonuniformFFTsBackend()

NonuniformFFTs.jl — pure Julia, and device-generic through KernelAbstractions. Needs
`using NonuniformFFTs`.
"""
struct NonuniformFFTsBackend end

"""
    AutoNUFFT()

Pick whichever fast backend is loaded, and error if none is. It will not silently fall back to
[`DirectSumBackend`](@ref): that is a `10⁶`-fold slowdown a caller would not notice until it had
already cost them the run.
"""
struct AutoNUFFT end

function _nufft_makeplan(::DirectSumBackend, x::AbstractVector, y::AbstractVector,
                         type, n_modes, iflag, ntrans, tol; modeord = 1, kwargs...)
    T = float(eltype(x))
    n1, n2 = Int(n_modes[1]), Int(n_modes[2])
    xb = similar(x, T)
    yb = similar(y, T)
    p = DirectSumNUFFT{T, typeof(xb)}(Int(type), n1, n2, Int(iflag), Int(ntrans),
                                     Int(modeord), xb, yb)
    _nufft_setpts!(p, x, y)
    return p
end

_nufft_setpts!(p::DirectSumNUFFT, x, y) = (copyto!(p.x, x); copyto!(p.y, y); p)
_nufft_destroy!(::DirectSumNUFFT) = nothing

@inline function _mode_k(i::Int, n::Int, modeord::Int)
    if modeord == 0
        # CMCL: i=1 is -floor(n/2)
        return i - 1 - (n ÷ 2)
    else
        # FFTW: 0,1,...,floor((n-1)/2), -floor(n/2),...,-1
        k = i - 1
        return k < cld(n, 2) ? k : k - n
    end
end

function _nufft_exec!(p::DirectSumNUFFT{T}, input, output) where {T}
    n1, n2, B = p.n1, p.n2, p.ntrans
    M = length(p.x)
    s = T(p.iflag)
    if p.type == 2
        fill!(output, zero(eltype(output)))
        @inbounds for b in 1:B, j in 1:M
            acc = zero(eltype(output))
            for i2 in 1:n2
                k2 = _mode_k(i2, n2, p.modeord)
                for i1 in 1:n1
                    k1 = _mode_k(i1, n1, p.modeord)
                    c = ndims(input) >= 3 ? input[i1, i2, b] : input[i1, i2]
                    acc += c * cis(s * (k1 * p.x[j] + k2 * p.y[j]))
                end
            end
            if ndims(output) == 1
                output[j] = acc
            else
                output[j, b] = acc
            end
        end
    elseif p.type == 1
        fill!(output, zero(eltype(output)))
        @inbounds for b in 1:B, j in 1:M
            fj = ndims(input) == 1 ? input[j] : input[j, b]
            iszero(fj) && continue
            for i2 in 1:n2
                k2 = _mode_k(i2, n2, p.modeord)
                for i1 in 1:n1
                    k1 = _mode_k(i1, n1, p.modeord)
                    inc = fj * cis(s * (k1 * p.x[j] + k2 * p.y[j]))
                    if ndims(output) >= 3
                        output[i1, i2, b] += inc
                    else
                        output[i1, i2] += inc
                    end
                end
            end
        end
    else
        throw(ArgumentError("DirectSum NUFFT type must be 1 or 2, got $(p.type)"))
    end
    return output
end

function _resolve_nufft(backend)
    return backend
end

function _resolve_nufft(::AutoNUFFT)
    _ext_loaded(:PartialFSHTFINUFFTExt) && return FINUFFTBackend()
    _ext_loaded(:PartialFSHTNonuniformFFTsExt) && return NonuniformFFTsBackend()
    throw(ArgumentError(
        "no fast NUFFT is loaded. Add `using NonuniformFFTs` (pure Julia, also GPU) or " *
        "`using FINUFFT`, or pass `nufft = DirectSumBackend()` to ask for the O(M·n₁n₂) " *
        "reference explicitly."))
end

@inline _ext_loaded(name::Symbol) = Base.get_extension(@__MODULE__, name) !== nothing

_nufft_finalize!(p) = p
