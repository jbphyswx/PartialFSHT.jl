"""
    PartialFSHTKernelAbstractionsExt

Device-generic DFS doubling/folding and selected-θ associated-Legendre steps as
KernelAbstractions `@kernel`s, dispatched on `GPUArraysCore.AbstractGPUArray`.
Loaded by `using KernelAbstractions`.
"""
module PartialFSHTKernelAbstractionsExt

using PartialFSHT: PartialFSHT
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
using GPUArraysCore: GPUArraysCore

@inline function _sync(backend)
    applicable(KernelAbstractions.synchronize, backend) && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _dfs_double_kern!(F̃, @Const(F), Nθ, Nφ, half, sgn)
    i, j, b = @index(Global, NTuple)
    if i <= Nθ
        @inbounds F̃[i, j, b] = F[i, j, b]
    else
        ii = i - Nθ
        js = mod1(j + half, Nφ)
        @inbounds F̃[i, j, b] = sgn * F[Nθ + 1 - ii, js, b]
    end
end

function PartialFSHT.dfs_double!(F̃::GPUArraysCore.AbstractGPUArray, F::GPUArraysCore.AbstractGPUArray)
    return PartialFSHT.dfs_double!(F̃, F, 0)
end

function PartialFSHT.dfs_double!(F̃::GPUArraysCore.AbstractGPUArray, F::GPUArraysCore.AbstractGPUArray, s::Integer)
    Nθ, Nφ, B = size(F, 1), size(F, 2), size(F, 3)
    @assert size(F̃, 1) == 2Nθ && size(F̃, 2) == Nφ && size(F̃, 3) == B
    backend = KernelAbstractions.get_backend(F̃)
    sgn = iseven(Int(s)) ? one(eltype(F)) : -one(eltype(F))
    _dfs_double_kern!(backend)(F̃, F, Nθ, Nφ, Nφ ÷ 2, sgn; ndrange = (2Nθ, Nφ, B))
    _sync(backend)
    return F̃
end

@kernel function _dfs_fold_kern!(F, @Const(F̃), Nθ, Nφ, half, sgn)
    i, j, b = @index(Global, NTuple)
    im = 2Nθ + 1 - i
    js = mod1(j - half, Nφ)
    @inbounds F[i, j, b] = F̃[i, j, b] + sgn * F̃[im, js, b]
end

function PartialFSHT.dfs_fold!(F::GPUArraysCore.AbstractGPUArray, F̃::GPUArraysCore.AbstractGPUArray)
    return PartialFSHT.dfs_fold!(F, F̃, 0)
end

function PartialFSHT.dfs_fold!(F::GPUArraysCore.AbstractGPUArray, F̃::GPUArraysCore.AbstractGPUArray, s::Integer)
    Nθ, Nφ, B = size(F, 1), size(F, 2), size(F, 3)
    @assert size(F̃, 1) == 2Nθ && size(F̃, 2) == Nφ && size(F̃, 3) == B
    backend = KernelAbstractions.get_backend(F)
    sgn = iseven(Int(s)) ? one(eltype(F)) : -one(eltype(F))
    _dfs_fold_kern!(backend)(F, F̃, Nθ, Nφ, Nφ ÷ 2, sgn; ndrange = (Nθ, Nφ, B))
    _sync(backend)
    return F
end

@kernel function _scale_legendre_kern!(out, @Const(x), @Const(P), c)
    i = @index(Global)
    @inbounds out[i] = c * x[i] * P[i]
end

@kernel function _legendre_step_kern!(out, @Const(x), @Const(pb), @Const(pa), a, b)
    i = @index(Global)
    @inbounds out[i] = a * x[i] * pb[i] - b * pa[i]
end

@kernel function _scale_inplace_kern!(P, @Const(s), fac)
    i = @index(Global)
    @inbounds P[i] *= fac * s[i]
end

function PartialFSHT._scale_legendre_device!(out::GPUArraysCore.AbstractGPUArray, x, P, c)
    backend = KernelAbstractions.get_backend(out)
    _scale_legendre_kern!(backend)(out, x, P, c; ndrange = length(out))
    _sync(backend)
    return out
end

function PartialFSHT._legendre_step_device!(out::GPUArraysCore.AbstractGPUArray, x, pb, pa, a, b)
    backend = KernelAbstractions.get_backend(out)
    _legendre_step_kern!(backend)(out, x, pb, pa, a, b; ndrange = length(out))
    _sync(backend)
    return out
end

function PartialFSHT._scale_inplace_device!(P::GPUArraysCore.AbstractGPUArray, s, fac)
    backend = KernelAbstractions.get_backend(P)
    _scale_inplace_kern!(backend)(P, s, fac; ndrange = length(P))
    _sync(backend)
    return P
end

end
