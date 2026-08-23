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

end
