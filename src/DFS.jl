"""
Double Fourier Sphere periodization (Merilees 1973; Townsend & Olver 2015).
"""

"""
    dfs_double!(F̃, F) -> F̃
    dfs_double!(F̃, F, s) -> F̃

`F` is `(Nθ, Nφ[, B])`; `F̃` is `(2Nθ, Nφ[, B])`. Southern hemisphere is the
pole reflection with a `φ+π` cyclic shift. Spin-`s` fields pick up `(-1)^s`
(Carron–Reinecke).
"""
dfs_double!(F̃::AbstractArray, F::AbstractArray) = dfs_double!(F̃, F, 0)

function dfs_double!(F̃::AbstractArray, F::AbstractArray, s::Integer)
    Nθ, Nφ = size(F, 1), size(F, 2)
    B = size(F, 3)
    @assert size(F̃, 1) == 2Nθ && size(F̃, 2) == Nφ && size(F̃, 3) == B
    half = Nφ ÷ 2
    sgn = iseven(Int(s)) ? one(eltype(F)) : -one(eltype(F))
    @inbounds for b in 1:B
        for j in 1:Nφ, i in 1:Nθ
            F̃[i, j, b] = F[i, j, b]
        end
        for j in 1:Nφ
            js = mod1(j + half, Nφ)
            for i in 1:Nθ
                F̃[Nθ + i, j, b] = sgn * F[Nθ + 1 - i, js, b]
            end
        end
    end
    return F̃
end

"""
    dfs_fold!(F, F̃) -> F
    dfs_fold!(F, F̃, s) -> F

Adjoint of [`dfs_double!`](@ref). Complex fields keep both parts (no `real`).
"""
dfs_fold!(F::AbstractArray, F̃::AbstractArray) = dfs_fold!(F, F̃, 0)

function dfs_fold!(F::AbstractArray, F̃::AbstractArray, s::Integer)
    Nθ, Nφ = size(F, 1), size(F, 2)
    B = size(F, 3)
    @assert size(F̃, 1) == 2Nθ && size(F̃, 2) == Nφ && size(F̃, 3) == B
    half = Nφ ÷ 2
    sgn = iseven(Int(s)) ? one(eltype(F)) : -one(eltype(F))
    @inbounds for b in 1:B
        for j in 1:Nφ
            js = mod1(j - half, Nφ)
            for i in 1:Nθ
                F[i, j, b] = F̃[i, j, b] + sgn * F̃[2Nθ + 1 - i, js, b]
            end
        end
    end
    return F
end

"""
    dfs_grid_coords(Nθ, Nφ) -> (θ, φ)

Equiangular CC nodes: `θ_i = π(i-1/2)/Nθ`, `φ_j = 2π(j-1)/Nφ`.
"""
function dfs_grid_coords(Nθ::Integer, Nφ::Integer)
    θ = [π * (i - 0.5) / Nθ for i in 1:Int(Nθ)]
    φ = [2π * (j - 1) / Nφ for j in 1:Int(Nφ)]
    return θ, φ
end
