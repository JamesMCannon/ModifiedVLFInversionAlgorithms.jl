"""
    SafeMesh(; re=(0.0, 89.99), imtop=0.0, seed=1, margin=20.0, step=0.1)

LMP's default search box (degrees) with its bottom at [`safebottom`](@ref) for the mode
equation's frequency. Each search seeds the task-local RNG with `seed`, so the triangulation
and the roots found do not depend on which task runs the search.
"""
Base.@kwdef struct SafeMesh
    re::Tuple{Float64,Float64} = (0.0, 89.99)
    imtop::Float64 = 0.0
    seed::Int = 1
    margin::Float64 = 20.0
    step::Float64 = 0.1
end

const SAFEBOTTOM = Dict{NTuple{4,Float64},Float64}()
const SAFEBOTTOM_LOCK = ReentrantLock()

"""
    safebottom(frequency; params=LMPParams(), margin=20.0, step=0.1) → degrees

Lowest Im θ of LMP's default search box for which the height-gain function exponent stays
below `log(floatmax) - margin` across Re θ (as `AVIDModes.safebottom`).
"""
function safebottom(frequency::LMP.Frequency; params=LMPParams(), margin=20.0, step=0.1)
    key = (frequency.f, params.earthradius, margin, step)
    lock(SAFEBOTTOM_LOCK) do
        get!(SAFEBOTTOM, key) do
            mesh = LMP.defaultmesh(frequency)
            remin, remax = extrema(real, mesh)
            imin = minimum(imag, mesh)
            limit = log(floatmax(Float64)) - margin
            K = LMP.pow23(frequency.k/(2/params.earthradius))
            growth(θ) = (2/3)*imag((K*LMP.referencetoground(LMP.EigenAngle(θ); params).cos²θ)^(3/2))
            res = range(remin, remax; length=901)
            for imθ in range(0.0, imin; step=-deg2rad(step))
                any(re -> growth(complex(re, imθ)) ≥ limit, res) && return rad2deg(imθ + deg2rad(step))
            end
            return rad2deg(imin)
        end
    end
end

function LMP.findmodes(me::LMP.ModeEquation, mesh::SafeMesh; params=LMPParams())
    lo = safebottom(me.frequency; params, margin=mesh.margin, step=mesh.step)
    re1, re2 = deg2rad.(mesh.re)
    im1, im2 = deg2rad(lo), deg2rad(mesh.imtop)
    pts = ComplexF64[complex(re1, im1), complex(re1, im2), complex(re2, im1), complex(re2, im2)]
    f = θ -> LMP.solvemodalequation(θ, me; params)
    saved = copy(Random.default_rng())
    roots = try
        Random.seed!(mesh.seed)
        first(LMP.RootsAndPoles.rootsandpoles(f, LMP.RootsAndPoles.ComplexMesh(pts), params.finderparams))
    finally
        copy!(Random.default_rng(), saved)
    end
    return LMP.EigenAngle.(roots)
end

"""Step-1 search settings: tol 3e-5, LMPParams' default budget, `numtasks = nthreads()`."""
lmpparams() = LMPParams(finderparams=FinderParams(tol=3e-5, maxadaptivenodes=8000,
    maxnodes=40000, numtasks=Threads.nthreads()))

"""
    PolishedMesh(inner; maxit=8, xtol=1e-8, maxstep=5e-4, retry_tolerance=1e-9, retry_maxit=20,
                 filter_radius=3e-4, filter_points=64, guard90=1e-3, dedupe=1e-7)

Roots of `inner`, finalized as in `AVIDModes.finalize`. Each root is Newton-polished on the modal
equation (to a step below `xtol` rad, moving at most `maxstep` rad), then again with Vern7 at
`retry_tolerance` if that fails. A root that still does not converge is kept unpolished only if
the winding number of the modal function on a `filter_radius` circle about it is +1. Roots within
`guard90` rad of π/2, and duplicates within `dedupe` rad, are dropped. Every outcome other than
convergence is reported with `@info`.
"""
Base.@kwdef struct PolishedMesh{M}
    inner::M
    maxit::Int = 8
    xtol::Float64 = 1e-8
    maxstep::Float64 = 5e-4
    retry_tolerance::Float64 = 1e-9
    retry_maxit::Int = 20
    filter_radius::Float64 = 3e-4
    filter_points::Int = 64
    guard90::Float64 = 1e-3
    dedupe::Float64 = 1e-7
end
PolishedMesh(inner; kw...) = PolishedMesh(; inner, kw...)

"""Default mesh of the forward models: [`SafeMesh`](@ref) with [`PolishedMesh`](@ref) roots."""
const SAFEMESH = PolishedMesh(SafeMesh())

function _polish(me, θ0, mesh::PolishedMesh; params, maxit)
    θ = θ0
    try
        for _ in 1:maxit
            dF, R, Rg = LMP.solvedmodalequation(θ, me; params)
            δ = LMP.modalequation(R, Rg) / dF
            isfinite(δ) || return θ0, false
            θ -= δ
            abs(θ - θ0) > mesh.maxstep && return θ0, false
            abs(δ) < mesh.xtol && return θ, true
        end
    catch err
        err isa InterruptException && rethrow()
    end
    return θ0, false
end

function _winding(me, θ0, r; n, params)
    z = θ0 .+ r .* cis.(range(0, 2π; length=n+1))
    f = [LMP.solvemodalequation(θ, me; params) for θ in z]
    return sum(angle.(f[2:end] ./ f[1:end-1])) / 2π
end

function LMP.findmodes(me::LMP.ModeEquation, mesh::PolishedMesh; params=LMPParams())
    raw = ComplexF64[ea.θ for ea in LMP.findmodes(me, mesh.inner; params)]
    kept = ComplexF64[]
    tight = nothing
    for θ0 in raw
        θ, ok = _polish(me, θ0, mesh; params, maxit=mesh.maxit)
        if !ok
            tight === nothing && (tight = LMPParams(params; integrationparams=LMP.IntegrationParams(
                solver=LMP.Vern7(), tolerance=mesh.retry_tolerance)))
            θ, ok = _polish(me, θ0, mesh; params=tight, maxit=mesh.retry_maxit)
        end
        if !ok
            w = _winding(me, θ0, mesh.filter_radius; n=mesh.filter_points, params)
            if !(isfinite(w) && round(Int, w) == 1)
                @info "Unconverged root dropped by the winding filter" θ=rad2deg(θ0) winding=w
                continue
            end
            @info "Unconverged root kept unpolished" θ=rad2deg(θ0) winding=w
        end
        if abs(θ - π/2) < mesh.guard90
            @info "Root within guard90 of 90° dropped" θ=rad2deg(θ)
        elseif any(x -> abs(x - θ) < mesh.dedupe, kept)
            @info "Duplicate root dropped" θ=rad2deg(θ)
        else
            push!(kept, θ)
        end
    end
    return LMP.EigenAngle.(kept)
end