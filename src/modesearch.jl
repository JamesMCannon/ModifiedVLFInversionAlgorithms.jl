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
below `log(floatmax) - margin` across Re θ.
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

Roots of `inner`. Each root is Newton-polished on the modal
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

"""[`SafeMesh`](@ref) with [`PolishedMesh`](@ref) roots."""
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

"""
    SearchRegion(re, im; owner=(-Inf, Inf), budget=(2000, 20000), seed=1)

Rectangle `re` × `im` (degrees) searched with `maxadaptivenodes, maxnodes = budget`. An `im`
lower bound of `-Inf` is replaced by [`safebottom`](@ref). Roots are kept when
`owner[1] ≤ Re θ < owner[2]` (degrees).
"""
struct SearchRegion
    re::Tuple{Float64,Float64}
    im::Tuple{Float64,Float64}
    owner::Tuple{Float64,Float64}
    maxadaptivenodes::Int
    maxnodes::Int
    seed::Int
end
SearchRegion(re, im; owner=(-Inf, Inf), budget=(2000, 20000), seed=1) =
    SearchRegion(Float64.(re), Float64.(im), Float64.(owner), budget[1], budget[2], seed)

"""
    TiledMesh(regions; tol=3e-5, remax=89.9, dedupe=2e-4, rpdedupe=true, quiet=true)

Raw roots of each of `regions`, searched separately with RootsAndPoles at `tol` and one task,
restricted to Im θ < 0 and Re θ ≤ `remax` (degrees). Roots within `dedupe` rad of an earlier
one are merged. `rpdedupe` is passed to RootsAndPoles and `quiet` silences its logging. Each
search seeds the task-local RNG with the region's `seed`. Wrap in [`PolishedMesh`](@ref) to
polish the roots.
"""
Base.@kwdef struct TiledMesh
    regions::Vector{SearchRegion}
    tol::Float64 = 3e-5
    remax::Float64 = 89.9
    dedupe::Float64 = 2e-4
    rpdedupe::Bool = true
    quiet::Bool = true
end
TiledMesh(regions::Vector{SearchRegion}; kw...) = TiledMesh(; regions, kw...)

function _regionpoints(r::SearchRegion, frequency; params)
    lo = isinf(r.im[1]) ? safebottom(frequency; params) : r.im[1]
    re, im = deg2rad.(r.re), (deg2rad(lo), deg2rad(r.im[2]))
    return ComplexF64[complex(re[1], im[1]), complex(re[1], im[2]), complex(re[2], im[1]), complex(re[2], im[2])]
end

function _searchregion(me::LMP.ModeEquation, r::SearchRegion, mesh::TiledMesh; params)
    pts = _regionpoints(r, me.frequency; params)
    fp = FinderParams(tol=mesh.tol, maxadaptivenodes=r.maxadaptivenodes, maxnodes=r.maxnodes, numtasks=1)
    p = LMPParams(params; finderparams=fp)
    f = θ -> LMP.solvemodalequation(θ, me; params=p)
    run() = (Random.seed!(r.seed);
             LMP.RootsAndPoles.rootsandpoles(f, LMP.RootsAndPoles.ComplexMesh(pts), fp; dedupe=mesh.rpdedupe))
    saved = copy(Random.default_rng())
    raw = try
        first(mesh.quiet ? with_logger(run, NullLogger()) : run())
    finally
        copy!(Random.default_rng(), saved)
    end
    lo, hi = deg2rad(r.owner[1]), deg2rad(r.owner[2])
    return ComplexF64[x for x in raw if imag(x) < 0 && real(x) ≤ deg2rad(mesh.remax) && lo ≤ real(x) < hi]
end

function LMP.findmodes(me::LMP.ModeEquation, mesh::TiledMesh; params=LMPParams())
    raw = ComplexF64[]
    for r in mesh.regions, x in _searchregion(me, r, mesh; params)
        any(y -> abs(y - x) < mesh.dedupe, raw) || push!(raw, x)
    end
    return LMP.EigenAngle.(raw)
end

"""
    avidregions() → Vector{SearchRegion}

F1r (degrees; adaptive nodes, node cap):
- L    0–51° × −8…0°, owning Re < 50°, 640                S1   49.9–65° × −6.5…+1°, 1321
- S2   64–82° × −4…+0.3°, 1298                            G4R2 64–82° × −6…−3.5°, 272
- T    81–89.9° × −3…+0.3°, 2250                          H84  83.9–84.9° × −1.15…−0.2°, 1000
- H82  82.3–82.65° × −0.95…−0.65°, 250                    H79  79.2–80.1° × −2.75…−0.75°, 500
- O    85.8–89.4° × −2.2…+0.2°, 400
The H boxes cover close root–pole pairs that recur at fixed θ and O those near the tip; H84 reaches
−0.2° to cover the close pairs near 84.8°, −0.35° that T alone misses. Nothing is searched below
−8° (Re < 50°), −6.5° (49.9–65°), −6° (64–82°) or −3° (82–89.9°); on validation 8 the roots left
out there put no path outside tolerance. Known miss: validation 8 S275 NML-CH, a 0.32 dB root
3e-5 rad from a pole at 84.12°, −0.76° (0.35 dB in Hy).
"""
avidregions() = [SearchRegion((0.0, 51.0), (-8.0, 0.0); owner=(-Inf, 50.0), budget=(640, 20000)),
                 SearchRegion((49.9, 65.0), (-6.5, 1.0); budget=(1321, 40000)),
                 SearchRegion((64.0, 82.0), (-4.0, 0.3); budget=(1298, 40000)),
                 SearchRegion((64.0, 82.0), (-6.0, -3.5); budget=(272, 20000)),
                 SearchRegion((81.0, 89.9), (-3.0, 0.3); budget=(2250, 25000)),
                 SearchRegion((83.9, 84.9), (-1.15, -0.2); budget=(1000, 20000)),
                 SearchRegion((82.3, 82.65), (-0.95, -0.65); budget=(250, 20000)),
                 SearchRegion((79.2, 80.1), (-2.75, -0.75); budget=(500, 20000)),
                 SearchRegion((85.8, 89.4), (-2.2, 0.2); budget=(400, 20000))]

"""
    avidmesh(; kw...) → PolishedMesh{TiledMesh}

F1r: [`avidregions`](@ref) in a [`TiledMesh`](@ref), whose fields `kw` overrides, with
[`PolishedMesh`](@ref) roots. Validated for the AVID paths at 24.8 and 25.2 kHz with equal-length
50 km segments. On validation 8 (7040 path-scenarios on a
held-out scenario seed) it met Hy ≤ 0.15 dB and 1.5°, s2 and s3 ≤ 0.019 against a dense truth on
7039, against 7037 for the fifteen-region reference mesh, at 0.615 of that mesh's search time.
"""
avidmesh(; kw...) = PolishedMesh(TiledMesh(avidregions(); kw...))

"""Default mesh of the forward models: [`avidmesh`](@ref)."""
const AVIDMESH = avidmesh()

const WAVEFIELD_SPACING = 110e3/2048
const WAVEFIELD_INTEGRATION = LMP.IntegrationParams(solver=LMP.Vern7(lazy=false), tolerance=10^-6.5, maxiters=10_000_000)

"""
    pathparams(H; spacing=WAVEFIELD_SPACING, wavefieldintegrationparams=WAVEFIELD_INTEGRATION, kw...) → LMPParams

`LMPParams` with `topheight = H`, `wavefieldheights` from `H` to 0 at about `spacing` (m), the
LMP default density of 2049 heights over 110 km, and mode-conversion wavefields integrated with
non-lazy Vern7 at 10^-6.5 (LMP requires a non-lazy wavefield solver); `kw` sets other `LMPParams`
fields.

Against a consensus of nine tight integrator settings, Vern7 at 10^-6.5 was outside the observation
tolerance on 11/369 tolerance-ladder cases and 11/219 held-out cases (LMP's default Tsit5 at 1e-6:
139/369 and 47/219), at 1.02–1.04 of the default's field time. Tighter tolerances were no better:
below about 1e-9 the error grows erratically. Tsit5 and Vern6 at 5e-7 to 1e-7 were statistically
indistinguishable from it; Vern8 at those tolerances failed on 40–48% of cases and should not be
used. Lazy and non-lazy Vern7 gave bit-identical fields on all 220 held-out cases.
"""
pathparams(H; spacing=WAVEFIELD_SPACING, wavefieldintegrationparams=WAVEFIELD_INTEGRATION, kw...) =
    LMPParams(; topheight=H, wavefieldheights=range(H, 0; length=round(Int, H/spacing) + 1),
              wavefieldintegrationparams, kw...)

"""
    densityheight(hprime, beta, N) → m

Height where the Wait profile 1.43e13 exp(−0.15h′) exp((β − 0.15)(z − h′)) m⁻³ reaches `N`
(h′ in km, β in km⁻¹).
"""
densityheight(h, b, N) = b ≤ 0.15 ? Inf : 1e3*(h + log(N/(1.43e13*exp(-0.15h)))/(b - 0.15))

"""
    waittopheight(hprimes, betas; N=1e9, margin=2e3, limits=(65e3, 200e3)) → m

Path top height: the highest `densityheight` over the segments plus `margin` (m), rounded up to
1 km and clamped to `limits`.
"""
function waittopheight(hprimes, betas; N=1e9, margin=2e3, limits=(65e3, 200e3))
    z = maximum(densityheight(h, b, N) for (h, b) in zip(hprimes, betas)) + margin
    return clamp(1e3*ceil(z/1e3), limits...)
end

"""
    avidrun(input; mesh=AVIDMESH, N=1e9, margin=2e3, limits=(65e3, 200e3), unwrap=true, kw...)

`LMP.buildrun` with `mesh` and [`pathparams`](@ref) at the path's [`waittopheight`](@ref). A
`BatchInput` is run input by input, so each path gets its own top height; `kw` is passed to
[`pathparams`](@ref) (e.g. `wavefieldintegrationparams`) and sets other `LMPParams` fields.
"""
function avidrun(s::LMP.ExponentialInput; mesh=AVIDMESH, N=1e9, margin=2e3, limits=(65e3, 200e3), unwrap=true, kw...)
    H = waittopheight(s.hprimes, s.betas; N, margin, limits)
    return LMP.buildrun(s; mesh, unwrap, params=pathparams(H; kw...))
end

function avidrun(s::LMP.BatchInput; kw...)
    LMP.consistentfieldcomponents(s) ||
        throw(ArgumentError("all inputs in a BatchInput must sample the same fieldcomponent"))
    batch = LMP.BatchOutput{LMP.batchoutputtype(s)}()
    batch.name, batch.description, batch.datetime = s.name, s.description, Dates.now()
    for input in s.inputs
        push!(batch.outputs, avidrun(input; kw...))
    end
    return batch
end

"""
True when `LMP.dmodalequation` is finite at a singular `A = Rg*R - I`, which requires its adjugate
form (LongwaveModePropagator 352b5da or later). Root polishing relies on it.
"""
function dmodal_finite()
    S = LMP.StaticArrays
    R = S.SMatrix{2,2,ComplexF64}(2, 1, 1, 2)
    dR = S.SMatrix{2,2,ComplexF64}(1, 0, 0, 1)
    return isfinite(LMP.dmodalequation(R, dR, S.SDiagonal(1.0 + 0im, 1.0 + 0im), S.SDiagonal(0.0 + 0im, 0.0 + 0im)))
end

__init__() = dmodal_finite() ||
    error("LongwaveModePropagator.dmodalequation is NaN at a singular A; use LongwaveModePropagator 352b5da or later")