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

const SAFEMESH = SafeMesh()
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