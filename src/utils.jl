"""
    pathname(p)

Return path name string for (transmitter, receiver) path tuple `p`.
"""
pathname(p) = p[1].name*"-"*p[2].name


"""
    rebuildpaths()

Given a vector of `(Transmitter, Receiver)` propagation paths used in the scenarios,
modify the power of the Transmitters using the values found in the NamedTuple tx_pwrs.
tx_pwrs should be a slice of a KeyedArray and indexed by "pwrs = :calsign", ie tx_pwrs(pwrs = :NLK) 
would return the desired new power for the NLK transmitter.
"""
function rebuildpaths(paths, tx_pwrs)
    #TODO Consider moving this to the RunLETKF code
    revised_paths = map(paths) do (tx, rx)
        new_tx = LongwaveModePropagator.Transmitter{VerticalDipole}(
            tx.name, tx.latitude, tx.longitude, tx.antenna, tx.frequency,
            tx_pwrs(pwrs = Symbol(tx.name))
        )
        (new_tx, rx)
    end
    return revised_paths
end

# ─────────────────────────────────────────────────────────────────────────────
# Observation-vector field layout
#
# The observation vector is a stack of per-path blocks in the order given by
# `datatypes`, a Tuple of field Symbols drawn from the canonical set
# (:amp, :phase, :s2, :s3):
#
#     [ field₁(path 1..npaths); field₂(path 1..npaths); ... ]
#
# `R` (diagonal observation-error variance) shares this layout. All LETKF update
# functions slice R and stack Y/Δ through these helpers so the layout is defined
# in exactly one place.
#
# Fields listed in `PHASE_FIELDS` are circular quantities: ensemble means use
# `circular_phase_stats` and residuals use `phasediff`. All other fields are
# ordinary linear observables. (:s2, :s3) are Cartesian coordinates of the
# complex component ratio Hx/Hy and are deliberately linear — the polar pair
# (γ, Δψ) is singular as |Hx/Hy| → 0, which is the operating regime on most
# quasi-TM-dominated paths.
# ─────────────────────────────────────────────────────────────────────────────

const PHASE_FIELDS = (:phase,)

"""
    is_phase_field(f::Symbol) → Bool

Whether observable field `f` is a circular (phase-like) quantity requiring
`circular_phase_stats` / `phasediff` treatment.
"""
is_phase_field(f::Symbol) = f in PHASE_FIELDS

"""
    fieldindex(f, datatypes) → Int

Position of field `f` within the `datatypes` Tuple. Errors if absent.
"""
function fieldindex(f::Symbol, datatypes)
    k = findfirst(==(f), datatypes)
    isnothing(k) && throw(ArgumentError("field $f not in datatypes $datatypes"))
    return k
end

"""
    fieldrange(f, datatypes, npaths) → UnitRange{Int}

Index range of field `f`'s per-path block within the stacked observation vector
(and within `R`) for the layout defined by `datatypes`.
"""
function fieldrange(f::Symbol, datatypes, npaths::Int)
    k = fieldindex(f, datatypes)
    return ((k - 1)*npaths + 1):(k*npaths)
end

"""
    stack_R(Rslice::KeyedArray, datatypes) → Vector{Float64}

Flatten a `(field × path)` KeyedArray slice of the observation-error variance
(one epoch of the per-path, per-epoch `R`) into the stacked vector layout
consumed by the LETKF update functions. Fields are stacked in `datatypes` order;
within each field, paths follow the slice's `path` axis order.
"""
function stack_R(Rslice::KeyedArray, datatypes)
    return reduce(vcat,
        (Vector{Float64}(parent(parent(Rslice(field=f)))) for f in datatypes))
end

"""
    circular_phase_stats(φ) → (ybar, Y)

Anchored circular mean and exactly-centered perturbations for a phase
ensemble `φ` (radians, any branch). `ref` is the resultant-vector mean;
deviations are wrapped about it via `phasediff` and re-centered so that
`sum(Y) == 0` exactly (the ETKF assumes zero-mean columns). Replaces the
arithmetic `mean`/`phasediff`-against-mean pair, which is branch-sensitive
for offset-multimodal ensembles.
"""
function circular_phase_stats(φ::AbstractVector)
    C = mean(cos, φ)
    S = mean(sin, φ)
    ref = atan(S, C)
    d = phasediff.(φ, ref)   # wrapped deviations in (−π, π]
    m = mean(d)
    return ref + m, d .- m
end

"""
    phasediff(a, b; deg=false)

Compute the smallest angle `a - b` in radians if `deg=false`, otherwise degrees.
"""
function phasediff(a, b; deg=false)
    if isnan(a) || isnan(b)
        # possible Julia bug: mod2pi(NaN) returns a number
        return NaN
    end
    
    if deg
        a, b = deg2rad(a), deg2rad(b)
    end

    d = mod2pi(a) - mod2pi(b)
    d = mod2pi(d + π) - π

    if deg
        d = rad2deg(d)
    end

    return d
end

"""
    circular_diff(b, a; period=4)
Compute the circular difference `b - a` with given `period`.
Used in the phase offsets in the RX offset state vector.
TODO : unify with `phasediff`?
"""
function circular_diff(b, a; period=4)
    return mod(b - a + period/2, period) - period/2
end

# ─────────────────────────────────────────────────────────────────────────────
# Categorical RX-offset inference helpers
#
# These support the alternate `rx_method=:categorical` path in runletkf, which
# treats `rx_phi_offset` as a per-path categorical Bϕ ∈ {0,1,2,3} representing
# the MSK 90° demodulation ambiguity rather than as a continuous LETKF state.
# Pure functions only — no dependency on filter loop state.
# ─────────────────────────────────────────────────────────────────────────────

"""
    logsumexp(x)

Numerically stable `log(sum(exp.(x)))` for any iterable `x`. Two-pass: one to
find the max, one to accumulate. Generator-safe — does not require `x` to be
indexable. Returns `-Inf` if all elements are `-Inf`.
"""
function logsumexp(x)
    m = -Inf
    for xi in x
        xi > m && (m = xi)
    end
    isfinite(m) || return m
    s = 0.0
    for xi in x
        s += exp(xi - m)
    end
    return m + log(s)
end

"""
    rx_phi_loglikelihood(yb_phase_path::AbstractVector, y_phase_path::Real, σ²; period=4)
        → Vector{Float64} of length `period`

For a single path, given the ensemble of modeled phases `yb_phase_path`
(length `ens_size`, *without* any rx offset baked in — see note), the scalar
observed phase `y_phase_path`, and observation variance `σ²`, return the
log-likelihood of each candidate offset `Bϕ ∈ 0:period-1` after marginalizing
over the ensemble.

The marginalization is

    ℓ(Bϕ) = log( (1/N) · Σ_e exp[ -½ · phasediff(y, yb_e + Bϕ·π/2)² / σ² ] )

i.e. logsumexp over members of the per-member Gaussian log-likelihood, minus
log(N). The minus-log(N) is a `Bϕ`-independent constant that drops out of the
posterior after normalization, so it can safely be omitted; we keep it so the
returned numbers are interpretable as honest log-likelihoods.

# Note on `yb_phase_path`
This expects `yb` *without* the current ensemble's rx-offset contribution mixed
in — i.e. the raw forward-model output for this path. If the calling site has
`yb` with offsets already added (as `ensemble_model!` does in the categorical
path), it must subtract them before calling this function. See
`categorical_rx_update!` for the canonical usage.
"""
function rx_phi_loglikelihood(yb_phase_path, y_phase_path, σ²; period=4)
    ℓ = Vector{Float64}(undef, period)
    N = length(yb_phase_path)
    invN_log = -log(N)
    quarter = π / (period / 2)  # = π/2 for period=4
    #σ²_model = var(yb_phase_path)  # per path, per iteration
    #σ²_eff = σ² + σ²_model
    for Bϕ in 0:period-1
        offset_rad = Bϕ * quarter
        ℓ[Bϕ+1] = invN_log + logsumexp(
            -0.5 * phasediff(y_phase_path, yb_e + offset_rad)^2 / σ²
            for yb_e in yb_phase_path
        )
    end
    return ℓ
end

"""
    rx_phi_sample(log_post_path, n, rng; period=4)
        → Vector{Int} of length `n`, values in `0:period-1`

Sample `n` integer offsets independently from the categorical posterior whose
unnormalized log-probabilities are `log_post_path` (length `period`). Used by
the categorical-path `ensemble_model!` to assign a per-member offset that
honestly represents the current uncertainty in `k_p`.
"""
function rx_phi_sample(log_post_path, n, rng; period=4)
    m = maximum(log_post_path)
    w = exp.(log_post_path .- m)
    w ./= sum(w)
    return [sample(rng, 0:period-1, Weights(w)) for _ in 1:n]
end

"""
    rx_phi_posterior(log_post_path) → Vector{Float64}

Normalize unnormalized log-posterior `log_post_path` to a probability vector
summing to 1. Convenience for diagnostics and heatmap generation.
"""
function rx_phi_posterior(log_post_path)
    m = maximum(log_post_path)
    w = exp.(log_post_path .- m)
    return w ./ sum(w)
end

"""
    sample_rx_offsets!(offset_matrix, rx_log_post, rng;
                       commit_threshold=1.0, period=4) → offset_matrix

Fill `offset_matrix::AbstractMatrix{<:Real}` of size `(npaths, ens_size)` with
per-(path, ens) integer offsets drawn from the per-path categorical posterior
encoded by `rx_log_post::KeyedArray(path, Bϕ)`. Paths whose normalized posterior
maximum exceeds `commit_threshold` deterministically receive the MAP Bϕ for every
ensemble member; all other paths sample independently per member.

Mirrors the per-path sampling logic embedded in the categorical-path
`ensemble_model!` so both the forward-model offset assignment and the
post-update `posterior_resample_correct!` call share one implementation.
"""
function sample_rx_offsets!(offset_matrix::AbstractMatrix, rx_log_post, rng;
                            commit_threshold=1.0, period=4)
    npaths, ens_size = size(offset_matrix)
    @assert length(rx_log_post.path) == npaths "sample_rx_offsets!: path count mismatch ($(length(rx_log_post.path)) vs $npaths)"
    for (n, p) in enumerate(rx_log_post.path)
        log_post_p   = collect(rx_log_post(path=p))
        post_p       = rx_phi_posterior(log_post_p)
        max_p, Bϕ_map = findmax(post_p)
        if max_p ≥ commit_threshold
            offset_matrix[n, :] .= Bϕ_map - 1     # findmax is 1-based; offsets are 0-based
        else
            offset_matrix[n, :] .= rx_phi_sample(log_post_p, ens_size, rng; period=period)
        end
    end
    return offset_matrix
end

"""
    strip(m::KeyedArray)
    strip(m::NamedDimsArray)

Remove named dims and axis keys from `m`, returning a view of the underlying array.
"""
Base.strip(m::KeyedArray) = AxisKeys.keyless(AxisKeys.unname(m))
Base.strip(m::NamedDimsArray) = AxisKeys.unname(m)

"""
    l2norm(r)

Compute the *squared* L2 norm, ``||r||₂² = r₁² + r₂² + … + rₙ²``.

This would normalize the sum of squared residuals, which is what oocurs in the least squares
problem.
"""
function l2norm(r)
    return sum(abs2, r)
end

"""
    l1norm(r)

Compute the L1 norm ``||r||₁ = |r₁| + |r₂| + … + |r₃|``.
"""
function l1norm(r)
    return sum(abs, r)
end

"""
    hubernorm(r, ϵ)

Compute the Huber norm, which is the L2 norm squared between `-ϵ` and `ϵ` and the L1 norm
outside these bounds.

Guitton and Symes 2003 Robust inversion...
"""
function hubernorm(r, ϵ)
    M(x) = abs(x) <= ϵ ? abs2(x)/(2*ϵ) : abs(x) - ϵ/2

    return sum(M, r)
end

"""
    pseudohubernorm(r, ϵ)

Compute the pseudo-Huber norm, which is a smooth approximation of [`hubernorm`](@ref),
approximately the L2 norm squared between `-ϵ` and `ϵ` and the L1 norm outside these bounds.

Hartley and Zisserman, 2004, Multiple View Geometry in Computer Vision
"""
function pseudohubernorm(r, ϵ)
    M(x) = ϵ^2*(sqrt(1 + (x/ϵ)^2) - 1)

    return sum(M, r)
end

"""
    tikhonov_gradient(itp, m, λh, λb; localizationfcn=nothing, step=100e3)

Compute Tikhonov regularization of the gradient of model `m` with ``h′`` scaled by `λh` and
``β`` by `λb`.

If `m` is a `KeyedArray`, then it is transformed to a vector where the first half is ``h′``
and the second half is ``β``.
"""
function tikhonov_gradient(itp, m, λh, λb; localizationfcn=nothing, step=100e3)
    (minx, maxx), (miny, maxy) = extrema(itp.coords; dims=2)
    x_grid = range(minx, maxx; step)
    y_grid = range(miny, maxy; step)

    npts = size(itp.coords, 2)

    h_grid = dense_grid(itp, m[1:npts], x_grid, y_grid)
    b_grid = dense_grid(itp, m[npts+1:end], x_grid, y_grid)

    if !isnothing(localizationfcn)
        trans = Proj.Transformation(itp.projection, wgs84())
        xy_grid = densify(x_grid, y_grid)
        lonlats = trans.(parent(parent(xy_grid)))  # undo reshape reinterpret to get vector of tuples
        localization = localizationfcn(lonlats)
        h_grid[.!localization] .= NaN
        b_grid[.!localization] .= NaN
    end

    h_gy, h_gx = diff(h_grid; dims=1), diff(h_grid; dims=2)
    b_gy, b_gx = diff(b_grid; dims=1), diff(b_grid; dims=2)

    h_gy, h_gx = filter(!isnan, h_gy), filter(!isnan, h_gx)
    b_gy, b_gx = filter(!isnan, b_gy), filter(!isnan, b_gx)

    return λh*(norm(h_gx, 2) + norm(h_gy, 2)) + λb*(norm(b_gx, 2) + norm(b_gy, 2))
end
tikhonov_gradient(itp, m::KeyedArray, λh, λb; localizationfcn=nothing, step=100e3) =
    tikhonov_gradient(itp, [filter(!isnan, m(:h)); filter(!isnan, m(:b))], λh, λb; localizationfcn, step)

"""
    totalvariation(itp, m, μh, μb, αh, αb; localizationfcn=nothing, step=100e3)

Compute total variation regularization of model `m` with regularization parameter `μ` and a
small stabilization term `α` such that
```
J(m) = μ ||∇m||₁ = μ Σ √( mx² + my² + α² )
```

The optional `localizationfcn` should be a function of `lonlats` that returns a `Bool` mask
that identifies which states are localized.

`step` is the step size in the fine grid on which the states are interpolated before
computing the gradient.
"""
function totalvariation(itp, m, μh, μb, αh, αb; localizationfcn=nothing, step=100e3)
    (minx, maxx), (miny, maxy) = extrema(itp.coords; dims=2)
    x_grid = range(minx, maxx; step)
    y_grid = range(miny, maxy; step)

    npts = size(itp.coords, 2)

    h_grid = dense_grid(itp, m[1:npts], x_grid, y_grid)
    b_grid = dense_grid(itp, m[npts+1:end], x_grid, y_grid)

    if !isnothing(localizationfcn)
        trans = Proj.Transformation(itp.projection, wgs84())
        xy_grid = densify(x_grid, y_grid)
        lonlats = trans.(parent(parent(xy_grid)))  # undo reshape reinterpret to get vector of tuples
        localization = localizationfcn(permutedims(lonlats))
        h_grid[.!localization] .= NaN
        b_grid[.!localization] .= NaN
    end

    h_gy, h_gx = imgradients(h_grid, KernelFactors.prewitt)
    b_gy, b_gx = imgradients(b_grid, KernelFactors.prewitt)

    h_gy, h_gx = filter(!isnan, h_gy), filter(!isnan, h_gx)
    b_gy, b_gx = filter(!isnan, b_gy), filter(!isnan, b_gx)
    @assert size(h_gy) == size(h_gx) == size(b_gy) == size(b_gx)

    αh² = αh^2
    αb² = αb^2

    h_total = 0.0
    b_total = 0.0
    for i in eachindex(h_gy)
        h_total += sqrt(h_gy[i]^2 + h_gx[i]^2 + αh²)
        b_total += sqrt(b_gy[i]^2 + b_gx[i]^2 + αb²)
    end

    return μh*h_total + μb*b_total
end
totalvariation(itp, m::KeyedArray, μh, μb, αh, αb; localizationfcn=nothing, step=100e3) =
    totalvariation(itp, [filter(!isnan, m(:h)); filter(!isnan, m(:b))], μh, μb, αh, αb; localizationfcn, step)