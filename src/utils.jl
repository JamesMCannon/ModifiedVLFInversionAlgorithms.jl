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

function circular_center(vals; period=4, threshold=0.15)
    # Compute resultant length to detect uniform vs concentrated
    θ = vals .* (2π / period)
    R = sqrt(mean(cos, θ)^2 + mean(sin, θ)^2)
    
    if R > threshold
        # Distribution has clear directionality — circular_mean is well-defined
        return circular_mean(vals, period=period)
    else
        # Near-uniform — circular_mean is noise, mode zeroes out members
        # Arithmetic mean gives balanced nonzero perturbations
        # and for near-uniform distributions, perturbation sum ≈ 0
        return mean(vals)
    end
end

"""
    circular_mean(x; period=4)
Compute the circular mean of `x` with given `period`.
Used in the caclulations of mean phase offset values.
"""
function circular_mean(x; period=4)
    θ = 2π .* x ./ period
    μθ = atan(mean(sin.(θ)), mean(cos.(θ)))
    μ = period * μθ / (2π)
    return mod(μ, period)
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

"""
    balanced_circular_diff(vals, center; period=4, tol=1e-9)
 
Compute `circular_diff.(vals, center)` with a rebalancing pass over members
on the exact antipode. The naive formula places all antipodal members at the
same sign (`-period/2` under Julia's `mod`), which biases their contribution
to Σpert when the cluster is large. This function assigns alternating signs
across the antipodal group; odd counts leave one stray `±period/2` whose
sign is chosen to minimize the residual.
 
Antipodes are detected by magnitude (`|p| ≈ period/2`), making the routine
robust to the underlying modulo convention.
"""
function balanced_circular_diff(vals, center; period=4, tol=1e-9)
    perturbs = circular_diff.(vals, center; period)
    half = period / 2
    anti_idx = findall(p -> abs(abs(p) - half) < tol, perturbs)
    for (k, i) in enumerate(anti_idx)
        perturbs[i] = isodd(k) ? +half : -half
    end
    if isodd(length(anti_idx))
        non_anti_sum = sum(perturbs) - half
        if abs(non_anti_sum - half) < abs(non_anti_sum + half)
            perturbs[anti_idx[end]] = -half
        end
    end
    return perturbs
end


"""
    circular_distance(a, b; period=4)
Unsigned shortest arc in [0, period/2].
"""
function circular_distance(a, b; period=4)
    d = mod(a - b, period)
    return min(d, period - d)
end

"""
    resultant_length(x; period=4)
R ∈ [0,1]; near 0 indicates near-uniform data.
"""
function resultant_length(x; period=4)
    θ = 2π .* x ./ period
    return sqrt(mean(sin.(θ))^2 + mean(cos.(θ))^2)
end

"""
    concentration_at(c, x; period=4)
Σ cos(2π(xᵢ-c)/P); higher ⇒ data more clustered around c.
"""
function concentration_at(c, x; period=4)
    return sum(cos.(2π .* (x .- c) ./ period))
end

"""
    zero_sum_centers(x; period=4, tol=1e-9)

Return all interior zero-sum centers: values c such that Σᵢ d(c, xᵢ) = 0,
where d is the signed circular difference in (-P/2, P/2].

S(c) = Σᵢ d(c, xᵢ) is a sawtooth function with slope -n on linear segments
and upward jumps at antipodes. This function evaluates S at each segment's
midpoint, solves for the zero linearly, and keeps crossings strictly interior
to the segment (boundary zeros at antipodes are excluded).
"""
function zero_sum_centers(x; period=4, tol=1e-9)
    n = length(x)
    P = period
    signed_diff(c, xi) = mod(xi - c + P/2, P) - P/2
    S(c) = sum(signed_diff(c, xi) for xi in x)

    antipodes_all = sort(mod.(x .+ P/2, P))
    antipodes = Float64[]
    for a in antipodes_all
        if isempty(antipodes) || !isapprox(a, antipodes[end]; atol=tol)
            push!(antipodes, a)
        end
    end
    m = length(antipodes)

    solutions = Float64[]
    for k in 1:m
        a_start = antipodes[k]
        a_end = antipodes[mod1(k+1, m)]
        ℓ = mod(a_end - a_start, P)
        ℓ == 0 && (ℓ = P)

        c_mid = mod(a_start + ℓ/2, P)
        s_mid = S(c_mid)
        offset_from_start = ℓ/2 + s_mid / n

        # Strictly interior
        if offset_from_start > tol && offset_from_start < ℓ - tol
            push!(solutions, mod(a_start + offset_from_start, P))
        end
    end

    sort!(solutions)
    unique_sols = Float64[]
    for s in solutions
        if isempty(unique_sols) || !isapprox(s, unique_sols[end]; atol=tol)
            push!(unique_sols, s)
        end
    end
    return unique_sols
end

"""
    robust_zero_sum_center(x; period=4, anchor=nothing, tol=1e-9)

Return a single "best" zero-sum center for the dataset `x`.

Selection rule:
  1. If no interior zero-sum center exists: return `anchor` if provided,
     else return `circular_mean(x)`.
  2. If one exists: return it.
  3. If multiple exist: return the one closest (in circular distance) to
     `anchor` if provided, else the one maximizing concentration.
"""
function robust_zero_sum_center(x; period=4, anchor=nothing, tol=1e-9)
    candidates = zero_sum_centers(x; period=period, tol=tol)

    if isempty(candidates)
        return anchor !== nothing ? mod(float(anchor), period) : circular_mean(x; period=period)
    end

    length(candidates) == 1 && return candidates[1]

    if anchor !== nothing
        _, idx = findmin(c -> circular_distance(c, anchor; period=period), candidates)
    else
        _, idx = findmax(c -> concentration_at(c, x; period=period), candidates)
    end
    return candidates[idx]
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
    σ²_model = var(yb_raw)  # per path, per iteration
    σ²_eff = σ² + σ²_model
    for Bϕ in 0:period-1
        offset_rad = Bϕ * quarter
        ℓ[Bϕ+1] = invN_log + logsumexp(
            -0.5 * phasediff(y_phase_path, yb_e + offset_rad)^2 / σ²_eff
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