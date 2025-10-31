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

    original_transmitters = unique(tx for (tx, _) in paths)
    receivers = unique(rx for (_, rx) in paths)

    revised_transmitters = [
    LongwaveModePropagator.Transmitter{VerticalDipole}(
        tx.name, tx.latitude, tx.longitude, tx.antenna, tx.frequency,
        tx_pwrs(pwrs = Symbol(tx.name))
    )
    for tx in original_transmitters
    ]

    paths = [(tx, rx) for tx in revised_transmitters for rx in receivers]

    return paths
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
