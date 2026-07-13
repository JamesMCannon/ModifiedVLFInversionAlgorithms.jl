# ─────────────────────────────────────────────────────────────────────────────
# Observation-ensemble statistics, generic over the observable field set
#
# yb / ybar / Y all carry a leading :field axis whose keys equal `datatypes`
# (see the layout definition in utils.jl). Phase-like fields (PHASE_FIELDS) use
# circular means and wrapped residuals; all other fields — including the
# Cartesian differential observables :s2 and :s3 — are ordinary linear rows.
# ─────────────────────────────────────────────────────────────────────────────

"""
    obs_ensemble_mean!(ybar, yb, datatypes) → ybar

Overwrite the entries of `ybar` (arithmetic ensemble mean of `yb`) belonging to
circular fields with the anchored circular mean from `circular_phase_stats`,
per path. Linear fields are left at their arithmetic mean.
"""
function obs_ensemble_mean!(ybar, yb, datatypes)
    for f in datatypes
        is_phase_field(f) || continue
        for p in yb.path
            ref_m, _ = circular_phase_stats(parent(parent(yb(field=f, path=p))))
            ybar(field=f, path=p) .= ref_m
        end
    end
    return ybar
end

"""
    obs_perturbations(yb, ybar, datatypes) → Y

Centered measurement perturbations with the same `(field × path × ens)` shape as
`yb`. Circular fields are centered with `phasediff`; linear fields by
subtraction.
"""
function obs_perturbations(yb, ybar, datatypes)
    Y = similar(yb)
    for f in datatypes
        if is_phase_field(f)
            Y(field=f) .= phasediff.(yb(field=f), ybar(field=f))
        else
            Y(field=f) .= yb(field=f) .- ybar(field=f)
        end
    end
    return Y
end

"""
    _innovation(f, y_loc, ybar_loc) → Array

Per-field innovation block `y − ybar` for field `f`, wrapped for circular fields.
"""
function _innovation(f, y_loc, ybar_loc)
    if is_phase_field(f)
        return phasediff.(Array(y_loc(field=f)), Array(ybar_loc(field=f)))
    else
        return Array(y_loc(field=f)) .- Array(ybar_loc(field=f))
    end
end

"""
LETKF_measupdate(H, xb, y, R; ρ=1.1, localization=nothing, datatypes=(:amp, :phase)) → (xa, yb)

LETKF (Local Ensemble Transform Kalman Filter) analysis update applied locally, following
the steps in [^1].

# Arguments

This function is specific to the VLF estimation problem and makes use of `KeyedArray`s from
AxisKeys.jl.

- `H → KeyedArray(yb; field=collect(datatypes), path=pathnames, ens=ens)`:
    Observation model that maps from state space to observation space (``y = H(x) + ϵ``).
- `xb::KeyedArray(xb; field=[:h, :b], y=y, x=x,  ens=ens)`:
    Ensemble matrix of states having size `(nstates, nensemble)`.
    It is assumed the first half of rows are ``h′`` and the second half are ``β``.
- `y::KeyedArray(data; field=..., path=pathnames)`:
    Observations, one row-block per entry of `datatypes` (extra fields such as
    `_noiseless` copies may be present; only `datatypes` entries are consumed).
- `R`: Vector of the diagonal data covariance ``σ²`` in the stacked layout
    `[field₁(paths); field₂(paths); ...]` in `datatypes` order (see `fieldrange`).
- `datatypes`: Tuple of observable field Symbols in canonical order, drawn from
    (:amp, :phase, :s2, :s3).

# References

[^1]: B. R. Hunt, E. J. Kostelich, and I. Szunyogh, “Efficient data assimilation for
spatiotemporal chaos: A local ensemble transform Kalman filter,” Physica D: Nonlinear
Phenomena, vol. 230, no. 1, pp. 112–126, Jun. 2007.
"""
function LETKF_measupdate(H, xb, y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase))

    # 1. Ensemble measurements
    yb = H(xb)

    ybar = mean(yb, dims=:ens)
    obs_ensemble_mean!(ybar, yb, datatypes)

    # 2. Centered measurement perturbations
    Y = obs_perturbations(yb, ybar, datatypes)

    xa = xy_state_update(xb, y, ybar, Y, R;
        ρ=ρ, localization=localization, datatypes=datatypes)

    return xa
end

function LETKF_measupdate(H, xb::NamedTuple, y, R;
        ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase),
        filtertype=:stacked, log10pwr_update=false,
        rng=nothing, η=1.0, commit_threshold=1.0)

    # Categorical RX needs phase data, an RNG for posterior sampling, and a
    # per-iteration rx_phi_offset buffer alongside the persistent log-posterior.
    if haskey(xb, :rx_phi_logpost)
        :phase in datatypes ||
            error("LETKF_measupdate: xb carries :rx_phi_logpost but :phase ∉ datatypes; \
                   categorical RX estimation requires phase observations.")
        isnothing(rng) &&
            error("LETKF_measupdate: xb carries :rx_phi_logpost; a non-nothing `rng` \
                   keyword is required for categorical RX sampling.")
        haskey(xb, :rx_phi_offset) ||
            error("LETKF_measupdate: xb carries :rx_phi_logpost but not :rx_phi_offset; \
                   both are required for the categorical RX update.")
    end

    if filtertype == :stacked
        return LETKF_stacked_update(H, xb, y, R; ρ=ρ, localization=localization,
            datatypes=datatypes, log10pwr_update=log10pwr_update,
            rng=rng, η=η, commit_threshold=commit_threshold)
    elseif filtertype == :dual && (haskey(xb, :tx_pwrs) || haskey(xb, :rx_phi_logpost))
        return LETKF_dual_update(H, xb, y, R; ρ=ρ, localization=localization,
            datatypes=datatypes, log10pwr_update=log10pwr_update,
            rng=rng, η=η, commit_threshold=commit_threshold)
    elseif filtertype == :split && (haskey(xb, :tx_pwrs) || haskey(xb, :rx_phi_logpost))
        return LETKF_split_update(H, xb, y, R; ρ=ρ, localization=localization,
            datatypes=datatypes, log10pwr_update=log10pwr_update,
            rng=rng, η=η, commit_threshold=commit_threshold)
    else
        error("Unknown filter type: $filtertype. Currently :stacked, :dual, and :split are implemented.")
    end
end

function LETKF_stacked_update(H, xb::NamedTuple, y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase),
    log10pwr_update=false, rng=nothing, η=1.0, commit_threshold=1.0)

    do_rx = haskey(xb, :rx_phi_logpost)

    # Draw per-member prior offsets so the forward model marginalizes over Bϕ.
    if do_rx
        _draw_prior_rx_offsets!(xb.rx_phi_offset, xb.rx_phi_logpost, rng;
                                commit_threshold=commit_threshold)
    end

    # 1. Ensemble measurements (prior offsets, if any, are baked into yb by ensemble_model!)
    yb = H(xb)
    ybar = mean(yb, dims=:ens)
    obs_ensemble_mean!(ybar, yb, datatypes)

    # 2. Centered measurement perturbations
    Y = obs_perturbations(yb, ybar, datatypes)

    # 3. Update each field if it exists
    updated_fields = NamedTuple()

    if haskey(xb, :xy_state)
        xy_state = xy_state_update(xb.xy_state, y, ybar, Y, R;
                                   ρ=ρ, localization=localization, datatypes=datatypes)
        updated_fields = merge(updated_fields, (; xy_state))
    end

    if haskey(xb, :tx_pwrs)
        if log10pwr_update
            log10_tx_pwrs   = log10.(xb.tx_pwrs)
            log10_tx_pwrs_a = tx_pwrs_update(log10_tx_pwrs, y, ybar, Y, R;
                ρ=ρ, datatypes=datatypes)
            tx_pwrs = 10 .^ log10_tx_pwrs_a
        else
            tx_pwrs = tx_pwrs_update(xb.tx_pwrs, y, ybar, Y, R;
                ρ=ρ, datatypes=datatypes)
        end
        updated_fields = merge(updated_fields, (; tx_pwrs))
    end

    # Categorical RX update. correct_yb=false: yb is left untouched, so the
    # xy_state / tx updates above never see the phase-bias correction.
    if do_rx
        rx_phi_offset, rx_phi_logpost = categorical_rx_measupdate!(
            xb.rx_phi_logpost, yb, xb.rx_phi_offset, y, R, rng;
            η=η, commit_threshold=commit_threshold, correct_yb=false,
            datatypes=datatypes)
        updated_fields = merge(updated_fields, (; rx_phi_offset, rx_phi_logpost))
    end

    return updated_fields
end

function LETKF_dual_update(H, xb::NamedTuple, y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase),
    log10pwr_update=false, rng=nothing, η=1.0, commit_threshold=1.0)

    do_rx = haskey(xb, :rx_phi_logpost)

    if do_rx
        _draw_prior_rx_offsets!(xb.rx_phi_offset, xb.rx_phi_logpost, rng;
                                commit_threshold=commit_threshold)
    end

    # 1. Ensemble measurements
    yb = H(xb)
    ybar = mean(yb, dims=:ens)   # arithmetic; circular fields overwritten below
    obs_ensemble_mean!(ybar, yb, datatypes)

    # 2. Centered measurement perturbations
    Y = obs_perturbations(yb, ybar, datatypes)

    # 3. Update each field if it exists, starting with the bias parameters
    updated_fields = NamedTuple()
    
    if haskey(xb, :tx_pwrs)
        if log10pwr_update
            log10_tx_pwrs = log10.(xb.tx_pwrs)
            log10_tx_pwrs_a = tx_pwrs_update(log10_tx_pwrs, y, ybar, Y, R;
                ρ=ρ, datatypes=datatypes)
            tx_pwrs = 10 .^ log10_tx_pwrs_a
        else
            tx_pwrs = tx_pwrs_update(xb.tx_pwrs, y, ybar, Y, R;
                ρ=ρ, datatypes=datatypes)
        end
        updated_fields = merge(updated_fields, (; tx_pwrs))

        pathnames=y.path
        ## G(b): apply TX power offsets. A TX power change scales the source
        ## moment, shifting the absolute Hy amplitude only; the differential
        ## observables (:s2, :s3) derive from the Hx/Hy ratio and are exactly
        ## invariant to it, so no correction applies to them.
        for e in tx_pwrs.ens
            for tx in tx_pwrs.pwrs
                txpaths = pathnames[startswith.(pathnames, String(tx) * "-")]
                Δpwr_log = log10(tx_pwrs(pwrs=tx, ens=e) / xb.tx_pwrs(pwrs=tx, ens=e))
                yb(field=:amp, ens=e, path=txpaths) .+= Δpwr_log * 10  #10 dB per decade
            end
        end
    end

    # Categorical RX update. correct_yb=true: posterior_resample_correct! shifts
    # yb(:phase) so the xy_state update consumes post-evidence offsets.
    if do_rx
        rx_phi_offset, rx_phi_logpost = categorical_rx_measupdate!(
            xb.rx_phi_logpost, yb, xb.rx_phi_offset, y, R, rng;
            η=η, commit_threshold=commit_threshold, correct_yb=true,
            datatypes=datatypes)
        updated_fields = merge(updated_fields, (; rx_phi_offset, rx_phi_logpost))
    end

    # Recompute Y after applying bias updates to yb
    ybar = mean(yb, dims=:ens)
    obs_ensemble_mean!(ybar, yb, datatypes)

    Y = obs_perturbations(yb, ybar, datatypes)
    
    if haskey(xb, :xy_state)
        xy_state = xy_state_update(xb.xy_state, y, ybar, Y, R;
            ρ=ρ, localization=localization, datatypes=datatypes)
        updated_fields = merge(updated_fields, (; xy_state))
    end

    return updated_fields
end

function LETKF_split_update(H, xb::NamedTuple, y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase),
    log10pwr_update=false, rng=nothing, η=1.0, commit_threshold=1.0)

    # Categorical RX carries no :split_ens dimension, so a split run with no TX
    # state is just a dual run — delegate rather than error.
    if !haskey(xb, :tx_pwrs)
        return LETKF_dual_update(H, xb, y, R; ρ=ρ, localization=localization,
            datatypes=datatypes, log10pwr_update=log10pwr_update,
            rng=rng, η=η, commit_threshold=commit_threshold)
    end

    (:split_ens in dimnames(xb.tx_pwrs)) ||
        error("LETKF_split_update: xb.tx_pwrs must contain a :split_ens dimension.")

    do_rx = haskey(xb, :rx_phi_logpost)

    if do_rx
        _draw_prior_rx_offsets!(xb.rx_phi_offset, xb.rx_phi_logpost, rng;
                                commit_threshold=commit_threshold)
    end

    # 1. Forward model once.
    yb = H(xb)

    updated_fields = NamedTuple()

    # 2a. TX bias: split-ensemble update; bias_only_update! folds the amplitude
    #     correction into yb in place.
    new_tx_pwrs = bias_only_update!(yb, xb.tx_pwrs, y, R;
                                    ρ=ρ, log10pwr_update=log10pwr_update,
                                    datatypes=datatypes)
    updated_fields = merge(updated_fields, (; tx_pwrs=new_tx_pwrs))

    # 2b. RX bias: categorical update on the dual path; correct_yb=true folds the
    #     phase correction into yb in place.
    if do_rx
        rx_phi_offset, rx_phi_logpost = categorical_rx_measupdate!(
            xb.rx_phi_logpost, yb, xb.rx_phi_offset, y, R, rng;
            η=η, commit_threshold=commit_threshold, correct_yb=true,
            datatypes=datatypes)
        updated_fields = merge(updated_fields, (; rx_phi_offset, rx_phi_logpost))
    end

    # 3. xy_state update on the fully bias-corrected yb (no extra forward call).
    if haskey(xb, :xy_state)
        xy_state = xy_only_update(yb, xb.xy_state, y, R;
            ρ=ρ, localization=localization, datatypes=datatypes)
        updated_fields = merge(updated_fields, (; xy_state))
    end

    return updated_fields
end
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Windowed pre-update helpers
# ─────────────────────────────────────────────────────────────────────────────
 
"""
    bias_only_update!(yb, tx_pwrs, y, R; ρ, log10pwr_update, datatypes) → new_tx_pwrs

Perform only the split-ensemble TX-power LETKF update, then immediately fold the
resulting amplitude correction back into `yb` so accumulated bias estimates are
reflected in the ensemble predictions before the next window step.

RX phase-offset estimation is no longer handled here — it is performed
categorically (see [`categorical_rx_measupdate!`]) and carries no `:split_ens`
dimension, so it is independent of the split TX machinery.

# Arguments
- `yb`: ensemble prediction array `(field, path, ens)` — **mutated in-place**.
- `tx_pwrs`: current TX-power prior with dims `(pwrs, ens, split_ens)`.
- `y`: observed data for this window step.
- `R`: diagonal observation-noise variance vector in the stacked `datatypes` layout.
- `datatypes`: observable field layout of `yb`, `y`, and `R`.

Returns `new_tx_pwrs`, the refined TX-power estimate to be used as the prior for
the next window iteration.

# Design notes
- The `yb` correction is the log-power change (in dB) relative to the prior mean,
  applied to the `:amp` rows only — the differential observables are invariant to
  TX power by construction.
- Calling this N times with successive observations accumulates corrections that
  telescope to `log10(mean(tx_N)/mean(tx_0))*10` — identical to a single step
  applied with the final estimate.
"""
function bias_only_update!(yb, tx_pwrs, y, R; ρ=1.1, log10pwr_update=false,
    datatypes::Tuple=(:amp, :phase))

    pathnames = y.path
    npaths = length(pathnames)

    (:split_ens in dimnames(tx_pwrs)) ||
        error("bias_only_update!: tx_pwrs must contain a :split_ens dimension")
    split_ens_size = length(tx_pwrs.split_ens)

    # ── TX power update (amplitude data only) ────────────────────────────────
    new_tx_pwrs = similar(tx_pwrs)
    @showprogress Threads.@threads for e in yb.ens
        split_tx_update!(yb(ens=e), tx_pwrs(ens=e), new_tx_pwrs(ens=e),
                         y, R, ρ, npaths, split_ens_size, pathnames, log10pwr_update,
                         datatypes)
    end

    # Accumulate amplitude correction: Δ = change in mean log-power per TX
    for e in new_tx_pwrs.ens
        for tx in new_tx_pwrs.pwrs
            txpaths  = pathnames[startswith.(pathnames, String(tx) * "-")]
            Δpwr_log = log10(mean(new_tx_pwrs(pwrs=tx, ens=e)) /
                             mean(tx_pwrs(pwrs=tx, ens=e)))
            yb(field=:amp, ens=e, path=txpaths) .+= Δpwr_log * 10  # 10 dB/decade
        end
    end

    return new_tx_pwrs
end
 
"""
    xy_only_update(yb, xy_state, y, R; ρ, localization, datatypes) → xy_state_a
 
Perform only the spatial (xy_state) LETKF update given an already-bias-corrected
ensemble prediction `yb`.  No forward-model call is made.
"""
function xy_only_update(yb, xy_state, y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase))
 
    ybar = mean(yb, dims=:ens)
    obs_ensemble_mean!(ybar, yb, datatypes)

    Y = obs_perturbations(yb, ybar, datatypes)
 
    return xy_state_update(xy_state, y, ybar, Y, R;
        ρ=ρ, localization=localization, datatypes=datatypes)
end
 

function split_tx_update!(yb, tx_pwrs_b, tx_pwrs_a, y, R, ρ, npaths, split_ens_size,
    pathnames, log10pwr_update, datatypes)
    
    @assert Set(dimnames(tx_pwrs_b)) == Set((:pwrs, :split_ens))
    #Necessary for mean in the for loop to calculate what we expect
    fields = collect(axiskeys(yb, :field))
    split_yb = KeyedArray(
        Array{Float64,3}(undef, length(fields), npaths, split_ens_size),
        field = fields,
        path  = pathnames,
        ens   = 1:split_ens_size,
    )

    for ee in split_yb.ens
        split_yb(ens=ee) .= yb #broadcast to split_ens_size
    end

    split_tx_pwrs = KeyedArray(
        fill(NaN,2, split_ens_size), 
        pwrs = tx_pwrs_b.pwrs, 
        ens=1:split_ens_size
        )
    #needed because tx_pwrs_update requires structure with dimesion ens, not split_ens. Currently keeping both so (dual_)tx_pwrs can be mutated for all ensembles.

    for tx in tx_pwrs_b.pwrs
        μ = mean(tx_pwrs_b(pwrs=tx))
        for ee in tx_pwrs_b.split_ens
            txpaths = pathnames[startswith.(pathnames, String(tx) * "-")]
            Δpwr_log = log10(tx_pwrs_b(pwrs=tx, split_ens=ee) / μ)
            #Assumes that for the H(xb), the mean of xb.tx_pwrs was used. This should be specified in the definition of f() passed to H().
            #TX power scales the source moment: it shifts :amp only; (:s2, :s3) are ratio observables and are invariant.
            split_yb(field=:amp, ens=ee, path=txpaths) .+= Δpwr_log * 10  #10 dB per decade

        end
    end

    split_ybar = mean(split_yb, dims=:ens)
    obs_ensemble_mean!(split_ybar, split_yb, datatypes)

    split_Y = obs_perturbations(split_yb, split_ybar, datatypes)

    for tx in tx_pwrs_b.pwrs
        split_tx_pwrs(pwrs=tx) .= strip(tx_pwrs_b(pwrs=tx)) 
    end

    if log10pwr_update
        xnew_amp = tx_pwrs_update(log10.(split_tx_pwrs), y, split_ybar, split_Y, R;
            ρ=ρ, datatypes=datatypes)

        for tx in tx_pwrs_b.pwrs
            tx_pwrs_a(pwrs=tx) .= 10 .^(strip(xnew_amp(pwrs=tx)))
        end
    else
        xnew_amp = tx_pwrs_update(split_tx_pwrs, y, split_ybar, split_Y, R;
            ρ=ρ, datatypes=datatypes)

        for tx in tx_pwrs_b.pwrs
            tx_pwrs_a(pwrs=tx) .= (strip(xnew_amp(pwrs=tx)))
        end
    end
end

"""
    xy_state_update(xy_state, y, ybar, Y, R; ρ=1.1, localization=nothing, datatypes=(:amp, :phase)) → xy_state_a
    Perform LETKF analysis update on only the `xy_state` state variable, given the measurements `y`, mean of the modeled measurements `ybar`, 
    ensemble differences from that mean `Y`, and the observation noise covariance `R`.

`Y` carries a leading `:field` axis with keys equal to `datatypes`; `R` is the
stacked variance vector in the same field-major layout (see `fieldrange`). Per
grid cell, the localized `Y`, `Δ`, and `R` are stacked field-by-field in
`datatypes` order.
"""
function xy_state_update(xy_state, y, ybar, Y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase))

    gridshape = (length(xy_state.y), length(xy_state.x))
    ncells = prod(gridshape)
    npaths = length(y.path)
    ens_size = length(xy_state.ens)
    nfields = length(datatypes)

    length(R) == nfields*npaths || throw(ArgumentError(
        "xy_state_update: length(R) = $(length(R)) does not match length(datatypes)·npaths = $(nfields*npaths)"))

    if !isnothing(localization)
        size(localization) == (ncells, npaths) ||
            throw(ArgumentError("Size of `localization` must be `(ncells, npaths)`"))
    end    

    xy_statebar = mean(xy_state, dims=:ens)
    Xxy_state = xy_state .- xy_statebar

     # 3. Localization, starting with grid
    xy_state_a = similar(xy_state)
    CI = CartesianIndices(gridshape)
    for n in 1:ncells
        yidx, xidx = CI[n][1], CI[n][2]

        # Currently localization is binary (cell is included or not)
        if isnothing(localization)
            loc_mask = trues(npaths)
        else
            loc = view(localization, n, :)
            loc_mask = loc .> 0
            if !any(loc_mask)
                # No measurements in range, nothing to update
                xy_state_a(y=Index(yidx), x=Index(xidx)) .= xy_state(y=Index(yidx), x=Index(xidx))
                continue
            end
        end

        # Localize measurements
        ybar_loc = ybar(path=Index(loc_mask))
        Y_loc = Y(path=Index(loc_mask))
        y_loc = y(path=Index(loc_mask))
        loc_paths = collect(Y_loc.path)

        # Stack fields in datatypes order: [field₁(loc paths); field₂(loc paths); ...]
        Y_stack = KeyedArray(
            reduce(vcat, (Array(Y_loc(field=f)) for f in datatypes));
            path = repeat(loc_paths, nfields),
            ens  = collect(Y_loc.ens))
        R_loc = Diagonal(
            reduce(vcat, (R[fieldrange(f, datatypes, npaths)][loc_mask] for f in datatypes)))

        # 4.
        C = strip(Y_stack)'/R_loc

        # 5.
        # Can apply ρ here if H is linear, or if ρ is close to 1
        Patilde = inv((ens_size - 1)*I/ρ + C*Y_stack)

        # 6.
        # Symmetric square root
        Wa = sqrt((ens_size - 1)*Hermitian(strip(Patilde)))

        # 7.
        Δ = KeyedArray(
            reduce(vcat, (_innovation(f, y_loc, ybar_loc) for f in datatypes));
            path = repeat(loc_paths, nfields),
            ens  = ybar_loc.ens,   # <-- keep OneTo instead of collecting
        )

        wabar = Patilde*C*Δ
        wa = Wa .+ wabar

        # 8.
        xy_statebar_loc = xy_statebar(y=Index(yidx), x=Index(xidx))
        Xxy_state_loc = Xxy_state(y=Index(yidx), x=Index(xidx))

        xy_state_a(y=Index(yidx), x=Index(xidx)) .= Xxy_state_loc*wa .+ xy_statebar_loc
    end
    return xy_state_a
end

"""
    tx_pwrs_update(tx_pwrs, y, ybar, Y, R; ρ=1.1, datatypes=(:amp, :phase)) → tx_pwrs_a
    Perform LETKF analysis update on only the `tx_pwrs` bias offset state variable, given the measurements `y`,
    mean of the modeled measurements `ybar`, ensemble differences from that mean `Y`, and the observation noise covariance `R`.

Uses `:amp` observations only, from paths originating at each transmitter. `R`
follows the stacked `datatypes` layout; the `:amp` block is located via
`fieldrange`.
"""
function tx_pwrs_update(tx_pwrs, y, ybar, Y, R; ρ=1.1, datatypes::Tuple=(:amp, :phase))

    :amp in datatypes || error(
        "tx_pwrs_update: TX power estimation requires :amp observations; datatypes = $datatypes")
    
    if !(:field in dimnames(Y))
        #Stacked/Dual update removes the field dimension, causing breakage here.
        #TODO determine why this behavior changed when first adding the different filtertype functions
        Y = KeyedArray(
            reshape(Y, size(Y, :path), size(Y, :ens), 1);
            path  = axiskeys(Y, :path),
            ens   = axiskeys(Y, :ens),
            field = [:amp]          # singleton dimension
        )
    end

    npaths = length(y.path)
    ens_size = length(tx_pwrs.ens)
    num_txs = length(tx_pwrs.pwrs)

    length(R) == length(datatypes)*npaths || throw(ArgumentError(
        "tx_pwrs_update: length(R) = $(length(R)) does not match length(datatypes)·npaths = $(length(datatypes)*npaths)"))

    amp_range = fieldrange(:amp, datatypes, npaths)

    # 2.
    tx_pwrsbar = mean(tx_pwrs,dims=:ens)
    Xtx_pwrs = tx_pwrs .- tx_pwrsbar

    #For localizing TX power state variable, we consider only amplitude data 
    #from paths that start at the current transmitter
    tx_pwrs_a = similar(tx_pwrs)
    for n in 1:num_txs
        tx_string = String(tx_pwrs.pwrs[n])
        # Currently localization is binary (cell is included or not)
        loc_mask = BitVector()
        loc_mask = BitVector([startswith(s, tx_string[1:3]) for s in y.path])

        # Localize and flatten measurements
        ybar_loc = ybar(path=Index(loc_mask), field=:amp)
        Y_loc = Y(path=Index(loc_mask), field=:amp)
        y_loc = y(path=Index(loc_mask), field=:amp)

        R_loc = @views Diagonal(R[amp_range][loc_mask])
 
        # 4.
        C = strip(Y_loc)'/R_loc

        # 5.
        # Can apply ρ here if H is linear, or if ρ is close to 1
        Patilde = inv((ens_size - 1)*I/ρ + C*Y_loc)

        # 6.
        # Symmetric square root
        Wa = sqrt((ens_size - 1)*Hermitian(strip(Patilde)))

        # 7.
        Δ = y_loc .- ybar_loc
        
        wabar = Patilde*C*Δ
        wa = Wa .+ wabar

        # 8.
        tx_pwrsbar_loc = tx_pwrsbar(pwrs = Symbol(tx_string))
        Xtx_pwrs_loc = Xtx_pwrs(pwrs = Symbol(tx_string), ens = Index(Xtx_pwrs.ens))' # Transpose necessary because Julia flattens 1x3 to (3,)

        tx_pwrs_a(pwrs = Symbol(tx_string)) .= parent(parent(Xtx_pwrs_loc*wa .+ tx_pwrsbar_loc))'
    end

    return tx_pwrs_a
end

# ─────────────────────────────────────────────────────────────────────────────
# Categorical RX-offset update path
#
# Alternative to `rx_phi_update` for the discrete MSK-ambiguity case where
# Bϕ ∈ {0,1,2,3} is constant in time. Maintains a per-path log-posterior and
# accumulates evidence across iterations. See `runletkf` `:categorical` branch.
#
# The Bϕ offset is a property of the (single, synchronized) demodulation trellis:
# it shifts the absolute Hy phase (`:phase`) only. The differential observables
# (:s2, :s3) derive from the inter-channel ratio, in which any trellis offset
# common to the two loop channels cancels, so no categorical machinery touches
# them.
# ─────────────────────────────────────────────────────────────────────────────

"""
    categorical_rx_update!(log_post, yb, current_offsets, y, R;
                           period=4, η=1.0, datatypes=(:amp, :phase))
        → log_post (mutated)

Accumulate Bayesian evidence into the per-path log-posterior `log_post`
(`KeyedArray` with dims `path × Bϕ`, where `Bϕ = 0:period-1`) given the current
ensemble forward-model output `yb`, the per-member offsets that were applied
inside the forward model `current_offsets` (`KeyedArray(path, ens)`), the
observation `y` for this iteration, and the diagonal observation-noise vector
`R`.

# Recovery of offset-free `yb`
`ensemble_model!` adds `current_offsets(path=p, ens=e) * (π/2)` to
`yb(:phase, path=p, ens=e)` before this function sees it. To compute the
likelihood of each candidate `Bϕ`, we need the raw forward-model output. We
recover it on the fly by subtracting `current_offsets * (π/2)` per member.
This avoids a duplicate forward-model call.

# Posterior accumulation
`Bϕ` is constant in time (MSK ambiguity is fixed at receiver lock per
preprocessing assumption), so iteration `t`'s posterior is

    log_post_t(Bϕ) = log_post_{t-1}(Bϕ) + ℓ_t(Bϕ)

where `ℓ_t(Bϕ)` is the marginal log-likelihood from `rx_phi_loglikelihood`.
Normalization is applied lazily — `log_post` accumulates as raw log-likelihoods,
and consumers (`rx_phi_posterior`, `rx_phi_sample`) normalize on read.

# `R` handling
`R` follows the stacked field-major `datatypes` layout; the per-path phase
variances are located via `fieldrange(:phase, datatypes, npaths)`. Each path
uses its own σ².
"""
function categorical_rx_update!(log_post, yb, current_offsets, y, R;
    period=4, η=1.0, datatypes::Tuple=(:amp, :phase))

    npaths = length(yb.path)
    quarter = π / (period / 2)  # = π/2 for period=4

    # Fail loud if path orderings ever diverge across the three KeyedArrays
    @assert axiskeys(yb, :path) == axiskeys(current_offsets, :path) == axiskeys(log_post, :path) ==
            axiskeys(y, :path) "categorical_rx_update!: :path axis mismatch across yb / current_offsets / log_post / y"

    :phase in datatypes || error(
        "categorical_rx_update!: categorical RX estimation requires :phase observations; datatypes = $datatypes")
    length(R) == length(datatypes)*npaths || error(
        "categorical_rx_update!: length(R) = $(length(R)) does not match length(datatypes)·npaths = $(length(datatypes)*npaths)")

    R_phase = view(R, fieldrange(:phase, datatypes, npaths))

    for (n, p) in enumerate(yb.path)
        σ² = R_phase[n]

        # Recover offset-free per-member modeled phases for this path.
        # parent(...) drops the KeyedArray wrapper so positional indexing is unambiguous.
        yb_path     = parent(parent(yb(field=:phase, path=p)))      # length-ens_size Vector{Float64}
        offsets_vec = parent(parent(current_offsets(path=p)))       # length-ens_size Vector{Float64}
        y_phase_p   = only(y(field=:phase, path=p))                 # scalar observation

        ens_size = length(yb_path)
        yb_raw = [yb_path[e] - offsets_vec[e]*quarter for e in 1:ens_size]

        ℓ = rx_phi_loglikelihood(yb_raw, y_phase_p, σ²; period=period)

        # Accumulate (constant Bϕ).
        log_post(path=p) .+= ℓ .* η  # optional learning rate η to temper updates and prevent overconfidence in early iterations
    end

    return log_post
end

"""
    posterior_resample_correct!(yb, rx_phi_offset, rx_log_post, rng;
                                commit_threshold=1.0, period=4) → rx_phi_offset

After `categorical_rx_update!` has folded the current iteration's observation
into `rx_log_post`, draw fresh per-(path, ens) offsets from the *updated*
posterior and apply the implied per-member quarter-turn shift to `yb` in place.
`rx_phi_offset` is overwritten with the freshly drawn values so downstream
recordkeeping reflects exactly what the subsequent `xy_only_update` saw.

Closes the information gap in the categorical branch: before this call, `yb`
was built inside `H_cat!` from samples drawn against `rx_log_post(t=i-1)`, so
the xy_state LETKF would otherwise update against pre-evidence offsets. After
this call, `yb` honors the post-evidence posterior — particularly important on
iterations where the new observation tipped one or more paths above
`commit_threshold`, collapsing their per-member spread to MAP and removing the
corresponding contribution to `Y_phase`.

Only the `:phase` rows of `yb` are shifted: the trellis offset is common-mode
across the two loop channels and cancels in the ratio observables (:s2, :s3).

# Arguments
- `yb`: ensemble prediction `(field, path, ens)` — `:phase` channel mutated.
- `rx_phi_offset`: per-(path, ens) offsets currently baked into `yb`; mutated
  in place to hold the freshly drawn values.
- `rx_log_post`: refined per-path log-posterior at this iteration.
- `rng`: RNG used for per-member sampling on uncertain paths.
- `commit_threshold`: posterior max above which the path commits to MAP.
- `period`: number of categorical levels (4 for MSK).
"""
function posterior_resample_correct!(yb, rx_phi_offset, rx_log_post, rng;
                                     commit_threshold=1.0, period=4)
    @assert axiskeys(yb, :path) == axiskeys(rx_phi_offset, :path) == axiskeys(rx_log_post, :path) "posterior_resample_correct!: :path axis mismatch"
    @assert axiskeys(yb, :ens)  == axiskeys(rx_phi_offset, :ens)  "posterior_resample_correct!: :ens axis mismatch"

    npaths   = length(rx_phi_offset.path)
    ens_size = length(rx_phi_offset.ens)
    quarter  = π / (period / 2)   # = π/2 for period=4

    new_sampled = Matrix{Float64}(undef, npaths, ens_size)
    sample_rx_offsets!(new_sampled, rx_log_post, rng;
                       commit_threshold=commit_threshold, period=period)

    for e in 1:ens_size
        old = parent(parent(rx_phi_offset(ens=e)))   # length-npaths plain Vector
        new = view(new_sampled, :, e)
        ΔBϕ = circular_diff.(new, old; period=period)
        yb(field=:phase, ens=e) .+= ΔBϕ .* quarter
        rx_phi_offset(ens=e) .= new
    end

    return rx_phi_offset
end

"""
    _draw_prior_rx_offsets!(rx_phi_offset, rx_log_post, rng;
                            commit_threshold=1.0, period=4) → rx_phi_offset

Fill `rx_phi_offset` (`KeyedArray(path, ens)`) with per-member integer offsets
drawn from the current per-path categorical posterior `rx_log_post`
(`KeyedArray(path, Bϕ)`). Thin `KeyedArray` wrapper over [`sample_rx_offsets!`].

Called by each `LETKF_*_update` *before* the forward model so that the ensemble
of modeled phases honestly marginalizes over Bϕ — every member is paired with an
independently drawn offset, which the standard `ensemble_model!` then bakes into
`yb(:phase)`.
"""
function _draw_prior_rx_offsets!(rx_phi_offset, rx_log_post, rng;
                                 commit_threshold=1.0, period=4)
    npaths   = length(rx_phi_offset.path)
    ens_size = length(rx_phi_offset.ens)
    sampled  = Matrix{Float64}(undef, npaths, ens_size)
    sample_rx_offsets!(sampled, rx_log_post, rng;
                       commit_threshold=commit_threshold, period=period)
    for e in 1:ens_size
        rx_phi_offset(ens=e) .= view(sampled, :, e)
    end
    return rx_phi_offset
end

"""
    categorical_rx_measupdate!(rx_log_post, yb, rx_phi_offset, y, R, rng;
                               η=1.0, commit_threshold=1.0,
                               correct_yb=true, period=4,
                               datatypes=(:amp, :phase))
        → (rx_phi_offset, rx_log_post)

Single-call categorical RX measurement update for use inside the
`LETKF_*_update` dispatchers — the categorical analogue of the (removed)
`rx_phi_update`.

Two steps:
1. [`categorical_rx_update!`] folds this iteration's phase observation into the
   persistent per-path log-posterior `rx_log_post` (mutated in place).
2. Per-(path, ens) offsets are refreshed from the *updated* posterior.

`correct_yb` selects between the two filter behaviors:
- `true`  (dual / split): [`posterior_resample_correct!`] shifts `yb(:phase)` in
  place so a subsequent `xy_state` update consumes post-evidence offsets.
- `false` (stacked): `yb` is left untouched — the `xy_state` update sees the
  prior `yb` — and only `rx_phi_offset` is refreshed (for saving and the next
  iteration's forward model).

`rx_log_post` is mutated in place; the caller passes the carried-forward
posterior. Returns `(rx_phi_offset, rx_log_post)` for merging into `updated_fields`.
"""
function categorical_rx_measupdate!(rx_log_post, yb, rx_phi_offset, y, R, rng;
                                    η=1.0, commit_threshold=1.0,
                                    correct_yb::Bool=true, period=4,
                                    datatypes::Tuple=(:amp, :phase))
    categorical_rx_update!(rx_log_post, yb, rx_phi_offset, y, R;
        period=period, η=η, datatypes=datatypes)

    if correct_yb
        posterior_resample_correct!(yb, rx_phi_offset, rx_log_post, rng;
                                    commit_threshold=commit_threshold, period=period)
    end

    return rx_phi_offset, rx_log_post
end


"""
    _assign_member!(ym, res, e)

Write a single ensemble member's forward-model result into `ym(ens=e)`.

Two result forms are supported:
- `res::KeyedArray` with a `(field × path)` layout (from [`model_observables`](@ref)):
  each field of `ym` is assigned by key, so a field-axis mismatch fails loudly.
- `res::Tuple` of `(amps, phases)` (from the single-component [`model`](@ref)):
  requires `ym` to carry exactly the `:amp` and `:phase` fields.
"""
function _assign_member!(ym, res::KeyedArray, e)
    for fld in axiskeys(ym, :field)
        ym(field=fld, ens=e) .= res(field=fld)
    end
    return ym
end

function _assign_member!(ym, res::Tuple, e)
    a, p = res
    ym(field=:amp, ens=e) .= a
    ym(field=:phase, ens=e) .= p
    return ym
end

"""
    _wrap_phase_fields!(ym)

Wrap the ensemble of each circular field of `ym` within ±π about a Gaussian fit
to that path's ensemble (see [`modgaussian`](@ref)). Linear fields, including
(:s2, :s3), are untouched.
"""
function _wrap_phase_fields!(ym)
    for fld in axiskeys(ym, :field)
        is_phase_field(fld) || continue
        for pth in ym.path
            ym(field=fld, path=pth) .= modgaussian(ym(field=fld, path=pth))
        end
    end
    return ym
end

"""
    ensemble_model!(ym, f, x)

Run the forward model `f` with `KeyedArray` argument `x` for each member of `x.ens`.

`f` may return either a `(field × path)` KeyedArray (multi-observable forward
model) or an `(amps, phases)` Tuple (single-component forward model); see
[`_assign_member!`](@ref).
"""
function ensemble_model!(ym, f, x)
    @showprogress Threads.@threads for e in x.ens
        _assign_member!(ym, f(x(ens=e)), e)
    end

    # Wrap circular-field ensembles within ±180° about the fit mean
    _wrap_phase_fields!(ym)

    return ym
end

function ensemble_model!(ym, f, x::NamedTuple)
    @showprogress Threads.@threads for e in x.xy_state.ens
        xy_state = x.xy_state(ens=e)

        # Construct the ensemble input dynamically
        ens_state = NamedTuple()
        ens_state = merge(ens_state, (; xy_state))
        if haskey(x, :tx_pwrs)
            tx_pwrs = x.tx_pwrs(ens=e)
            ens_state = merge(ens_state, (; tx_pwrs))
        end

        # Evaluate model
        _assign_member!(ym, f(ens_state), e)

        # Apply receiver phase offsets if present. The Bϕ trellis offset shifts
        # the absolute Hy phase only; the ratio observables (:s2, :s3) are
        # invariant to it and receive no offset.
        if haskey(x, :rx_phi_offset)
            if (:split_ens in dimnames(x.rx_phi_offset))
                offsets = mode.(eachslice(x.rx_phi_offset(ens=e), dims=:path))
                ym(field=:phase, ens=e) .+= offsets .* (π/2)
            else
                ym(field=:phase, ens=e) .+= x.rx_phi_offset(ens=e) .* (π/2) #implicitly, all paths must be in the same order
            end
        end
    end

    # Wrap circular-field ensembles around ±180°
    _wrap_phase_fields!(ym)

    return ym
end

"""
    modgaussian(phases)

Fit a Gaussian distribution to a vector of `phases` in radians and return the phases shifted
such they are wrapped within ±π about the mean of the fit.
"""
function modgaussian(phases)
    μ = fit(Normal{Float64}, phases).μ
    return mod2pi.(phases .- μ .+ π) .+ μ .- π
end