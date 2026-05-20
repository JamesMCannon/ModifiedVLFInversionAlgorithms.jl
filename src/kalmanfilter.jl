"""
LETKF_measupdate(H, xb, y, R; ρ=1.1, localization=nothing, datatypes=(:amp, :phase)) → (xa, yb)

LETKF (Local Ensemble Transform Kalman Filter) analysis update applied locally, following
the steps in [^1].

# Arguments

This function is specific to the VLF estimation problem and makes use of `KeyedArray`s from
AxisKeys.jl.

- `H → KeyedArray(yb; field=[:amp, :phase], path=pathnames, ens=ens)`:
    Observation model that maps from state space to observation space (``y = H(x) + ϵ``).
- `xb::KeyedArray(xb; field=[:h, :b], y=y, x=x,  ens=ens)`:
    Ensemble matrix of states having size `(nstates, nensemble)`.
    It is assumed the first half of rows are ``h′`` and the second half are ``β``.
- `y::KeyedArray(data; field=[:amp, :phase], path=pathnames)`:
    Stacked vector of observations `[amps...; phases...]`.
- `R`: Vector of the diagonal data covariance matrix ``σ²``.
- `y_grid`: 

# References

[^1]: B. R. Hunt, E. J. Kostelich, and I. Szunyogh, “Efficient data assimilation for
spatiotemporal chaos: A local ensemble transform Kalman filter,” Physica D: Nonlinear
Phenomena, vol. 230, no. 1, pp. 112–126, Jun. 2007.
"""
function LETKF_measupdate(H, xb, y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase))

    # Make sure xb, yb, and y are correct KeyedArrays
    # xb = KeyedArray(xb; field=[:h, :b], y=xb.y, x=xb.x, ens=xb.ens)
    # y = KeyedArray(y; field=[:amp, :phase], path=y.path)
    
    # 1.
    yb = H(xb)
    # yb = KeyedArray(yb; field=[:amp, :phase], path=y.path, ens=xb.ens)
    
    ybar = mean(yb, dims=:ens)

    if :amp in datatypes && :phase in datatypes
        Y = similar(yb)
        Y(:amp) .= yb(:amp) .- ybar(:amp)
        Y(:phase) .= phasediff.(yb(:phase), ybar(:phase))
    elseif :amp in datatypes
        Y = yb(:amp) .- ybar(:amp)
    elseif :phase in datatypes
        Y = phasediff.(yb(:phase), ybar(:phase))
    end

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

    # 2. Centered measurement perturbations
    if :amp in datatypes && :phase in datatypes
        Y = similar(yb)
        Y(:amp) .= yb(:amp) .- ybar(:amp)
        Y(:phase) .= phasediff.(yb(:phase), ybar(:phase))
    elseif :amp in datatypes
        Y = yb(:amp) .- ybar(:amp)
    elseif :phase in datatypes
        Y = phasediff.(yb(:phase), ybar(:phase))
    else
        error("Unknown datatypes: $datatypes")
    end

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
            log10_tx_pwrs_a = tx_pwrs_update(log10_tx_pwrs, y, ybar, Y, R; ρ=ρ)
            tx_pwrs = 10 .^ log10_tx_pwrs_a
        else
            tx_pwrs = tx_pwrs_update(xb.tx_pwrs, y, ybar, Y, R; ρ=ρ)
        end
        updated_fields = merge(updated_fields, (; tx_pwrs))
    end

    # Categorical RX update. correct_yb=false: yb is left untouched, so the
    # xy_state / tx updates above never see the phase-bias correction.
    if do_rx
        rx_phi_offset, rx_phi_logpost = categorical_rx_measupdate!(
            xb.rx_phi_logpost, yb, xb.rx_phi_offset, y, R, rng;
            η=η, commit_threshold=commit_threshold, correct_yb=false)
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
    ybar = mean(yb, dims=:ens)

    # 2. Centered measurement perturbations
    if :amp in datatypes && :phase in datatypes
        Y = similar(yb)
        Y(:amp) .= yb(:amp) .- ybar(:amp)
        Y(:phase) .= phasediff.(yb(:phase), ybar(:phase))
    elseif :amp in datatypes
        Y = yb(:amp) .- ybar(:amp)
    elseif :phase in datatypes
        Y = phasediff.(yb(:phase), ybar(:phase))
    else
        error("Unknown datatypes: $datatypes")
    end

    # 3. Update each field if it exists, starting with the bias parameters
    updated_fields = NamedTuple()
    
    if haskey(xb, :tx_pwrs)
        if log10pwr_update
            log10_tx_pwrs = log10.(xb.tx_pwrs)
            log10_tx_pwrs_a = tx_pwrs_update(log10_tx_pwrs, y, ybar, Y, R; ρ=ρ)
            tx_pwrs = 10 .^ log10_tx_pwrs_a
        else
            tx_pwrs = tx_pwrs_update(xb.tx_pwrs, y, ybar, Y, R; ρ=ρ)
        end
        updated_fields = merge(updated_fields, (; tx_pwrs))

        pathnames=y.path
        ## G(b): apply TX power offsets
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
            η=η, commit_threshold=commit_threshold, correct_yb=true)
        updated_fields = merge(updated_fields, (; rx_phi_offset, rx_phi_logpost))
    end

    # Recompute Y after applying bias updates to yb
    ybar = mean(yb, dims=:ens)

    # Centered measurement perturbations
    if :amp in datatypes && :phase in datatypes
        Y = similar(yb)
        Y(:amp) .= yb(:amp) .- ybar(:amp)
        Y(:phase) .= phasediff.(yb(:phase), ybar(:phase))
    elseif :amp in datatypes
        Y = yb(:amp) .- ybar(:amp)
    elseif :phase in datatypes
        Y = phasediff.(yb(:phase), ybar(:phase))
    else
        error("Unknown datatypes: $datatypes")
    end
    
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
                                    ρ=ρ, log10pwr_update=log10pwr_update)
    updated_fields = merge(updated_fields, (; tx_pwrs=new_tx_pwrs))

    # 2b. RX bias: categorical update on the dual path; correct_yb=true folds the
    #     phase correction into yb in place.
    if do_rx
        rx_phi_offset, rx_phi_logpost = categorical_rx_measupdate!(
            xb.rx_phi_logpost, yb, xb.rx_phi_offset, y, R, rng;
            η=η, commit_threshold=commit_threshold, correct_yb=true)
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
    bias_only_update!(yb, tx_pwrs, y, R; ρ, log10pwr_update) → new_tx_pwrs

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
- `R`: diagonal observation-noise variance vector.

Returns `new_tx_pwrs`, the refined TX-power estimate to be used as the prior for
the next window iteration.

# Design notes
- The `yb` correction is the log-power change (in dB) relative to the prior mean.
- Calling this N times with successive observations accumulates corrections that
  telescope to `log10(mean(tx_N)/mean(tx_0))*10` — identical to a single step
  applied with the final estimate.
"""
function bias_only_update!(yb, tx_pwrs, y, R; ρ=1.1, log10pwr_update=false)

    pathnames = y.path
    npaths = length(pathnames)

    (:split_ens in dimnames(tx_pwrs)) ||
        error("bias_only_update!: tx_pwrs must contain a :split_ens dimension")
    split_ens_size = length(tx_pwrs.split_ens)

    # ── TX power update (amplitude data only) ────────────────────────────────
    new_tx_pwrs = similar(tx_pwrs)
    @showprogress Threads.@threads for e in yb.ens
        split_tx_update!(yb(ens=e), tx_pwrs(ens=e), new_tx_pwrs(ens=e),
                         y, R, ρ, npaths, split_ens_size, pathnames, log10pwr_update)
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
 
    if :amp in datatypes && :phase in datatypes
        Y = similar(yb)
        Y(:amp) .= yb(:amp) .- ybar(:amp)
        Y(:phase) .= phasediff.(yb(:phase), ybar(:phase))
    elseif :amp in datatypes
        Y = yb(:amp) .- ybar(:amp)
    elseif :phase in datatypes
        Y = phasediff.(yb(:phase), ybar(:phase))
    else
        error("xy_only_update: unknown datatypes $datatypes")
    end
 
    return xy_state_update(xy_state, y, ybar, Y, R;
        ρ=ρ, localization=localization, datatypes=datatypes)
end
 

function split_tx_update!(yb, tx_pwrs_b, tx_pwrs_a, y, R, ρ, npaths, split_ens_size, pathnames, log10pwr_update)
    
    @assert Set(dimnames(tx_pwrs_b)) == Set((:pwrs, :split_ens))
    #Necessary for mean in the for loop to calculate what we expect
    split_yb = KeyedArray(
        Array{Float64,3}(undef, 2, npaths, split_ens_size),
        field = [:amp, :phase],
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
            split_yb(field=:amp, ens=ee, path=txpaths) .+= Δpwr_log * 10  #10 dB per decade

        end
    end

    split_ybar = mean(split_yb, dims=:ens)

    split_Y = similar(split_yb)
    split_Y(:amp)   .= split_yb(field=:amp) .- split_ybar(:amp)
    split_Y(:phase) .= phasediff.(split_yb(field=:phase), split_ybar(:phase))

    for tx in tx_pwrs_b.pwrs
        split_tx_pwrs(pwrs=tx) .= strip(tx_pwrs_b(pwrs=tx)) 
    end

    if log10pwr_update
        xnew_amp = tx_pwrs_update(log10.(split_tx_pwrs), y, split_ybar, split_Y, R; ρ = ρ)

        for tx in tx_pwrs_b.pwrs
            tx_pwrs_a(pwrs=tx) .= 10 .^(strip(xnew_amp(pwrs=tx)))
        end
    else
        xnew_amp = tx_pwrs_update(split_tx_pwrs, y, split_ybar, split_Y, R; ρ = ρ)

        for tx in tx_pwrs_b.pwrs
            tx_pwrs_a(pwrs=tx) .= (strip(xnew_amp(pwrs=tx)))
        end
    end
end

"""
    xy_state_update(xy_state, y, ybar, Y, R; ρ=1.1, localization=nothing, datatypes=(:amp, :phase)) → xy_state_a
    Perform LETKF analysis update on only the `xy_state` state variable, given the measurements `y`, mean of the modeled measurements `ybar`, 
    ensemble differences from that mean `Y`, and the observation noise covariance `R`.
"""
function xy_state_update(xy_state, y, ybar, Y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase))

    gridshape = (length(xy_state.y), length(xy_state.x))
    ncells = prod(gridshape)
    npaths = length(y.path)
    ens_size = length(xy_state.ens)

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

        # Localize and flatten measurements
        ybar_loc = ybar(path=Index(loc_mask))
        Y_loc = Y(path=Index(loc_mask))
        y_loc = y(path=Index(loc_mask))

        if :amp in datatypes && :phase in datatypes
            Y_loc = KeyedArray([Array(Y_loc(:amp)); Array(Y_loc(:phase))];
                   path = vcat(collect(Y_loc.path), collect(Y_loc.path)),
                   ens  = collect(Y_loc.ens))
            R_loc = @views Diagonal([R[1:npaths][loc_mask]; R[npaths+1:end][loc_mask]])
        else
            # Only amp or phase
            R_loc = @views Diagonal(R[loc_mask])
        end

        # 4.
        C = strip(Y_loc)'/R_loc

        # 5.
        # Can apply ρ here if H is linear, or if ρ is close to 1
        Patilde = inv((ens_size - 1)*I/ρ + C*Y_loc)

        # 6.
        # Symmetric square root
        Wa = sqrt((ens_size - 1)*Hermitian(strip(Patilde)))

        # 7.
        if :amp in datatypes && :phase in datatypes
            Δ = KeyedArray(
                vcat(Array(y_loc(:amp)) .- Array(ybar_loc(:amp)),
                    phasediff.(Array(y_loc(:phase)), Array(ybar_loc(:phase))));
                path = vcat(collect(y_loc.path), collect(y_loc.path)),
                ens  = ybar_loc.ens,   # <-- keep OneTo instead of collecting
            )
        elseif :amp in datatypes
            Δ = y_loc(:amp) .- ybar_loc(:amp)
        elseif :phase in datatypes
            Δ = phasediff.(y_loc(:phase), ybar_loc(:phase))
        end

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
    tx_pwrs_update(tx_pwrs, y, ybar, Y, R; ρ=1.1) → tx_pwrs_a
    Perform LETKF analysis update on only the `tx_pwrs` bias offset state variable, given the measurements `y`,
    mean of the modeled measurements `ybar`, ensemble differences from that mean `Y`, and the observation noise covariance `R`.
"""
function tx_pwrs_update(tx_pwrs, y, ybar, Y, R; ρ=1.1)
    
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

        R_loc = @views Diagonal(R[1:npaths][loc_mask])
 
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
# ─────────────────────────────────────────────────────────────────────────────

"""
    categorical_rx_update!(log_post, yb, current_offsets, y, R; period=4)
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
Same convention as `rx_phi_update`: `R` may be length `npaths` (phase only) or
`2*npaths` (amp first, then phase). Phase variances are the second half in the
combined case. Each path uses its own σ².
"""
function categorical_rx_update!(log_post, yb, current_offsets, y, R; period=4, η=1.0)
    npaths = length(yb.path)
    quarter = π / (period / 2)  # = π/2 for period=4

    # Fail loud if path orderings ever diverge across the three KeyedArrays
    @assert axiskeys(yb, :path) == axiskeys(current_offsets, :path) == axiskeys(log_post, :path) ==
            axiskeys(y, :path) "categorical_rx_update!: :path axis mismatch across yb / current_offsets / log_post / y"

    # Locate phase variances in R
    if length(R) == npaths
        R_phase = R
    elseif length(R) == 2*npaths
        R_phase = R[npaths+1:end]
    else
        error("categorical_rx_update!: length(R) = $(length(R)) does not match $npaths or 2·$npaths")
    end

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
                               correct_yb=true, period=4)
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
                                    correct_yb::Bool=true, period=4)
    categorical_rx_update!(rx_log_post, yb, rx_phi_offset, y, R; period=period, η=η)

    if correct_yb
        posterior_resample_correct!(yb, rx_phi_offset, rx_log_post, rng;
                                    commit_threshold=commit_threshold, period=period)
    end

    return rx_phi_offset, rx_log_post
end

"""
    ensemble_model!(ym, f, x::NamedTuple, rx_log_post, rng)

Categorical-path variant of `ensemble_model!`. Identical to the standard
`NamedTuple` method except that the per-member rx offset added to `ym(:phase)`
is **sampled from the current per-path posterior** `rx_log_post` rather than
read from `x.rx_phi_offset` directly.

This implements the marginalization-over-Bϕ step: each ensemble member of
`xy_state` is paired with an independently-drawn `Bϕ` per path, so the
ensemble of modeled phases honestly represents joint uncertainty in `(x, Bϕ)`.
The drawn offsets are written into `x.rx_phi_offset(ens=e)` so that downstream
consumers (saving, heatmaps, the next iteration's `categorical_rx_update!`)
see a consistent record of what was actually used.

`x.rx_phi_offset` is expected to have dims `(path, ens)` — the categorical
path does not use the `:split_ens` dimension.
"""
function ensemble_model!(ym, f, x::NamedTuple, rx_log_post, rng; commit_threshold=1.0)
    haskey(x, :rx_phi_offset) ||
        error("ensemble_model! categorical variant requires x.rx_phi_offset")
    :split_ens in dimnames(x.rx_phi_offset) &&
        error("ensemble_model! categorical variant does not support :split_ens dimension")

    npaths = length(x.rx_phi_offset.path)
    ens_size = length(x.xy_state.ens)

    # Per-path: sample if posterior is uncertain, deterministically use MAP if it
    # exceeds `commit_threshold`. Pre-build the (path × ens) offset matrix so the
    # threaded forward-model loop never touches `rng`.
    sampled = Matrix{Float64}(undef, npaths, ens_size)
    sample_rx_offsets!(sampled, rx_log_post, rng; commit_threshold=commit_threshold)
    for e in 1:ens_size
        x.rx_phi_offset(ens=e) .= view(sampled, :, e)
    end

    @showprogress Threads.@threads for e in x.xy_state.ens
        xy_state = x.xy_state(ens=e)

        ens_state = NamedTuple()
        ens_state = merge(ens_state, (; xy_state))
        if haskey(x, :tx_pwrs)
            tx_pwrs = x.tx_pwrs(ens=e)
            ens_state = merge(ens_state, (; tx_pwrs))
        end

        a, p = f(ens_state)
        ym(:amp)(ens=e) .= a
        ym(:phase)(ens=e) .= p

        # Apply the per-member sampled offsets for this iteration
        ym(field=:phase, ens=e) .+= x.rx_phi_offset(ens=e) .* (π/2)
    end

    for pth in ym.path
        ym(:phase)(path=pth) .= modgaussian(ym(:phase)(path=pth))
    end

    return ym
end

"""
    ensemble_model!(ym, f, x)

Run the forward model `f` with `KeyedArray` argument `x` for each member of `x.ens`.
"""
function ensemble_model!(ym, f, x)
    # ym = KeyedArray(Array{Float64,3}(undef, 2, length(pathnames), length(x.ens));
    #         field=SVector(:amp, :phase), path=pathnames, ens=x.ens)
    @showprogress Threads.@threads for e in x.ens
        a, p = f(x(ens=e))
        ym(:amp)(ens=e) .= a
        ym(:phase)(ens=e) .= p
    end

    # Fit a Gaussian to phase data ensemble, then use wrap the phases from ±180° from the mean
    for p in ym.path
        ym(:phase)(path=p) .= modgaussian(ym(:phase)(path=p))
    end

    return ym
end

function ensemble_model!(ym, f, x::NamedTuple)
    # ym = KeyedArray(Array{Float64,3}(undef, 2, length(pathnames), length(x.ens));
    #         field=SVector(:amp, :phase), path=pathnames, ens=x.ens)
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
        a, p = f(ens_state)
        ym(:amp)(ens=e) .= a
        ym(:phase)(ens=e) .= p

        # Apply receiver phase offsets if present
        if haskey(x, :rx_phi_offset)
            if (:split_ens in dimnames(x.rx_phi_offset))
                offsets = mode.(eachslice(x.rx_phi_offset(ens=e), dims=:path))
                ym(field=:phase, ens=e) .+= offsets .* (π/2)
            else
                ym(field=:phase, ens=e) .+= x.rx_phi_offset(ens=e) .* (π/2) #implicitly, all paths must be in the same order
            end
        end
    end

    # Wrap phase ensemble around ±180°
    for pth in ym.path
        ym(:phase)(path=pth) .= modgaussian(ym(:phase)(path=pth))
    end

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