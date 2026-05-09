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
        ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase), filtertype=:stacked, log10pwr_update=false)

    if filtertype==:stacked 
        return LETKF_stacked_update(H, xb, y, R; ρ=ρ, localization=localization, datatypes=datatypes, log10pwr_update=log10pwr_update)
    elseif filtertype==:dual && (haskey(xb, :tx_pwrs) || haskey(xb, :rx_phi_offset))
        return LETKF_dual_update(H, xb, y, R; ρ=ρ, localization=localization, datatypes=datatypes, log10pwr_update=log10pwr_update)
    elseif filtertype==:split && (haskey(xb, :tx_pwrs) || haskey(xb, :rx_phi_offset))
        return LETKF_split_update(H, xb, y, R; ρ=ρ, localization=localization, datatypes=datatypes, log10pwr_update=log10pwr_update)
    else
        error("Unknown filter type: $filtertype. Currently :stacked, :dual, and :split are implemented.")
    end

end

function LETKF_stacked_update(H, xb::NamedTuple, y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase), log10pwr_update=false)
    
    # 1. Compute ensemble measurements
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
            log10_tx_pwrs = log10.(xb.tx_pwrs)
            log10_tx_pwrs_a = tx_pwrs_update(log10_tx_pwrs, y, ybar, Y, R; ρ=ρ)
            tx_pwrs = 10 .^ log10_tx_pwrs_a
        else
            tx_pwrs = tx_pwrs_update(xb.tx_pwrs, y, ybar, Y, R; ρ=ρ)
        end
        updated_fields = merge(updated_fields, (; tx_pwrs))
    end

    if haskey(xb, :rx_phi_offset)
        rx_phi_offset = rx_phi_update(xb.rx_phi_offset, y, ybar, Y, R; ρ=ρ)
        updated_fields = merge(updated_fields, (; rx_phi_offset))
    end

    return updated_fields
end

function LETKF_dual_update(H, xb::NamedTuple, y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase), log10pwr_update=false)
    
    # 1. Compute ensemble measurements
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

    if haskey(xb, :rx_phi_offset)
        rx_phi_offset = rx_phi_update(xb.rx_phi_offset, y, ybar, Y, R; ρ=ρ)
        updated_fields = merge(updated_fields, (; rx_phi_offset))

        # Apply final estimated RX offsets to ym prior to measurement update for then calculating the XY update
        for e in rx_phi_offset.ens
            yb(field=:phase, ens=e) .+= circular_diff.(rx_phi_offset(ens=e), xb.rx_phi_offset(ens=e)) .* (π/2)
        end
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
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase), log10pwr_update=false)
    
    if !((haskey(xb, :tx_pwrs) && (:split_ens in dimnames(xb.tx_pwrs))) || (haskey(xb, :rx_phi_offset) && (:split_ens in dimnames(xb.rx_phi_offset))))
            error("Must have split ensemble dimension in tx_pwrs or rx_phi_offset to use split update.")
    end
 
    # 1. Run forward model once
    yb = H(xb)
 
    # 2. Bias-only update (TX amplitude, then RX phase); yb corrected in-place
    tx_prior = haskey(xb, :tx_pwrs)       ? xb.tx_pwrs       : nothing
    rx_prior = haskey(xb, :rx_phi_offset) ? xb.rx_phi_offset : nothing
 
    new_tx_pwrs, new_rx_phi_offset = bias_only_update!(yb, tx_prior, rx_prior, y, R;
        ρ=ρ, log10pwr_update=log10pwr_update)
 
    updated_fields = NamedTuple()
    if !isnothing(new_tx_pwrs)
        updated_fields = merge(updated_fields, (; tx_pwrs=new_tx_pwrs))
    end
    if !isnothing(new_rx_phi_offset)
        updated_fields = merge(updated_fields, (; rx_phi_offset=new_rx_phi_offset))
    end
 
    # 3. xy_state update on the bias-corrected yb (ybar/Y recomputed inside)
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
    bias_only_update!(yb, tx_pwrs, rx_phi_offset, y, R; ρ, log10pwr_update)
 
Perform only the TX-power and RX-phase-offset LETKF updates from `LETKF_split_update`,
then immediately apply the resulting corrections back into `yb` so that accumulated
bias estimates are reflected in the ensemble predictions before the next window step.
 
# Arguments
- `yb`: ensemble prediction array `(field, path, ens)` — **mutated in-place**.
- `tx_pwrs`: current TX-power prior with dims `(pwrs, ens, split_ens)`, or `nothing`.
- `rx_phi_offset`: current RX-phase-offset prior with dims `(path, ens, split_ens)`, or `nothing`.
- `y`: observed data for this window step.
- `R`: diagonal observation-noise variance vector.
 
Returns `(new_tx_pwrs, new_rx_phi_offset)` — refined bias estimates to be used as the
prior for the next window iteration.  Either return value is `nothing` when the
corresponding statetype is absent.
 
# Design notes
- TX updates consume only amplitude data; RX updates consume only phase data.
  The two updates are therefore independent and can be applied in either order.
- The yb correction for TX is the log-power change (in dB) relative to the prior mean.
- The yb correction for RX is the circular change in mode offset (in radians).
- Calling this function N times with successive observations accumulates corrections
  that telescope to `log10(mean(tx_N)/mean(tx_0))*10` — identical to a single step
  applied with the final estimate.
"""
function bias_only_update!(yb, tx_pwrs, rx_phi_offset, y, R; ρ=1.1, log10pwr_update=false)
 
    pathnames = y.path
    npaths = length(pathnames)
 
    if !isnothing(tx_pwrs) && :split_ens in dimnames(tx_pwrs)
        split_ens_size = length(tx_pwrs.split_ens)
    elseif !isnothing(rx_phi_offset) && :split_ens in dimnames(rx_phi_offset)
        split_ens_size = length(rx_phi_offset.split_ens)
    else
        error("bias_only_update!: tx_pwrs or rx_phi_offset must contain a :split_ens dimension")
    end
 
    new_tx_pwrs = nothing
    new_rx_phi_offset = nothing
 
    # ── TX power update (amplitude data only) ────────────────────────────────
    if !isnothing(tx_pwrs)
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
    end
 
    # ── RX phase-offset update (phase data only) ─────────────────────────────
    if !isnothing(rx_phi_offset)
        new_rx_phi_offset = similar(rx_phi_offset)
        @showprogress Threads.@threads for e in yb.ens
            split_rx_update!(yb(ens=e), rx_phi_offset(ens=e), new_rx_phi_offset(ens=e),
                             y, R, ρ, npaths, split_ens_size, pathnames)
        end
 
        # Accumulate phase correction: Δ = change in mode offset per path (quarter-turns → radians)
        for e in new_rx_phi_offset.ens
            new_offsets = mode.(eachslice(new_rx_phi_offset(ens=e), dims=:path))
            old_offsets = mode.(eachslice(rx_phi_offset(ens=e), dims=:path))
            yb(field=:phase, ens=e) .+= circular_diff.(new_offsets, old_offsets) .* (π/2)
            # We apply the mode, subtracting the offsets already applied in ensemble_model, as the indicator of the most likely phase offset.
            # With modulo 4 discrete variables, where intermediate values between the integer values are meaningless, 
            # mean is not a good measure of central tendency, and mode is more appropriate. 
        end
    end
 
    return new_tx_pwrs, new_rx_phi_offset
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

function split_rx_update!(yb, rx_phi_offset_b, rx_phi_offset_a, y, R, ρ, npaths, split_ens_size, pathnames)
    split_yb = KeyedArray(
        Array{Float64,3}(undef, 2, npaths, split_ens_size),
        field = [:amp, :phase],
        path  = pathnames,
        ens   = 1:split_ens_size,
    )

    for ee in split_yb.ens
        split_yb(ens=ee) .= yb #broadcast to split_ens_size
    end

    split_rx_phi_offset = KeyedArray(
        fill(NaN, npaths, split_ens_size), 
        path = pathnames, 
        ens=1:split_ens_size
        )
    #needed because rx_phi_offset_update requires structure with dimesion ens, not split_ens. Currently keeping both so (dual_)rx_phi_offset can be mutated for all ensembles.
    # Compute mode per path before the perturbation loop
    # We use the mode here because ensemble_model!() uses the mode when computing the modeled measurements.
    # This means we have to use deviation from the mode when calculating the Y matrix for the LETKF update.
    mode_offset = mode.(eachslice(rx_phi_offset_b, dims=:path))

    for ee in rx_phi_offset_b.split_ens
        split_yb(field=:phase, ens=ee) .+= 
            circular_diff.(rx_phi_offset_b(split_ens=ee), mode_offset) .* (π/2)
    end

    split_ybar = mean(split_yb, dims=:ens)

    split_Y = similar(split_yb)
    split_Y(:amp)   .= split_yb(field=:amp) .- split_ybar(:amp)
    split_Y(:phase) .= phasediff.(split_yb(field=:phase), split_ybar(:phase))

    for p in rx_phi_offset_b.path
        split_rx_phi_offset(path=p) .= strip(rx_phi_offset_b(path=p)) 
    end

    xnew_phi = rx_phi_update(split_rx_phi_offset, y, split_ybar, split_Y, R; ρ = ρ)

    for p in rx_phi_offset_a.path
        rx_phi_offset_a(path=p) .= mod.(round.(strip(xnew_phi(path=p))), 4) #force to integer values ∈ [0, 3]
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


"""
    rx_phi_update(tx_pwrs, y, ybar, Y, R; ρ=1.1) → tx_pwrs_a
    Perform LETKF analysis update on only the `tx_pwrs` bias offset state variable, given the measurements `y`,
    mean of the modeled measurements `ybar`, ensemble differences from that mean `Y`, and the observation noise covariance `R`.
"""
function rx_phi_update(rx_phi_offset, y, ybar, Y, R; ρ=1.1)
    
    missing_paths = setdiff(rx_phi_offset.path, y.path)
    #check that all paths in rx_phi_offset are in y.path
    if isempty(missing_paths)
        # ok
    else
        error("Missing entries from y.path: $(missing_paths)")
    end

    if !(:field in dimnames(Y))
        #Stacked/Dual update removes the field dimension, causing breakage here.
        #TODO determine why this behavior changed when first adding the different filtertype functions
        Y = KeyedArray(
           reshape(Y, size(Y, :path), size(Y, :ens), 1);
           path  = axiskeys(Y, :path),
           ens   = axiskeys(Y, :ens),
           field = [:phase]          # singleton dimension
        )
    end

    npaths = length(y.path)
    ens_size = length(rx_phi_offset.ens)

    # 2.
    #=
    θ = rx_phi_offset .* (π/2)
    zc = cos.(θ)
    zs = sin.(θ)
    zbar_c = mean(zc, dims=:ens)
    zbar_s = mean(zs, dims=:ens)
    Xc = zc .- zbar_c
    Xs = zs .- zbar_s
    =#
    rx_phibar = mean(rx_phi_offset,dims=:ens)
    #Mean is just used to create a structure of the correct dimensions

    for p in rx_phi_offset.path
        rx_phibar(path=p) .= robust_zero_sum_center(rx_phi_offset(path=p))
        #rx_phibar(path=p).= circular_mean(rx_phi_offset(path=p))
        #rx_phibar(path=p) .= mean(rx_phi_offset(path=p))
        #rx_phibar(path=p).= mode(rx_phi_offset(path=p))
        #=
        center = mean(rx_phi_offset(path=p))
        perturbs = circular_diff.(rx_phi_offset(path=p), center)
        correction = mean(perturbs)
        rx_phibar(path=p) .= center + correction
        =#
    end

    Xrx_phi = similar(rx_phi_offset)
    for e in rx_phi_offset.ens
        #perturbs = circular_diff.(rx_phi_offset(ens=e), dropdims(rx_phibar, dims=:ens))
        #Xrx_phi(ens=e) .= perturbs .- mean(perturbs) 
        #Xrx_phi(ens=e) .= dropdims(circular_diff.(rx_phi_offset(ens=e),rx_phibar),dims=:ens)
        Xrx_phi(ens=e) .= dropdims(balanced_circular_diff(rx_phi_offset(ens=e),rx_phibar),dims=:ens)
    end
    

    #For localizing RX phase offset state variable, we consider only phase data 
    #from the current path.
    rx_phi_offset_a = similar(rx_phi_offset)
    for n in 1:npaths
        p_string = String(rx_phi_offset.path[n])
        # Currently localization is binary (cell is included or not)
        loc_mask = BitVector()
        loc_mask = BitVector([s == p_string for s in y.path])

        # Localize and flatten measurements
        ybar_loc = ybar(path=Index(loc_mask), field=:phase)
        Y_loc = Y(path=Index(loc_mask), field=:phase)
        y_loc = y(path=Index(loc_mask), field=:phase)

        # Indices of R to be copied to R_loc depends on whether R has amp and phase measurement covariances or just phase. 
        # Regardless, only phase is used to update rx_phi_offset
        if length(R) == length(rx_phi_offset.path)
            R_loc = @views Diagonal(R[1:end][loc_mask])
        elseif length(R) == 2*length(rx_phi_offset.path)
            R_loc = @views Diagonal(R[npaths+1:end][loc_mask])
        else
            error("Length of R must be equal to number of paths or twice the number of paths. Length of R: $(length(R)), number of paths: $(length(rx_phi_offset.path))")
        end
        # 4.
        C = transpose(strip(Y_loc))/R_loc

        # 5.
        # Can apply ρ here if H is linear, or if ρ is close to 1
        Patilde = inv(Hermitian((ens_size - 1)*I/ρ + C*Y_loc)) #Possibly persue explicit symmetric construction and use / rather than inv()

        # 6.
        # Symmetric square root
        Wa = sqrt((ens_size - 1)*Hermitian(Patilde))

        # 7.
        Δ = phasediff.(y_loc, ybar_loc)
        
        wabar = Patilde*C*Δ
        wa = Wa .+ wabar

        # 8.
        rx_phibar_loc = rx_phibar(path = p_string)
        Xrx_phi_loc = transpose(Xrx_phi(path = p_string, ens = Index(Xrx_phi.ens))) # Transpose necessary because Julia flattens 1xk to (k,)

        rx_phi_offset_a(path = p_string) .= transpose(parent(parent(Xrx_phi_loc*wa .+ rx_phibar_loc)))
        #=
        # Stack (cos, sin) rows for this path; shape (2, ens_size)
        Xloc_c = transpose(parent(parent(Xc(path=p_string))))
        Xloc_s = transpose(parent(parent(Xs(path=p_string))))
        X2 = vcat(Xloc_c, Xloc_s)          # 2 × ens_size

        zbar2 = [only(zbar_c(path=p_string)); only(zbar_s(path=p_string))]
        z_a = X2*wa .+ zbar2               # 2 × ens_size (one column per member)

        # Project back: angle → quarter-turn count in [0,4). Downstream `round`+`mod 4` snaps to {0,1,2,3}.
        θa = atan.(z_a[2,:], z_a[1,:])
        rx_phi_offset_a(path=p_string) .= mod.(θa ./ (π/2), 4)
        =#
    end

    return rx_phi_offset_a
end

# ─────────────────────────────────────────────────────────────────────────────
# Categorical RX-offset update path
#
# Alternative to `rx_phi_update` for the discrete MSK-ambiguity case where
# k_p ∈ {0,1,2,3} is constant in time. Maintains a per-path log-posterior and
# accumulates evidence across iterations. See `runletkf` `:categorical` branch.
# ─────────────────────────────────────────────────────────────────────────────

"""
    categorical_rx_update!(log_post, yb, current_offsets, y, R; period=4)
        → log_post (mutated)

Accumulate Bayesian evidence into the per-path log-posterior `log_post`
(`KeyedArray` with dims `path × k`, where `k = 0:period-1`) given the current
ensemble forward-model output `yb`, the per-member offsets that were applied
inside the forward model `current_offsets` (`KeyedArray(path, ens)`), the
observation `y` for this iteration, and the diagonal observation-noise vector
`R`.

# Recovery of offset-free `yb`
`ensemble_model!` adds `current_offsets(path=p, ens=e) * (π/2)` to
`yb(:phase, path=p, ens=e)` before this function sees it. To compute the
likelihood of each candidate `k`, we need the raw forward-model output. We
recover it on the fly by subtracting `current_offsets * (π/2)` per member.
This avoids a duplicate forward-model call.

# Posterior accumulation
`k_p` is constant in time (MSK ambiguity is fixed at receiver lock per
preprocessing assumption), so iteration `t`'s posterior is

    log_post_t(k) = log_post_{t-1}(k) + ℓ_t(k)

where `ℓ_t(k)` is the marginal log-likelihood from `rx_phi_loglikelihood`.
Normalization is applied lazily — `log_post` accumulates as raw log-likelihoods,
and consumers (`rx_phi_posterior`, `rx_phi_sample`) normalize on read.

# `R` handling
Same convention as `rx_phi_update`: `R` may be length `npaths` (phase only) or
`2*npaths` (amp first, then phase). Phase variances are the second half in the
combined case. Each path uses its own σ².
"""
function categorical_rx_update!(log_post, yb, current_offsets, y, R; period=4)
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

        # Accumulate (constant k_p ⇒ no forgetting factor).
        log_post(path=p) .+= ℓ
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
        Δk  = circular_diff.(new, old; period=period)
        yb(field=:phase, ens=e) .+= Δk .* quarter
        rx_phi_offset(ens=e) .= new
    end

    return rx_phi_offset
end

"""
    ensemble_model!(ym, f, x::NamedTuple, rx_log_post, rng)

Categorical-path variant of `ensemble_model!`. Identical to the standard
`NamedTuple` method except that the per-member rx offset added to `ym(:phase)`
is **sampled from the current per-path posterior** `rx_log_post` rather than
read from `x.rx_phi_offset` directly.

This implements the marginalization-over-k step: each ensemble member of
`xy_state` is paired with an independently-drawn `k_p` per path, so the
ensemble of modeled phases honestly represents joint uncertainty in `(x, k)`.
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