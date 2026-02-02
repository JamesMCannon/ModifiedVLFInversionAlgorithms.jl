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
                          ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase))

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
        tx_pwrs = tx_pwrs_update(xb.tx_pwrs, y, ybar, Y, R; ρ=ρ)
        updated_fields = merge(updated_fields, (; tx_pwrs))
    end

    if haskey(xb, :rx_phi_offset)
        rx_phi_offset = rx_phi_update(xb.rx_phi_offset, y, ybar, Y, R; ρ=ρ)
        updated_fields = merge(updated_fields, (; rx_phi_offset))
    end

    return updated_fields
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

    npaths = length(y.path)
    ens_size = length(rx_phi_offset.ens)

    # 2.
    rx_phibar = mean(rx_phi_offset,dims=:ens)

    for p in rx_phi_offset.path
        rx_phibar(path=p).= circular_mean(rx_phi_offset(path=p))
    end

    Xrx_phi = similar(rx_phi_offset)
    for e in rx_phi_offset.ens
        Xrx_phi(ens=e) .= dropdims(circular_diff.(rx_phi_offset(ens=e),rx_phibar),dims=:ens)
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

        R_loc = @views Diagonal(R[npaths+1:end][loc_mask])
 
        # 4.
        C = transpose(strip(Y_loc))/R_loc

        # 5.
        # Can apply ρ here if H is linear, or if ρ is close to 1
        Patilde = inv(Hermitian((ens_size - 1)*I/ρ + C*Y_loc))

        # 6.
        # Symmetric square root
        Wa = sqrt((ens_size - 1)*Hermitian(strip(Patilde)))

        # 7.
        Δ = phasediff.(y_loc, ybar_loc)
        
        wabar = Patilde*C*Δ
        wa = Wa .+ wabar

        # 8.
        rx_phibar_loc = rx_phibar(path = p_string)
        Xrx_phi_loc = transpose(Xrx_phi(path = p_string, ens = Index(Xrx_phi.ens))) # Transpose necessary because Julia flattens 1xk to (k,)

        rx_phi_offset_a(path = p_string) .= transpose(parent(parent(Xrx_phi_loc*wa .+ rx_phibar_loc)))
    end

    return rx_phi_offset_a
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
            ym(field=:phase, ens=e) .+= x.rx_phi_offset(ens=e) .* (π/2) #implicitly, all paths must be in the same order
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
