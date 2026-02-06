"""
Extended Local Projection methods.

This module provides:
- LP-IV: Instrumental Variables (Stock & Watson 2018)
- Smooth LP: B-spline parameterization (Barnichon & Brownlees 2019)
- State-dependent LP: Regime switching (Auerbach & Gorodnichenko 2013)
- Propensity LP: Treatment effects via IPW (Angrist et al. 2018)

All methods use shared utilities from lp_core.jl.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# LP-IV: Instrumental Variables (Stock & Watson 2018)
# =============================================================================

"""
    first_stage_regression(endog::AbstractVector{T}, instruments::AbstractMatrix{T},
                           controls::AbstractMatrix{T}) -> NamedTuple

First-stage regression for 2SLS with F-statistic for instrument relevance.
"""
function first_stage_regression(endog::AbstractVector{T}, instruments::AbstractMatrix{T},
                                 controls::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = length(endog)
    n_inst = size(instruments, 2)

    X = hcat(ones(T, n), instruments, controls)
    k = size(X, 2)

    XtX_inv = robust_inv(X' * X)
    beta = XtX_inv * (X' * endog)
    fitted = X * beta
    residuals = endog - fitted
    sigma2 = sum(residuals.^2) / (n - k)

    # F-statistic for instruments
    inst_coef = beta[2:(n_inst + 1)]
    V_inst = XtX_inv[2:(n_inst + 1), 2:(n_inst + 1)]
    F_stat = n_inst == 1 ? inst_coef[1]^2 / (sigma2 * V_inst[1, 1]) :
                           inst_coef' * inv(V_inst) * inst_coef / (n_inst * sigma2)

    (fitted=fitted, residuals=residuals, F_stat=T(F_stat), coefficients=beta,
     vcov=sigma2 * XtX_inv, sigma2=sigma2, n_instruments=n_inst)
end

"""
    tsls_regression(Y::AbstractMatrix{T}, endog::AbstractVector{T},
                    endog_fitted::AbstractVector{T}, controls::AbstractMatrix{T};
                    cov_estimator::AbstractCovarianceEstimator=NeweyWestEstimator()) -> NamedTuple

Second-stage regression using fitted values from first stage.
"""
function tsls_regression(Y::AbstractMatrix{T}, endog::AbstractVector{T},
                         endog_fitted::AbstractVector{T}, controls::AbstractMatrix{T};
                         cov_estimator::AbstractCovarianceEstimator=NeweyWestEstimator{T}()) where {T<:AbstractFloat}
    n, n_resp = size(Y)

    X_2s = hcat(ones(T, n), endog_fitted, controls)
    k = size(X_2s, 2)

    XtX_inv = robust_inv(X_2s' * X_2s)
    beta = XtX_inv * (X_2s' * Y)

    # Residuals using actual endog for correct SE
    X_actual = hcat(ones(T, n), endog, controls)
    residuals = Y - X_actual * beta

    V = compute_block_robust_vcov(X_2s, residuals, cov_estimator)

    (coefficients=beta, residuals=residuals, vcov=V, n_obs=n, n_regressors=k)
end

"""
    estimate_lp_iv(Y::AbstractMatrix{T}, shock_var::Int, instruments::AbstractMatrix{T},
                   horizon::Int; lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y,2)),
                   cov_type::Symbol=:newey_west, bandwidth::Int=0) -> LPIVModel{T}

Estimate LP with Instrumental Variables (Stock & Watson 2018) using 2SLS.
"""
function estimate_lp_iv(Y::AbstractMatrix{T}, shock_var::Int, instruments::AbstractMatrix{T},
                        horizon::Int; lags::Int=4,
                        response_vars::Vector{Int}=collect(1:size(Y, 2)),
                        cov_type::Symbol=:newey_west, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    n_inst = size(instruments, 2)

    @assert size(instruments, 1) == T_obs "instruments must have same T as Y"
    @assert n_inst >= 1 "need at least one instrument"
    @assert 1 <= shock_var <= n "shock_var must be in 1:$n"

    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)
    n_response = length(response_vars)

    B = Vector{Matrix{T}}(undef, horizon + 1)
    residuals_store = Vector{Matrix{T}}(undef, horizon + 1)
    vcov = Vector{Matrix{T}}(undef, horizon + 1)
    first_stage_F = Vector{T}(undef, horizon + 1)
    first_stage_coef = Vector{Vector{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)

    for h in 0:horizon
        t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
        T_h = t_end - t_start + 1
        T_eff[h + 1] = T_h

        Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)

        # Endogenous and instruments at t
        endog = [Y[t, shock_var] for t in t_start:t_end]
        Z = instruments[t_start:t_end, :]

        # Controls: lagged Y
        n_ctrl = n * lags
        controls = Matrix{T}(undef, T_h, n_ctrl)
        col = 1
        @inbounds for (i, t) in enumerate(t_start:t_end)
            col_local = 1
            for lag in 1:lags, var in 1:n
                controls[i, col_local] = Y[t - lag, var]
                col_local += 1
            end
        end

        # First and second stage
        fs = first_stage_regression(endog, Z, controls)
        first_stage_F[h + 1] = fs.F_stat
        first_stage_coef[h + 1] = fs.coefficients[2:(n_inst + 1)]

        ss = tsls_regression(Y_h, endog, fs.fitted, controls; cov_estimator=cov_estimator)
        B[h + 1] = ss.coefficients
        residuals_store[h + 1] = ss.residuals
        vcov[h + 1] = ss.vcov
    end

    LPIVModel{T}(Matrix{T}(Y), shock_var, response_vars, Matrix{T}(instruments),
                 horizon, lags, B, residuals_store, vcov, first_stage_F,
                 first_stage_coef, T_eff, cov_estimator)
end

estimate_lp_iv(Y::AbstractMatrix, shock_var::Int, instruments::AbstractMatrix,
               horizon::Int; kwargs...) =
    estimate_lp_iv(Float64.(Y), shock_var, Float64.(instruments), horizon; kwargs...)

"""
    weak_instrument_test(model::LPIVModel{T}; threshold::T=T(10.0)) -> NamedTuple

Test for weak instruments using Stock-Yogo rule of thumb (F > 10).
"""
function weak_instrument_test(model::LPIVModel{T}; threshold::T=T(10.0)) where {T<:AbstractFloat}
    F_stats = model.first_stage_F
    weak_horizons = findall(F_stats .< threshold)
    (F_stats=F_stats, weak_horizons=weak_horizons, min_F=minimum(F_stats),
     passes_threshold=isempty(weak_horizons), threshold=threshold)
end

weak_instrument_test(F_stats::Vector{T}; threshold::T=T(10.0)) where {T<:AbstractFloat} =
    (F_stats=F_stats, weak_horizons=findall(F_stats .< threshold),
     min_F=minimum(F_stats), passes_threshold=all(F_stats .>= threshold), threshold=threshold)

"""
    lp_iv_irf(model::LPIVModel{T}; conf_level::Real=0.95) -> LPImpulseResponse{T}

Extract IRF from LP-IV model.
"""
function lp_iv_irf(model::LPIVModel{T}; conf_level::Real=0.95) where {T<:AbstractFloat}
    irf_data = extract_shock_irf(model.B, model.vcov, model.response_vars, 2;
                                  conf_level=conf_level)

    response_names = default_var_names(length(model.response_vars); prefix="Var")
    shock_name = "Instrumented Shock $(model.shock_var)"
    cov_type_sym = model.cov_estimator isa NeweyWestEstimator ? :newey_west : :white

    LPImpulseResponse{T}(irf_data.values, irf_data.ci_lower, irf_data.ci_upper,
                         irf_data.se, model.horizon, response_names, shock_name,
                         cov_type_sym, T(conf_level))
end

"""
    sargan_test(model::LPIVModel{T}, h::Int) -> NamedTuple

Sargan-Hansen J-test for overidentification at horizon h.
"""
function sargan_test(model::LPIVModel{T}, h::Int) where {T<:AbstractFloat}
    n_inst = n_instruments(model)
    n_inst <= 1 && return (J_stat=T(NaN), p_value=T(NaN), df=0, valid=false)

    U_h = model.residuals[h + 1]
    n_resp = size(U_h, 2)
    T_h = model.T_eff[h + 1]

    t_start, t_end = compute_horizon_bounds(size(model.Y, 1), h, model.lags)
    Z = model.instruments[t_start:t_end, :]

    J_stats = [begin
        u = @view U_h[:, eq]
        sigma2 = sum(u.^2) / T_h
        Zu = Z' * u
        ZtZ_inv = robust_inv(Z' * Z)
        T_h * (Zu' * ZtZ_inv * Zu) / sigma2
    end for eq in 1:n_resp]

    J_avg = mean(J_stats)
    df = n_inst - 1
    (J_stat=J_avg, p_value=1 - cdf(Chisq(df), J_avg), df=df, valid=true)
end

# =============================================================================
# Smooth LP: B-splines (Barnichon & Brownlees 2019)
# =============================================================================

"""
    bspline_basis_value(x::T, i::Int, degree::Int, knots::Vector{T}) where T

Evaluate B-spline basis function using Cox-de Boor recursion.
"""
function bspline_basis_value(x::T, i::Int, degree::Int, knots::Vector{T}) where {T<:AbstractFloat}
    if degree == 0
        return (i == length(knots) - 1 ? knots[i] <= x <= knots[i + 1] :
                                          knots[i] <= x < knots[i + 1]) ? one(T) : zero(T)
    end

    n = length(knots)
    denom1 = knots[i + degree] - knots[i]
    w1 = abs(denom1) < eps(T) ? zero(T) : (x - knots[i]) / denom1

    denom2 = knots[i + degree + 1] - knots[i + 1]
    w2 = abs(denom2) < eps(T) ? zero(T) : (knots[i + degree + 1] - x) / denom2

    left = i + degree <= n ? w1 * bspline_basis_value(x, i, degree - 1, knots) : zero(T)
    right = i + 1 + degree <= n ? w2 * bspline_basis_value(x, i + 1, degree - 1, knots) : zero(T)

    left + right
end

"""
    bspline_basis(horizons::AbstractVector{Int}, degree::Int, n_interior_knots::Int;
                  T::Type{<:AbstractFloat}=Float64) -> BSplineBasis{T}

Construct B-spline basis matrix for given horizons.
"""
function bspline_basis(horizons::AbstractVector{Int}, degree::Int, n_interior_knots::Int;
                       T::Type{<:AbstractFloat}=Float64)
    @assert degree >= 0 && n_interior_knots >= 0

    h_min, h_max = T(minimum(horizons)), T(maximum(horizons))
    n_b = n_interior_knots + degree + 1

    # Construct clamped knot vector
    knots = if n_interior_knots > 0
        interior = range(h_min, h_max, length=n_interior_knots + 2)[2:end-1]
        vcat(fill(h_min, degree + 1), collect(T, interior), fill(h_max, degree + 1))
    else
        vcat(fill(h_min, degree + 1), fill(h_max, degree + 1))
    end

    basis_matrix = [bspline_basis_value(T(h), j, degree, knots)
                    for h in horizons, j in 1:n_b]

    BSplineBasis{T}(degree, n_interior_knots, knots, basis_matrix, collect(horizons))
end

"""
    roughness_penalty_matrix(basis::BSplineBasis{T}) -> Matrix{T}

Compute roughness penalty matrix R for B-splines (second derivative penalty).
"""
function roughness_penalty_matrix(basis::BSplineBasis{T}) where {T<:AbstractFloat}
    n_b = n_basis(basis)
    h_min, h_max = T(minimum(basis.horizons)), T(maximum(basis.horizons))
    n_grid = max(100, 10n_b)
    grid = range(h_min, h_max, length=n_grid)
    dx = (h_max - h_min) / (n_grid - 1)

    B = [bspline_basis_value(T(x), j, basis.degree, basis.knots)
         for x in grid, j in 1:n_b]

    # Second derivative via finite differences
    B_dd = [(B[i+1, j] - 2B[i, j] + B[i-1, j]) / dx^2
            for i in 2:(n_grid-1), j in 1:n_b]

    R = dx * (B_dd' * B_dd)
    (R + R') / 2
end

"""
    estimate_smooth_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                       degree::Int=3, n_knots::Int=4, lambda::T=T(0.0),
                       lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y,2)),
                       cov_type::Symbol=:newey_west, bandwidth::Int=0) -> SmoothLPModel{T}

Estimate Smooth LP with B-spline parameterization (Barnichon & Brownlees 2019).
"""
function estimate_smooth_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                            degree::Int=3, n_knots::Int=4, lambda::T=T(0.0),
                            lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                            cov_type::Symbol=:newey_west, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    n_response = length(response_vars)
    horizons = collect(0:horizon)

    basis = bspline_basis(horizons, degree, n_knots; T=T)
    n_b = n_basis(basis)
    B_mat = basis.basis_matrix
    R = roughness_penalty_matrix(basis)

    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)

    # Step 1: Standard LP estimates
    lp_model = estimate_lp(Y, shock_var, horizon; lags=lags, response_vars=response_vars,
                           cov_type=cov_type, bandwidth=bandwidth)
    lp_result = lp_irf(lp_model)

    beta_hat = lp_result.values
    beta_se = lp_result.se

    # Step 2: Fit spline with weighted least squares + penalty
    theta = Matrix{T}(undef, n_b, n_response)
    vcov_theta_blocks = Vector{Matrix{T}}(undef, n_response)

    for j in 1:n_response
        w = 1 ./ (beta_se[:, j].^2 .+ eps(T))
        W = Diagonal(w)
        reg_mat = B_mat' * W * B_mat + lambda * R
        theta[:, j] = reg_mat \ (B_mat' * W * beta_hat[:, j])

        reg_inv = inv(reg_mat)
        Sigma_beta = Diagonal(beta_se[:, j].^2)
        vcov_theta_blocks[j] = reg_inv * (B_mat' * W * Sigma_beta * W * B_mat) * reg_inv
    end

    vcov_theta = zeros(T, n_b * n_response, n_b * n_response)
    for j in 1:n_response
        idx = ((j-1)*n_b + 1):(j*n_b)
        vcov_theta[idx, idx] .= vcov_theta_blocks[j]
    end

    irf_values = B_mat * theta
    irf_se = [sqrt(B_mat[h_idx, :]' * vcov_theta_blocks[j] * B_mat[h_idx, :])
              for h_idx in 1:length(horizons), j in 1:n_response]

    residuals = vcat([lp_model.residuals[h+1] for h in horizons]...)
    T_total = sum(T_obs - h - lags for h in horizons)

    SmoothLPModel{T}(Matrix{T}(Y), shock_var, response_vars, horizon, lags, basis,
                     theta, vcov_theta, lambda, irf_values, Matrix{T}(irf_se),
                     residuals, T_total, cov_estimator)
end

estimate_smooth_lp(Y::AbstractMatrix, shock_var::Int, horizon::Int; kwargs...) =
    estimate_smooth_lp(Float64.(Y), shock_var, horizon; kwargs...)

"""
    cross_validate_lambda(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                          lambda_grid::Vector{T}=T.(10.0 .^ (-4:0.5:2)),
                          k_folds::Int=5, kwargs...) -> T

Select optimal λ via k-fold cross-validation.
"""
function cross_validate_lambda(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                               lambda_grid::Vector{T}=T.(10.0 .^ (-4:0.5:2)),
                               k_folds::Int=5, kwargs...) where {T<:AbstractFloat}
    T_obs = size(Y, 1)
    lags = get(kwargs, :lags, 4)
    t_start, t_end = compute_horizon_bounds(T_obs, horizon, lags)
    n_usable = t_end - t_start + 1
    fold_size = n_usable ÷ k_folds

    cv_errors = zeros(T, length(lambda_grid))

    for (i, lam) in enumerate(lambda_grid)
        fold_mse = zeros(T, k_folds)
        for k in 1:k_folds
            start_k = t_start + (k - 1) * fold_size
            end_k = k == k_folds ? t_end : start_k + fold_size - 1
            test_idx = collect(start_k:end_k)

            train_mask = trues(T_obs)
            for t in test_idx, h in 0:horizon
                t + h <= T_obs && (train_mask[t + h] = false)
            end
            train_idx = findall(train_mask)

            length(train_idx) < lags + horizon + 10 && continue

            try
                model = estimate_smooth_lp(Y[train_idx, :], shock_var, horizon; lambda=lam, kwargs...)
                fold_mse[k] = mean((model.irf_values .- model.spline_basis.basis_matrix * model.theta).^2)
            catch
                fold_mse[k] = T(Inf)
            end
        end
        cv_errors[i] = mean(fold_mse[fold_mse .< Inf])
    end

    lambda_grid[argmin(cv_errors)]
end

"""
    smooth_lp_irf(model::SmoothLPModel{T}; conf_level::Real=0.95) -> LPImpulseResponse{T}

Extract smoothed IRF from SmoothLPModel.
"""
function smooth_lp_irf(model::SmoothLPModel{T}; conf_level::Real=0.95) where {T<:AbstractFloat}
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = model.irf_values .- z .* model.irf_se
    ci_upper = model.irf_values .+ z .* model.irf_se

    response_names = default_var_names(length(model.response_vars); prefix="Var")
    cov_type_sym = model.cov_estimator isa NeweyWestEstimator ? :newey_west : :white

    LPImpulseResponse{T}(model.irf_values, ci_lower, ci_upper, model.irf_se, model.horizon,
                         response_names, "Shock $(model.shock_var)", cov_type_sym, T(conf_level))
end

"""
    compare_smooth_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                      lambda::T=T(1.0), kwargs...) -> NamedTuple

Compare standard LP and smooth LP.
"""
function compare_smooth_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                           lambda::T=T(1.0), kwargs...) where {T<:AbstractFloat}
    std_irf = lp_irf(estimate_lp(Y, shock_var, horizon; kwargs...))
    sm_irf = smooth_lp_irf(estimate_smooth_lp(Y, shock_var, horizon; lambda=lambda, kwargs...))
    (standard_irf=std_irf, smooth_irf=sm_irf,
     variance_reduction=mean(sm_irf.se.^2) / mean(std_irf.se.^2))
end

# =============================================================================
# State-Dependent LP (Auerbach & Gorodnichenko 2013)
# =============================================================================

"""
    logistic_transition(z::AbstractVector{T}, gamma::T, c::T) -> Vector{T}

Logistic transition function: F(z) = exp(-γ(z - c)) / (1 + exp(-γ(z - c)))
"""
logistic_transition(z::AbstractVector{T}, gamma::T, c::T) where {T<:AbstractFloat} =
    @. exp(-gamma * (z - c)) / (1 + exp(-gamma * (z - c)))

logistic_transition(z::T, gamma::T, c::T) where {T<:AbstractFloat} =
    exp(-gamma * (z - c)) / (1 + exp(-gamma * (z - c)))

"""
    exponential_transition(z::AbstractVector{T}, gamma::T, c::T) -> Vector{T}

Exponential (symmetric) transition: F(z) = 1 - exp(-γ(z - c)²)
"""
exponential_transition(z::AbstractVector{T}, gamma::T, c::T) where {T<:AbstractFloat} =
    @. 1 - exp(-gamma * (z - c)^2)

"""
    indicator_transition(z::AbstractVector{T}, c::T) -> Vector{T}

Sharp indicator transition: F(z) = 1 if z ≥ c, else 0
"""
indicator_transition(z::AbstractVector{T}, c::T) where {T<:AbstractFloat} = T.(z .>= c)

"""
    estimate_transition_params(state_var::AbstractVector{T}, Y::AbstractMatrix{T},
                               shock_var::Int; method::Symbol=:nlls, ...) -> NamedTuple

Estimate smooth transition parameters (γ, c).
"""
function estimate_transition_params(state_var::AbstractVector{T}, Y::AbstractMatrix{T},
                                    shock_var::Int; method::Symbol=:nlls,
                                    gamma_init::T=T(1.5),
                                    c_init::Union{T,Symbol}=:median) where {T<:AbstractFloat}
    @assert length(state_var) == size(Y, 1)

    c0 = c_init isa Symbol ? (c_init == :median ? median(state_var) :
                               c_init == :mean ? mean(state_var) : zero(T)) : c_init

    compute_ssr(gamma, c) = begin
        F = logistic_transition(state_var, gamma, c)
        T_obs, n = size(Y)
        ssr = zero(T)
        for t in 2:T_obs
            shock_t = Y[t-1, shock_var]
            for var in 1:n
                pred = (1 - F[t-1]) * shock_t + F[t-1] * shock_t
                ssr += (Y[t, var] - pred)^2
            end
        end
        ssr
    end

    if method == :grid_search
        gamma_grid = T.(range(0.5, 5.0, length=20))
        c_grid = T.(quantile(state_var, range(0.1, 0.9, length=20)))

        best_gamma, best_c, best_ssr = gamma_init, c0, T(Inf)
        for gamma in gamma_grid, c in c_grid
            ssr = compute_ssr(gamma, c)
            if ssr < best_ssr
                best_ssr, best_gamma, best_c = ssr, gamma, c
            end
        end
        (gamma=best_gamma, c=best_c, F_values=logistic_transition(state_var, best_gamma, best_c),
         convergence_info=(method=:grid_search, ssr=best_ssr))
    else  # :nlls
        gamma_curr, c_curr = gamma_init, c0
        for iter in 1:50
            gamma_prev, c_prev = gamma_curr, c_curr

            # Optimize γ
            for gamma in T.(range(max(0.1, gamma_curr - 1), gamma_curr + 1, length=10))
                compute_ssr(gamma, c_curr) < compute_ssr(gamma_curr, c_curr) && (gamma_curr = gamma)
            end

            # Optimize c
            for c in T.(quantile(state_var, range(0.1, 0.9, length=15)))
                compute_ssr(gamma_curr, c) < compute_ssr(gamma_curr, c_curr) && (c_curr = c)
            end

            abs(gamma_curr - gamma_prev) < 1e-4 && abs(c_curr - c_prev) < 1e-4 &&
                return (gamma=gamma_curr, c=c_curr, F_values=logistic_transition(state_var, gamma_curr, c_curr),
                        convergence_info=(method=:nlls, iterations=iter, converged=true))
        end
        (gamma=gamma_curr, c=c_curr, F_values=logistic_transition(state_var, gamma_curr, c_curr),
         convergence_info=(method=:nlls, iterations=50, converged=false))
    end
end

"""
    estimate_state_lp(Y::AbstractMatrix{T}, shock_var::Int, state_var::AbstractVector{T},
                      horizon::Int; gamma::Union{T,Symbol}=:estimate, ...) -> StateLPModel{T}

Estimate state-dependent LP (Auerbach & Gorodnichenko 2013).
"""
function estimate_state_lp(Y::AbstractMatrix{T}, shock_var::Int, state_var::AbstractVector{T},
                           horizon::Int; gamma::Union{T,Symbol}=:estimate,
                           threshold::Union{T,Symbol}=:estimate,
                           lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                           cov_type::Symbol=:newey_west, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    @assert length(state_var) == T_obs

    # Estimate or set transition parameters
    gamma_val, c_val, F_values = if gamma == :estimate || threshold == :estimate
        result = estimate_transition_params(state_var, Y, shock_var;
                                            c_init=threshold isa Symbol ? :median : threshold)
        (gamma == :estimate ? result.gamma : gamma,
         threshold == :estimate ? result.c : (threshold isa Symbol ? result.c : threshold),
         logistic_transition(state_var, gamma == :estimate ? result.gamma : gamma,
                             threshold == :estimate ? result.c : (threshold isa Symbol ? result.c : threshold)))
    else
        c_fixed = threshold isa Symbol ? (threshold == :median ? median(state_var) :
                                           threshold == :mean ? mean(state_var) : zero(T)) : threshold
        (gamma, c_fixed, logistic_transition(state_var, gamma, c_fixed))
    end

    state_transition = StateTransition(state_var, gamma_val, c_val, :logistic)
    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)
    n_response = length(response_vars)

    B_expansion = Vector{Matrix{T}}(undef, horizon + 1)
    B_recession = Vector{Matrix{T}}(undef, horizon + 1)
    residuals_store = Vector{Matrix{T}}(undef, horizon + 1)
    vcov_expansion = Vector{Matrix{T}}(undef, horizon + 1)
    vcov_recession = Vector{Matrix{T}}(undef, horizon + 1)
    vcov_diff = Vector{Matrix{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)

    k_per_regime = 2 + n * lags

    for h in 0:horizon
        t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
        T_h = t_end - t_start + 1
        T_eff[h + 1] = T_h

        Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)

        # Build state-dependent regressor matrix
        k_total = 2 * k_per_regime
        X_h = Matrix{T}(undef, T_h, k_total)

        @inbounds for (i, t) in enumerate(t_start:t_end)
            F_t = F_values[t]

            # Expansion regime (F_t)
            X_h[i, 1] = F_t
            X_h[i, 2] = F_t * Y[t, shock_var]
            col = 3
            for lag in 1:lags, var in 1:n
                X_h[i, col] = F_t * Y[t - lag, var]
                col += 1
            end

            # Recession regime (1 - F_t)
            X_h[i, k_per_regime + 1] = 1 - F_t
            X_h[i, k_per_regime + 2] = (1 - F_t) * Y[t, shock_var]
            col = k_per_regime + 3
            for lag in 1:lags, var in 1:n
                X_h[i, col] = (1 - F_t) * Y[t - lag, var]
                col += 1
            end
        end

        XtX_inv = robust_inv(X_h' * X_h)
        B_h = XtX_inv * (X_h' * Y_h)
        U_h = Y_h - X_h * B_h

        B_expansion[h + 1] = B_h[1:k_per_regime, :]
        B_recession[h + 1] = B_h[(k_per_regime + 1):end, :]
        residuals_store[h + 1] = U_h

        V_h = compute_block_robust_vcov(X_h, U_h, cov_estimator)

        # Extract regime-specific covariances
        V_exp = zeros(T, k_per_regime * n_response, k_per_regime * n_response)
        V_rec = zeros(T, k_per_regime * n_response, k_per_regime * n_response)
        V_df = zeros(T, k_per_regime * n_response, k_per_regime * n_response)

        for eq in 1:n_response
            eq_start = (eq - 1) * k_total
            exp_idx_full = (eq_start + 1):(eq_start + k_per_regime)
            rec_idx_full = (eq_start + k_per_regime + 1):(eq_start + k_total)
            local_idx = ((eq-1)*k_per_regime + 1):(eq*k_per_regime)

            V_exp[local_idx, local_idx] .= V_h[exp_idx_full, exp_idx_full]
            V_rec[local_idx, local_idx] .= V_h[rec_idx_full, rec_idx_full]
            V_df[local_idx, local_idx] .= V_h[exp_idx_full, exp_idx_full] +
                                           V_h[rec_idx_full, rec_idx_full] -
                                           2 .* V_h[exp_idx_full, rec_idx_full]
        end

        vcov_expansion[h + 1] = V_exp
        vcov_recession[h + 1] = V_rec
        vcov_diff[h + 1] = V_df
    end

    StateLPModel{T}(Matrix{T}(Y), shock_var, response_vars, horizon, lags, state_transition,
                    B_expansion, B_recession, residuals_store, vcov_expansion, vcov_recession,
                    vcov_diff, T_eff, cov_estimator)
end

estimate_state_lp(Y::AbstractMatrix, shock_var::Int, state_var::AbstractVector,
                  horizon::Int; kwargs...) =
    estimate_state_lp(Float64.(Y), shock_var, Float64.(state_var), horizon; kwargs...)

"""
    state_irf(model::StateLPModel{T}; regime::Symbol=:both, conf_level::Real=0.95) -> NamedTuple

Extract state-dependent IRFs.
"""
function state_irf(model::StateLPModel{T}; regime::Symbol=:both,
                   conf_level::Real=0.95) where {T<:AbstractFloat}
    H = model.horizon
    n_response = length(model.response_vars)
    shock_coef_idx = 2
    k_per_regime = size(model.B_expansion[1], 1)

    irf_exp = Matrix{T}(undef, H + 1, n_response)
    irf_rec = Matrix{T}(undef, H + 1, n_response)
    se_exp = Matrix{T}(undef, H + 1, n_response)
    se_rec = Matrix{T}(undef, H + 1, n_response)
    se_diff = Matrix{T}(undef, H + 1, n_response)

    for h in 0:H
        for (j, _) in enumerate(model.response_vars)
            var_idx = (j - 1) * k_per_regime + shock_coef_idx
            irf_exp[h + 1, j] = model.B_expansion[h + 1][shock_coef_idx, j]
            irf_rec[h + 1, j] = model.B_recession[h + 1][shock_coef_idx, j]
            se_exp[h + 1, j] = sqrt(model.vcov_expansion[h + 1][var_idx, var_idx])
            se_rec[h + 1, j] = sqrt(model.vcov_recession[h + 1][var_idx, var_idx])
            se_diff[h + 1, j] = sqrt(model.vcov_diff[h + 1][var_idx, var_idx])
        end
    end

    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    irf_diff = irf_exp .- irf_rec

    regime == :expansion && return (values=irf_exp, ci_lower=irf_exp .- z .* se_exp,
                                     ci_upper=irf_exp .+ z .* se_exp, se=se_exp,
                                     regime=:expansion, conf_level=T(conf_level))
    regime == :recession && return (values=irf_rec, ci_lower=irf_rec .- z .* se_rec,
                                     ci_upper=irf_rec .+ z .* se_rec, se=se_rec,
                                     regime=:recession, conf_level=T(conf_level))
    regime == :difference && return (values=irf_diff, ci_lower=irf_diff .- z .* se_diff,
                                      ci_upper=irf_diff .+ z .* se_diff, se=se_diff,
                                      regime=:difference, conf_level=T(conf_level))

    (expansion=(values=irf_exp, ci_lower=irf_exp .- z .* se_exp, ci_upper=irf_exp .+ z .* se_exp, se=se_exp),
     recession=(values=irf_rec, ci_lower=irf_rec .- z .* se_rec, ci_upper=irf_rec .+ z .* se_rec, se=se_rec),
     difference=(values=irf_diff, ci_lower=irf_diff .- z .* se_diff, ci_upper=irf_diff .+ z .* se_diff, se=se_diff),
     gamma=model.state.gamma, threshold=model.state.threshold, conf_level=T(conf_level))
end

"""
    test_regime_difference(model::StateLPModel{T}; h::Union{Int,Nothing}=nothing) -> NamedTuple

Test whether IRFs differ across regimes.
"""
function test_regime_difference(model::StateLPModel{T}; h::Union{Int,Nothing}=nothing) where {T<:AbstractFloat}
    H = model.horizon
    n_response = length(model.response_vars)
    shock_coef_idx = 2
    k_per_regime = size(model.B_expansion[1], 1)

    if !isnothing(h)
        @assert 0 <= h <= H
        t_stats = Vector{T}(undef, n_response)
        p_values = Vector{T}(undef, n_response)

        for (j, _) in enumerate(model.response_vars)
            diff = model.B_expansion[h + 1][shock_coef_idx, j] - model.B_recession[h + 1][shock_coef_idx, j]
            var_idx = (j - 1) * k_per_regime + shock_coef_idx
            se = sqrt(model.vcov_diff[h + 1][var_idx, var_idx])
            t_stats[j] = diff / se
            p_values[j] = 2 * (1 - cdf(Normal(), abs(t_stats[j])))
        end
        return (horizon=h, t_stats=t_stats, p_values=p_values)
    end

    t_stats = Matrix{T}(undef, H + 1, n_response)
    p_values = Matrix{T}(undef, H + 1, n_response)

    for h_idx in 0:H
        for (j, _) in enumerate(model.response_vars)
            diff = model.B_expansion[h_idx + 1][shock_coef_idx, j] - model.B_recession[h_idx + 1][shock_coef_idx, j]
            var_idx = (j - 1) * k_per_regime + shock_coef_idx
            se = sqrt(model.vcov_diff[h_idx + 1][var_idx, var_idx])
            t_stats[h_idx + 1, j] = diff / se
            p_values[h_idx + 1, j] = 2 * (1 - cdf(Normal(), abs(t_stats[h_idx + 1, j])))
        end
    end

    avg_t = mean(abs.(t_stats))
    (t_stats=t_stats, p_values=p_values, joint_test=(avg_t_stat=avg_t, p_value=2 * (1 - cdf(Normal(), avg_t))))
end

# =============================================================================
# Propensity Score LP (Angrist et al. 2018)
# =============================================================================

"""
    estimate_propensity_score(treatment::AbstractVector{Bool}, X::AbstractMatrix{T};
                              method::Symbol=:logit) -> Vector{T}

Estimate propensity scores P(D=1|X) via logit or probit.
"""
function estimate_propensity_score(treatment::AbstractVector{Bool}, X::AbstractMatrix{T};
                                   method::Symbol=:logit) where {T<:AbstractFloat}
    n = length(treatment)
    X_aug = hcat(ones(T, n), X)
    k = size(X_aug, 2)
    y = T.(treatment)
    beta = zeros(T, k)

    for iter in 1:50
        eta = X_aug * beta
        p, dp = if method == :logit
            (@. 1 / (1 + exp(-eta)), @. 1 / (1 + exp(-eta)) * (1 - 1 / (1 + exp(-eta))))
        elseif method == :probit
            (@. cdf(Normal(), eta), @. pdf(Normal(), eta))
        else
            throw(ArgumentError("method must be :logit or :probit"))
        end

        p = clamp.(p, T(1e-10), T(1 - 1e-10))
        dp = max.(dp, T(1e-10))

        score = X_aug' * (y .- p)
        hessian = X_aug' * Diagonal(dp) * X_aug
        delta = hessian \ score

        maximum(abs.(delta)) < 1e-8 && break
        beta .+= delta
    end

    eta = X_aug * beta
    p = method == :logit ? (@. 1 / (1 + exp(-eta))) : (@. cdf(Normal(), eta))
    clamp.(p, T(1e-10), T(1 - 1e-10))
end

estimate_propensity_score(treatment::AbstractVector{<:Integer}, X::AbstractMatrix{T};
                          method::Symbol=:logit) where {T<:AbstractFloat} =
    estimate_propensity_score(treatment .!= 0, X; method=method)

"""
    inverse_propensity_weights(treatment::AbstractVector{Bool}, propensity::AbstractVector{T};
                               trimming::Tuple{T,T}=(T(0.01), T(0.99)), normalize::Bool=true) -> Vector{T}

Compute IPW weights with optional trimming.
"""
function inverse_propensity_weights(treatment::AbstractVector{Bool}, propensity::AbstractVector{T};
                                    trimming::Tuple{T,T}=(T(0.01), T(0.99)),
                                    normalize::Bool=true) where {T<:AbstractFloat}
    p_trimmed = clamp.(propensity, trimming[1], trimming[2])
    weights = [t ? one(T) / p : one(T) / (one(T) - p) for (t, p) in zip(treatment, p_trimmed)]

    if normalize
        for (idx, mask) in [(findall(treatment), true), (findall(.!treatment), false)]
            isempty(idx) && continue
            w_sum = sum(weights[idx])
            weights[idx] ./= w_sum
            weights[idx] .*= length(idx)
        end
    end
    weights
end

"""
    estimate_propensity_lp(Y::AbstractMatrix{T}, treatment::AbstractVector{Bool},
                           covariates::AbstractMatrix{T}, horizon::Int; ...) -> PropensityLPModel{T}

Estimate LP with Inverse Propensity Weighting (Angrist et al. 2018).
"""
function estimate_propensity_lp(Y::AbstractMatrix{T}, treatment::AbstractVector{Bool},
                                covariates::AbstractMatrix{T}, horizon::Int;
                                ps_method::Symbol=:logit,
                                trimming::Tuple{T,T}=(T(0.01), T(0.99)),
                                lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                                cov_type::Symbol=:newey_west, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    @assert length(treatment) == T_obs

    propensity = estimate_propensity_score(treatment, covariates; method=ps_method)
    ipw_weights = inverse_propensity_weights(treatment, propensity; trimming=trimming)
    config = PropensityScoreConfig{T}(ps_method, trimming, true)
    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)
    n_response = length(response_vars)
    n_cov = size(covariates, 2)

    B = Vector{Matrix{T}}(undef, horizon + 1)
    residuals_store = Vector{Matrix{T}}(undef, horizon + 1)
    vcov_vec = Vector{Matrix{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)
    ate = Matrix{T}(undef, horizon + 1, n_response)
    ate_se = Matrix{T}(undef, horizon + 1, n_response)

    for h in 0:horizon
        t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
        T_h = t_end - t_start + 1
        T_eff[h + 1] = T_h

        Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)
        D_h = treatment[t_start:t_end]
        w_h = ipw_weights[t_start:t_end]

        # Build regressor matrix: [1, D, lagged_Y, covariates]
        k = 2 + n * lags + n_cov
        X_h = Matrix{T}(undef, T_h, k)

        @inbounds for (i, t) in enumerate(t_start:t_end)
            X_h[i, 1] = one(T)
            X_h[i, 2] = D_h[i] ? one(T) : zero(T)
            col = 3
            for lag in 1:lags, var in 1:n
                X_h[i, col] = Y[t - lag, var]
                col += 1
            end
            for cov in 1:n_cov
                X_h[i, col] = covariates[t, cov]
                col += 1
            end
        end

        # Weighted least squares
        W = Diagonal(w_h)
        XtWX_inv = robust_inv(X_h' * W * X_h)
        B_h = XtWX_inv * (X_h' * W * Y_h)
        U_h = Y_h - X_h * B_h

        B[h + 1] = B_h
        residuals_store[h + 1] = U_h

        for (j, _) in enumerate(response_vars)
            ate[h + 1, j] = B_h[2, j]
        end

        # Robust covariance with weights
        V_h = zeros(T, k * n_response, k * n_response)
        for eq in 1:n_response
            weighted_u = sqrt.(w_h) .* @view(U_h[:, eq])
            weighted_X = sqrt.(w_h) .* X_h
            V_eq = robust_vcov(weighted_X, weighted_u, cov_estimator)
            idx = ((eq-1)*k + 1):(eq*k)
            V_h[idx, idx] .= V_eq
        end
        vcov_vec[h + 1] = V_h

        for (j, _) in enumerate(response_vars)
            ate_se[h + 1, j] = sqrt(V_h[(j - 1) * k + 2, (j - 1) * k + 2])
        end
    end

    PropensityLPModel{T}(Matrix{T}(Y), treatment, response_vars, Matrix{T}(covariates),
                         horizon, propensity, ipw_weights, B, residuals_store, vcov_vec,
                         ate, ate_se, config, T_eff, cov_estimator)
end

estimate_propensity_lp(Y::AbstractMatrix, treatment::AbstractVector, covariates::AbstractMatrix,
                       horizon::Int; kwargs...) =
    estimate_propensity_lp(Float64.(Y), Bool.(treatment .!= 0), Float64.(covariates), horizon; kwargs...)

"""
    doubly_robust_lp(Y::AbstractMatrix{T}, treatment::AbstractVector{Bool},
                     covariates::AbstractMatrix{T}, horizon::Int; ...) -> PropensityLPModel{T}

Doubly robust LP estimator combining IPW and regression adjustment.
"""
function doubly_robust_lp(Y::AbstractMatrix{T}, treatment::AbstractVector{Bool},
                          covariates::AbstractMatrix{T}, horizon::Int;
                          ps_method::Symbol=:logit, trimming::Tuple{T,T}=(T(0.01), T(0.99)),
                          lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                          cov_type::Symbol=:newey_west, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    n_response = length(response_vars)
    n_cov = size(covariates, 2)

    propensity = estimate_propensity_score(treatment, covariates; method=ps_method)
    p_trimmed = clamp.(propensity, trimming[1], trimming[2])
    ipw_weights = inverse_propensity_weights(treatment, propensity; trimming=trimming)
    config = PropensityScoreConfig{T}(ps_method, trimming, true)
    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)

    B = Vector{Matrix{T}}(undef, horizon + 1)
    residuals_store = Vector{Matrix{T}}(undef, horizon + 1)
    vcov_vec = Vector{Matrix{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)
    ate = Matrix{T}(undef, horizon + 1, n_response)
    ate_se = Matrix{T}(undef, horizon + 1, n_response)

    for h in 0:horizon
        t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
        T_h = t_end - t_start + 1
        T_eff[h + 1] = T_h

        Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)
        D_h = treatment[t_start:t_end]
        p_h = p_trimmed[t_start:t_end]

        # Covariates for outcome regression
        k_cov = 1 + n * lags + n_cov
        X_cov = Matrix{T}(undef, T_h, k_cov)
        @inbounds for (i, t) in enumerate(t_start:t_end)
            X_cov[i, 1] = one(T)
            col = 2
            for lag in 1:lags, var in 1:n
                X_cov[i, col] = Y[t - lag, var]
                col += 1
            end
            for cov in 1:n_cov
                X_cov[i, col] = covariates[t, cov]
                col += 1
            end
        end

        treated_idx = findall(D_h)
        control_idx = findall(.!D_h)

        ate_dr = Vector{T}(undef, n_response)
        ate_se_h = Vector{T}(undef, n_response)

        for (j, _) in enumerate(response_vars)
            y_j = Y_h[:, j]

            # Outcome regressions
            mu1 = length(treated_idx) > k_cov ?
                  X_cov * (robust_inv(X_cov[treated_idx, :]' * X_cov[treated_idx, :]) *
                           (X_cov[treated_idx, :]' * y_j[treated_idx])) :
                  fill(mean(y_j[treated_idx]), T_h)

            mu0 = length(control_idx) > k_cov ?
                  X_cov * (robust_inv(X_cov[control_idx, :]' * X_cov[control_idx, :]) *
                           (X_cov[control_idx, :]' * y_j[control_idx])) :
                  fill(mean(y_j[control_idx]), T_h)

            # Doubly robust estimator
            psi = [D_h[i] ? (y_j[i] - mu1[i]) / p_h[i] + mu1[i] - mu0[i] :
                            mu1[i] - (y_j[i] - mu0[i]) / (1 - p_h[i]) - mu0[i] for i in 1:T_h]

            ate_dr[j] = mean(psi)
            ate_se_h[j] = std(psi) / sqrt(T_h)
        end

        ate[h + 1, :] = ate_dr
        ate_se[h + 1, :] = ate_se_h

        # Store regression coefficients (IPW version)
        k = 2 + n * lags + n_cov
        X_full = hcat(ones(T, T_h), T.(D_h), X_cov[:, 2:end])
        W = Diagonal(ipw_weights[t_start:t_end])
        B[h + 1] = robust_inv(X_full' * W * X_full) * (X_full' * W * Y_h)
        residuals_store[h + 1] = Y_h - X_full * B[h + 1]

        vcov_vec[h + 1] = zeros(T, k * n_response, k * n_response)
        for (j, _) in enumerate(response_vars)
            vcov_vec[h + 1][(j-1)*k + 2, (j-1)*k + 2] = ate_se_h[j]^2
        end
    end

    PropensityLPModel{T}(Matrix{T}(Y), treatment, response_vars, Matrix{T}(covariates),
                         horizon, propensity, ipw_weights, B, residuals_store, vcov_vec,
                         ate, ate_se, config, T_eff, cov_estimator)
end

"""
    propensity_irf(model::PropensityLPModel{T}; conf_level::Real=0.95) -> LPImpulseResponse{T}

Extract treatment effect (ATE) IRF from PropensityLPModel.
"""
function propensity_irf(model::PropensityLPModel{T}; conf_level::Real=0.95) where {T<:AbstractFloat}
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = model.ate .- z .* model.ate_se
    ci_upper = model.ate .+ z .* model.ate_se

    response_names = default_var_names(length(model.response_vars); prefix="Var")
    cov_type_sym = model.cov_estimator isa NeweyWestEstimator ? :newey_west : :white

    LPImpulseResponse{T}(model.ate, ci_lower, ci_upper, model.ate_se, model.horizon,
                         response_names, "Treatment Effect (ATE)", cov_type_sym, T(conf_level))
end

"""
    propensity_diagnostics(model::PropensityLPModel{T}) -> NamedTuple

Propensity score diagnostics (overlap, balance).
"""
function propensity_diagnostics(model::PropensityLPModel{T}) where {T<:AbstractFloat}
    p = model.propensity_scores
    w = model.ipw_weights
    treated_idx = findall(model.treatment)
    control_idx = findall(.!model.treatment)

    ps_summary = (treated=(min=minimum(p[treated_idx]), max=maximum(p[treated_idx]),
                           mean=mean(p[treated_idx]), std=std(p[treated_idx])),
                  control=(min=minimum(p[control_idx]), max=maximum(p[control_idx]),
                           mean=mean(p[control_idx]), std=std(p[control_idx])))

    overlap_min = max(minimum(p[treated_idx]), minimum(p[control_idx]))
    overlap_max = min(maximum(p[treated_idx]), maximum(p[control_idx]))
    overlap = (common_support=(overlap_min, overlap_max),
               treated_in_support=mean((p[treated_idx] .>= overlap_min) .& (p[treated_idx] .<= overlap_max)),
               control_in_support=mean((p[control_idx] .>= overlap_min) .& (p[control_idx] .<= overlap_max)))

    X = model.covariates
    n_cov = size(X, 2)
    balance_raw = Vector{T}(undef, n_cov)
    balance_weighted = Vector{T}(undef, n_cov)

    for j in 1:n_cov
        x = X[:, j]
        pooled_sd = sqrt((var(x[treated_idx]) + var(x[control_idx])) / 2)
        balance_raw[j] = (mean(x[treated_idx]) - mean(x[control_idx])) / pooled_sd

        w_t, w_c = w[treated_idx], w[control_idx]
        mean_t_w = sum(w_t .* x[treated_idx]) / sum(w_t)
        mean_c_w = sum(w_c .* x[control_idx]) / sum(w_c)
        balance_weighted[j] = (mean_t_w - mean_c_w) / pooled_sd
    end

    (propensity_summary=ps_summary, overlap=overlap,
     balance=(raw_smd=balance_raw, weighted_smd=balance_weighted,
              max_raw=maximum(abs.(balance_raw)), max_weighted=maximum(abs.(balance_weighted))))
end
