"""
Core Local Projection estimation and HAC covariance estimators.

This module provides:
- HAC covariance estimators (Newey-West, White)
- Shared utility functions for all LP variants
- Core LP estimation (Jordà 2005)
- IRF extraction and cumulative IRF

References:
- Jordà, Ò. (2005). "Estimation and Inference of Impulse Responses by Local Projections."
- Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix."
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Shared Utility Functions
# =============================================================================

"""
    create_cov_estimator(cov_type::Symbol, ::Type{T}; bandwidth::Int=0) where T

Create covariance estimator from symbol specification.
Eliminates repeated if/else patterns across LP variants.
"""
function create_cov_estimator(cov_type::Symbol, ::Type{T}; bandwidth::Int=0) where {T<:AbstractFloat}
    if cov_type == :newey_west
        NeweyWestEstimator{T}(bandwidth, :bartlett, false)
    elseif cov_type == :white
        WhiteEstimator()
    elseif cov_type == :driscoll_kraay
        DriscollKraayEstimator{T}(bandwidth, :bartlett)
    else
        throw(ArgumentError("cov_type must be :newey_west, :white, or :driscoll_kraay"))
    end
end

"""
    compute_horizon_bounds(T_obs::Int, h::Int, lags::Int) -> (t_start, t_end)

Compute valid observation bounds for horizon h.
"""
function compute_horizon_bounds(T_obs::Int, h::Int, lags::Int)
    t_start = lags + 1
    t_end = T_obs - h
    if t_end < t_start
        throw(ArgumentError("Not enough observations for horizon $h with $lags lags"))
    end
    (t_start, t_end)
end

"""
    build_response_matrix(Y::AbstractMatrix{T}, h::Int, t_start::Int, t_end::Int,
                          response_vars::Vector{Int}) where T

Build response matrix Y_h at horizon h.
"""
function build_response_matrix(Y::AbstractMatrix{T}, h::Int, t_start::Int, t_end::Int,
                                response_vars::Vector{Int}) where {T<:AbstractFloat}
    T_eff = t_end - t_start + 1
    n_response = length(response_vars)
    Y_h = Matrix{T}(undef, T_eff, n_response)
    @inbounds for (j, var) in enumerate(response_vars)
        for (i, t) in enumerate(t_start:t_end)
            Y_h[i, j] = Y[t + h, var]
        end
    end
    Y_h
end

"""
    build_control_columns!(X_h::AbstractMatrix{T}, Y::AbstractMatrix{T},
                           t_start::Int, t_end::Int, lags::Int, start_col::Int) where T

Fill control (lagged Y) columns into regressor matrix X_h.
Returns the next available column index.
"""
function build_control_columns!(X_h::AbstractMatrix{T}, Y::AbstractMatrix{T},
                                 t_start::Int, t_end::Int, lags::Int, start_col::Int) where {T<:AbstractFloat}
    n = size(Y, 2)
    col = start_col
    @inbounds for (i, t) in enumerate(t_start:t_end)
        col_local = start_col
        for lag in 1:lags
            for var in 1:n
                X_h[i, col_local] = Y[t - lag, var]
                col_local += 1
            end
        end
    end
    start_col + n * lags
end

"""
    compute_block_robust_vcov(X::AbstractMatrix{T}, U::AbstractMatrix{T},
                              cov_estimator::AbstractCovarianceEstimator) where T

Compute block-diagonal robust covariance for multi-equation system.
"""
function compute_block_robust_vcov(X::AbstractMatrix{T}, U::AbstractMatrix{T},
                                    cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
    n_eq = size(U, 2)
    k = size(X, 2)
    V = zeros(T, k * n_eq, k * n_eq)
    @inbounds for eq in 1:n_eq
        V_eq = robust_vcov(X, @view(U[:, eq]), cov_estimator)
        idx = ((eq-1)*k + 1):(eq*k)
        V[idx, idx] .= V_eq
    end
    V
end

"""
    extract_shock_irf(B::Vector{Matrix{T}}, vcov::Vector{Matrix{T}},
                      response_vars::Vector{Int}, shock_coef_idx::Int;
                      conf_level::Real=0.95) where T

Generic IRF extraction from coefficient and covariance vectors.
Works for LPModel, LPIVModel, PropensityLPModel.
"""
function extract_shock_irf(B::Vector{Matrix{T}}, vcov::Vector{Matrix{T}},
                           response_vars::Vector{Int}, shock_coef_idx::Int;
                           conf_level::Real=0.95) where {T<:AbstractFloat}
    H = length(B) - 1
    n_response = length(response_vars)
    k = size(B[1], 1)

    values = Matrix{T}(undef, H + 1, n_response)
    se = Matrix{T}(undef, H + 1, n_response)

    @inbounds for h in 0:H
        B_h = B[h + 1]
        V_h = vcov[h + 1]
        for (j, _) in enumerate(response_vars)
            values[h + 1, j] = B_h[shock_coef_idx, j]
            var_idx = (j - 1) * k + shock_coef_idx
            se[h + 1, j] = sqrt(V_h[var_idx, var_idx])
        end
    end

    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = values .- z .* se
    ci_upper = values .+ z .* se

    (values=values, se=se, ci_lower=ci_lower, ci_upper=ci_upper)
end

# =============================================================================
# Kernel Functions
# =============================================================================

"""
    kernel_weight(j::Int, bandwidth::Int, kernel::Symbol, ::Type{T}=Float64) -> T

Compute kernel weight for lag j given bandwidth and kernel type.

Kernels:
- :bartlett (Newey-West): w(x) = 1 - |x| for |x| ≤ 1
- :parzen: quartic spline kernel
- :quadratic_spectral (Andrews): optimal for Gaussian data
- :tukey_hanning: cosine kernel
"""
function kernel_weight(j::Int, bandwidth::Int, kernel::Symbol, ::Type{T}=Float64) where {T<:AbstractFloat}
    bandwidth == 0 && return zero(T)
    x = T(j) / T(bandwidth + 1)
    abs(x) > 1 && return zero(T)

    if kernel == :bartlett
        one(T) - abs(x)
    elseif kernel == :parzen
        ax = abs(x)
        ax <= 0.5 ? one(T) - 6ax^2 + 6ax^3 : 2(one(T) - ax)^3
    elseif kernel == :quadratic_spectral
        j == 0 && return one(T)
        z = 6π * x / 5
        25 / (12π^2 * x^2) * (sin(z) / z - cos(z))
    elseif kernel == :tukey_hanning
        (one(T) + cos(π * x)) / 2
    else
        throw(ArgumentError("Unknown kernel: $kernel"))
    end
end

# =============================================================================
# Automatic Bandwidth Selection
# =============================================================================

"""
    optimal_bandwidth_nw(residuals::AbstractVector{T}) -> Int

Compute optimal bandwidth using Newey-West (1994) automatic selection.
"""
function optimal_bandwidth_nw(residuals::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(residuals)
    n < 4 && return 0

    # Estimate AR(1) coefficient
    r_lag = @view residuals[1:end-1]
    r_lead = @view residuals[2:end]
    rho = dot(r_lag, r_lead) / dot(r_lag, r_lag)

    # Newey-West (1994) formula for Bartlett kernel
    rho_abs = min(abs(rho), T(0.99))
    alpha = 4rho_abs^2 / (1 - rho_abs)^4
    m = ceil(Int, 1.1447 * (alpha * n)^(1/3))
    min(m, floor(Int, n^(1/3)))
end

"""
    optimal_bandwidth_nw(residuals::AbstractMatrix{T}) -> Int

Multivariate version: average optimal bandwidth across columns.
"""
function optimal_bandwidth_nw(residuals::AbstractMatrix{T}) where {T<:AbstractFloat}
    n_vars = size(residuals, 2)
    n_vars == 0 && return 0
    round(Int, mean(optimal_bandwidth_nw(@view residuals[:, j]) for j in 1:n_vars))
end

# =============================================================================
# Newey-West HAC Estimator
# =============================================================================

"""
    newey_west(X::AbstractMatrix{T}, residuals::AbstractVector{T};
               bandwidth::Int=0, kernel::Symbol=:bartlett, prewhiten::Bool=false) -> Matrix{T}

Compute Newey-West HAC covariance matrix.

V_NW = (X'X)^{-1} S (X'X)^{-1}
where S = Γ₀ + Σⱼ₌₁ᵐ w(j) (Γⱼ + Γⱼ')
"""
function newey_west(X::AbstractMatrix{T}, residuals::AbstractVector{T};
                    bandwidth::Int=0, kernel::Symbol=:bartlett,
                    prewhiten::Bool=false) where {T<:AbstractFloat}
    n, k = size(X)
    @assert length(residuals) == n "X and residuals must have same number of rows"

    bw = bandwidth == 0 ? optimal_bandwidth_nw(residuals) : bandwidth

    # Prewhitening
    u, X_use = if prewhiten && n > 2
        u_lag = @view residuals[1:end-1]
        u_lead = @view residuals[2:end]
        rho = dot(u_lag, u_lead) / dot(u_lag, u_lag)
        u_pw = residuals[2:end] .- rho .* residuals[1:end-1]
        (vcat([residuals[1]], u_pw), X)
    else
        (residuals, X)
    end

    XtX_inv = robust_inv(X_use' * X_use)

    # Compute S = long-run variance of X'u
    S = zeros(T, k, k)
    @inbounds for t in 1:n
        xu = @view(X_use[t, :]) * u[t]
        S .+= xu * xu'
    end

    @inbounds for j in 1:bw
        w = kernel_weight(j, bw, kernel, T)
        w == 0 && continue
        Gamma_j = zeros(T, k, k)
        for t in (j+1):n
            xu_t = @view(X_use[t, :]) * u[t]
            xu_tj = @view(X_use[t-j, :]) * u[t-j]
            Gamma_j .+= xu_t * xu_tj'
        end
        S .+= w * (Gamma_j + Gamma_j')
    end

    # Recolor if prewhitened
    if prewhiten && n > 2
        u_lag = @view residuals[1:end-1]
        u_lead = @view residuals[2:end]
        rho = dot(u_lag, u_lead) / dot(u_lag, u_lag)
        S ./= (1 - rho)^2
    end

    S ./= n
    XtX_inv * S * XtX_inv
end

"""
    newey_west(X::AbstractMatrix{T}, residuals::AbstractMatrix{T}; ...) -> Matrix{T}

Multivariate version for systems of equations.
"""
function newey_west(X::AbstractMatrix{T}, residuals::AbstractMatrix{T};
                    bandwidth::Int=0, kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n, n_eq = size(residuals)
    n_eq == 1 && return newey_west(X, vec(residuals); bandwidth, kernel)

    k = size(X, 2)
    V_full = zeros(T, k * n_eq, k * n_eq)
    for eq in 1:n_eq
        V_eq = newey_west(X, @view(residuals[:, eq]); bandwidth, kernel)
        idx = ((eq-1)*k + 1):(eq*k)
        V_full[idx, idx] .= V_eq
    end
    V_full
end

# =============================================================================
# White Heteroscedasticity-Robust Estimator
# =============================================================================

"""
    white_vcov(X::AbstractMatrix{T}, residuals::AbstractVector{T}; variant::Symbol=:hc0) -> Matrix{T}

White heteroscedasticity-robust covariance estimator.

Variants: :hc0, :hc1, :hc2, :hc3
"""
function white_vcov(X::AbstractMatrix{T}, residuals::AbstractVector{T};
                    variant::Symbol=:hc0) where {T<:AbstractFloat}
    n, k = size(X)
    @assert length(residuals) == n

    XtX_inv = robust_inv(X' * X)

    # Compute leverage if needed
    h_diag = if variant in (:hc2, :hc3)
        diag(X * XtX_inv * X')
    else
        nothing
    end

    Omega = zeros(T, k, k)
    @inbounds for t in 1:n
        u2 = residuals[t]^2
        u2_adj = if variant == :hc0
            u2
        elseif variant == :hc1
            u2 * T(n) / T(n - k)
        elseif variant == :hc2
            u2 / (1 - h_diag[t])
        elseif variant == :hc3
            u2 / (1 - h_diag[t])^2
        else
            throw(ArgumentError("Unknown HC variant: $variant"))
        end
        x_t = @view X[t, :]
        Omega .+= u2_adj * (x_t * x_t')
    end

    XtX_inv * Omega * XtX_inv
end

"""
    white_vcov(X::AbstractMatrix{T}, residuals::AbstractMatrix{T}; ...) -> Matrix{T}

Multivariate version.
"""
function white_vcov(X::AbstractMatrix{T}, residuals::AbstractMatrix{T};
                    variant::Symbol=:hc0) where {T<:AbstractFloat}
    n, n_eq = size(residuals)
    n_eq == 1 && return white_vcov(X, vec(residuals); variant)

    k = size(X, 2)
    V_full = zeros(T, k * n_eq, k * n_eq)
    for eq in 1:n_eq
        V_eq = white_vcov(X, @view(residuals[:, eq]); variant)
        idx = ((eq-1)*k + 1):(eq*k)
        V_full[idx, idx] .= V_eq
    end
    V_full
end

# =============================================================================
# Covariance Estimator Dispatch
# =============================================================================

"""
    robust_vcov(X::AbstractMatrix{T}, residuals::AbstractVecOrMat{T},
                estimator::AbstractCovarianceEstimator) -> Matrix{T}

Dispatch to appropriate covariance estimator.
"""
function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractVector{T},
                     estimator::NeweyWestEstimator) where {T<:AbstractFloat}
    newey_west(X, residuals; bandwidth=estimator.bandwidth, kernel=estimator.kernel,
               prewhiten=estimator.prewhiten)
end

function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractVector{T},
                     estimator::WhiteEstimator) where {T<:AbstractFloat}
    white_vcov(X, residuals)
end

function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractMatrix{T},
                     estimator::NeweyWestEstimator) where {T<:AbstractFloat}
    newey_west(X, residuals; bandwidth=estimator.bandwidth, kernel=estimator.kernel)
end

function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractMatrix{T},
                     estimator::WhiteEstimator) where {T<:AbstractFloat}
    white_vcov(X, residuals)
end

"""
    driscoll_kraay(X::AbstractMatrix{T}, u::AbstractVector{T};
                   bandwidth::Int=0, kernel::Symbol=:bartlett) -> Matrix{T}

Driscoll-Kraay standard errors for time series regression.

In a pure time series context, this is equivalent to Newey-West HAC estimation
applied to the moment conditions X'u. For panel data applications, it would
average across cross-sectional units first, but here we treat the data as a
single time series.

# Arguments
- `X`: Design matrix (T × k)
- `u`: Residuals vector (T × 1)
- `bandwidth`: Bandwidth for kernel. If 0, uses optimal bandwidth selection.
- `kernel`: Kernel function (:bartlett, :parzen, :quadratic_spectral, :tukey_hanning)

# Returns
- Robust covariance matrix (k × k)

# References
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation
  with spatially dependent panel data. Review of Economics and Statistics.
"""
function driscoll_kraay(X::AbstractMatrix{T}, u::AbstractVector{T};
                        bandwidth::Int=0, kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n, k = size(X)

    # Moment conditions: g_t = X_t' * u_t (k × 1 for each t)
    # In time series context, this is just the score contribution at each t
    G = X .* u  # T × k matrix of moment contributions

    # Compute (X'X)^(-1)
    XtX = X' * X
    XtX_inv = robust_inv(XtX)

    # Compute long-run covariance of moment conditions
    # S = Σⱼ₌₋∞^∞ E[gₜgₜ₋ⱼ']
    S = long_run_covariance(G; bandwidth=bandwidth, kernel=kernel)

    # Sandwich formula: V = n * (X'X)^(-1) * S * (X'X)^(-1)
    V = n * XtX_inv * S * XtX_inv

    # Ensure symmetry
    (V + V') / 2
end

"""
    driscoll_kraay(X::AbstractMatrix{T}, U::AbstractMatrix{T};
                   bandwidth::Int=0, kernel::Symbol=:bartlett) -> Matrix{T}

Driscoll-Kraay standard errors for multi-equation system.

# Arguments
- `X`: Design matrix (T × k)
- `U`: Residuals matrix (T × n_eq)
- `bandwidth`: Bandwidth for kernel
- `kernel`: Kernel function

# Returns
- Block-diagonal robust covariance matrix (k*n_eq × k*n_eq)
"""
function driscoll_kraay(X::AbstractMatrix{T}, U::AbstractMatrix{T};
                        bandwidth::Int=0, kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n, k = size(X)
    n_eq = size(U, 2)

    V = zeros(T, k * n_eq, k * n_eq)
    @inbounds for eq in 1:n_eq
        V_eq = driscoll_kraay(X, @view(U[:, eq]); bandwidth=bandwidth, kernel=kernel)
        idx = ((eq-1)*k + 1):(eq*k)
        V[idx, idx] .= V_eq
    end
    V
end

# Dispatch for DriscollKraayEstimator
function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractVector{T},
                     estimator::DriscollKraayEstimator) where {T<:AbstractFloat}
    driscoll_kraay(X, residuals; bandwidth=estimator.bandwidth, kernel=estimator.kernel)
end

function robust_vcov(X::AbstractMatrix{T}, residuals::AbstractMatrix{T},
                     estimator::DriscollKraayEstimator) where {T<:AbstractFloat}
    driscoll_kraay(X, residuals; bandwidth=estimator.bandwidth, kernel=estimator.kernel)
end

# =============================================================================
# Long-Run Variance Estimation
# =============================================================================

"""
    long_run_variance(x::AbstractVector{T}; bandwidth::Int=0, kernel::Symbol=:bartlett) -> T

Estimate long-run variance: S = Σⱼ₌₋∞^∞ γⱼ
"""
function long_run_variance(x::AbstractVector{T}; bandwidth::Int=0,
                           kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n = length(x)
    n < 2 && return var(x)

    bw = bandwidth == 0 ? optimal_bandwidth_nw(x) : bandwidth
    x_demean = x .- mean(x)
    S = sum(x_demean.^2) / n

    @inbounds for j in 1:bw
        w = kernel_weight(j, bw, kernel, T)
        w == 0 && continue
        gamma_j = sum(x_demean[j+1:n] .* x_demean[1:n-j]) / n
        S += 2w * gamma_j
    end

    max(S, zero(T))
end

"""
    long_run_covariance(X::AbstractMatrix{T}; bandwidth::Int=0, kernel::Symbol=:bartlett) -> Matrix{T}

Estimate long-run covariance matrix of multivariate time series.
"""
function long_run_covariance(X::AbstractMatrix{T}; bandwidth::Int=0,
                             kernel::Symbol=:bartlett) where {T<:AbstractFloat}
    n, k = size(X)
    n < 2 && return cov(X)

    bw = bandwidth == 0 ? optimal_bandwidth_nw(X) : bandwidth
    X_demean = X .- mean(X, dims=1)
    S = (X_demean' * X_demean) / n

    @inbounds for j in 1:bw
        w = kernel_weight(j, bw, kernel, T)
        w == 0 && continue
        Gamma_j = (X_demean[j+1:n, :]' * X_demean[1:n-j, :]) / n
        S .+= w * (Gamma_j + Gamma_j')
    end

    # Ensure positive semi-definite
    S_sym = Hermitian((S + S') / 2)
    eigvals_S = eigen(S_sym).values
    if minimum(eigvals_S) < 0
        F = eigen(S_sym)
        D = max.(F.values, zero(T))
        S = F.vectors * Diagonal(D) * F.vectors'
    end

    Matrix(S)
end

# =============================================================================
# LP Matrix Construction
# =============================================================================

"""
    construct_lp_matrices(Y::AbstractMatrix{T}, shock_var::Int, h::Int, lags::Int;
                          response_vars::Vector{Int}=collect(1:size(Y,2))) where T

Construct regressor and response matrices for LP regression at horizon h.

Returns: (Y_h, X_h, valid_idx)
"""
function construct_lp_matrices(Y::AbstractMatrix{T}, shock_var::Int, h::Int, lags::Int;
                                response_vars::Vector{Int}=collect(1:size(Y, 2))) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
    T_eff = t_end - t_start + 1

    # Response matrix
    Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)

    # Regressor matrix: [1, shock_t, y_{t-1}, ..., y_{t-lags}]
    k = 2 + n * lags
    X_h = Matrix{T}(undef, T_eff, k)

    @inbounds for (i, t) in enumerate(t_start:t_end)
        X_h[i, 1] = one(T)
        X_h[i, 2] = Y[t, shock_var]
    end
    build_control_columns!(X_h, Y, t_start, t_end, lags, 3)

    valid_idx = collect(t_start:t_end)
    (Y_h, X_h, valid_idx)
end

# =============================================================================
# Core LP Estimation
# =============================================================================

"""
    estimate_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y,2)),
                cov_type::Symbol=:newey_west, bandwidth::Int=0,
                conf_level::Real=0.95) -> LPModel{T}

Estimate Local Projection impulse response functions (Jordà 2005).

The LP regression for horizon h:
    y_{t+h} = α_h + β_h * shock_t + Γ_h * controls_t + ε_{t+h}
"""
function estimate_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                     lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                     cov_type::Symbol=:newey_west, bandwidth::Int=0,
                     conf_level::Real=0.95) where {T<:AbstractFloat}
    T_obs, n = size(Y)

    validate_positive(horizon, "horizon")
    @assert 1 <= shock_var <= n "shock_var must be in 1:$n"
    @assert all(1 .<= response_vars .<= n) "response_vars must be in 1:$n"
    @assert lags >= 0 "lags must be non-negative"
    @assert T_obs > lags + horizon + 1 "Not enough observations"

    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)

    B = Vector{Matrix{T}}(undef, horizon + 1)
    residuals = Vector{Matrix{T}}(undef, horizon + 1)
    vcov = Vector{Matrix{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)

    for h in 0:horizon
        Y_h, X_h, valid_idx = construct_lp_matrices(Y, shock_var, h, lags;
                                                     response_vars=response_vars)
        T_eff[h + 1] = length(valid_idx)

        # OLS: B_h = (X'X)^{-1} X'Y
        XtX_inv = robust_inv(X_h' * X_h)
        B_h = XtX_inv * (X_h' * Y_h)
        U_h = Y_h - X_h * B_h

        B[h + 1] = B_h
        residuals[h + 1] = U_h
        vcov[h + 1] = compute_block_robust_vcov(X_h, U_h, cov_estimator)
    end

    LPModel(Matrix{T}(Y), shock_var, response_vars, horizon, lags,
            B, residuals, vcov, T_eff, cov_estimator)
end

# Float fallback
estimate_lp(Y::AbstractMatrix, shock_var::Int, horizon::Int; kwargs...) =
    estimate_lp(Float64.(Y), shock_var, horizon; kwargs...)

# =============================================================================
# Multiple Shocks
# =============================================================================

"""
    estimate_lp_multi(Y::AbstractMatrix{T}, shock_vars::Vector{Int}, horizon::Int;
                      kwargs...) -> Vector{LPModel{T}}

Estimate LP for multiple shock variables.
"""
function estimate_lp_multi(Y::AbstractMatrix{T}, shock_vars::Vector{Int}, horizon::Int;
                           kwargs...) where {T<:AbstractFloat}
    [estimate_lp(Y, shock, horizon; kwargs...) for shock in shock_vars]
end

# =============================================================================
# LP with Orthogonalized Shocks
# =============================================================================

"""
    estimate_lp_cholesky(Y::AbstractMatrix{T}, horizon::Int;
                         lags::Int=4, cov_type::Symbol=:newey_west, kwargs...) -> Vector{LPModel{T}}

Estimate LP with Cholesky-orthogonalized shocks.
"""
function estimate_lp_cholesky(Y::AbstractMatrix{T}, horizon::Int;
                              lags::Int=4, cov_type::Symbol=:newey_west,
                              kwargs...) where {T<:AbstractFloat}
    T_obs, n = size(Y)

    var_model = estimate_var(Y, lags)
    U = var_model.U
    L = identify_cholesky(var_model)
    eps = (inv(L) * U')'

    Y_eff = Y[(lags+1):end, :]
    @assert size(eps, 1) == size(Y_eff, 1) "Dimension mismatch"

    models = Vector{LPModel{T}}(undef, n)
    for shock in 1:n
        Y_aug = hcat(eps[:, shock], Y_eff)
        models[shock] = estimate_lp(Y_aug, 1, horizon; lags=lags,
                                     response_vars=collect(2:(n+1)),
                                     cov_type=cov_type, kwargs...)
    end
    models
end

# =============================================================================
# Model Comparison
# =============================================================================

"""
    compare_var_lp(Y::AbstractMatrix{T}, horizon::Int; lags::Int=4) where T

Compare VAR-based and LP-based impulse responses.
"""
function compare_var_lp(Y::AbstractMatrix{T}, horizon::Int; lags::Int=4) where {T<:AbstractFloat}
    n = size(Y, 2)

    var_model = estimate_var(Y, lags)
    var_result = irf(var_model, horizon; method=:cholesky)

    lp_models = estimate_lp_cholesky(Y, horizon; lags=lags)
    lp_results = [lp_irf(m) for m in lp_models]

    var_values = var_result.values
    lp_values = zeros(T, horizon, n, n)

    for shock in 1:n
        for (h_idx, h) in enumerate(1:horizon)
            for resp in 1:n
                lp_values[h_idx, resp, shock] = lp_results[shock].values[h + 1, resp]
            end
        end
    end

    (var_irf=var_values, lp_irf=lp_values, difference=var_values - lp_values)
end
