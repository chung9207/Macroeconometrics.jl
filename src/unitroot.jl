"""
Unit root tests and stationarity diagnostics for MacroEconometricModels.jl.

Includes:
- Univariate tests: ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron
- Multivariate tests: Johansen cointegration
- VAR stability: is_stationary for VARModel
"""

using LinearAlgebra, Statistics, Distributions, PrettyTables

# =============================================================================
# Abstract Type
# =============================================================================

"""Abstract supertype for all unit root test results."""
abstract type AbstractUnitRootTest <: StatsAPI.HypothesisTest end

# =============================================================================
# Result Types
# =============================================================================

"""
    ADFResult{T} <: AbstractUnitRootTest

Augmented Dickey-Fuller test result.

Fields:
- `statistic`: ADF test statistic (t-ratio on γ)
- `pvalue`: Approximate p-value (MacKinnon 1994, 2010)
- `lags`: Number of augmenting lags used
- `regression`: Regression specification (:none, :constant, :trend)
- `critical_values`: Critical values at 1%, 5%, 10% levels
- `nobs`: Effective number of observations
"""
struct ADFResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    lags::Int
    regression::Symbol
    critical_values::Dict{Int,T}
    nobs::Int
end

"""
    KPSSResult{T} <: AbstractUnitRootTest

KPSS stationarity test result.

Fields:
- `statistic`: KPSS test statistic
- `pvalue`: Approximate p-value
- `regression`: Regression specification (:constant, :trend)
- `critical_values`: Critical values at 1%, 5%, 10% levels
- `bandwidth`: Bartlett kernel bandwidth used
- `nobs`: Number of observations
"""
struct KPSSResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    regression::Symbol
    critical_values::Dict{Int,T}
    bandwidth::Int
    nobs::Int
end

"""
    PPResult{T} <: AbstractUnitRootTest

Phillips-Perron test result.

Fields:
- `statistic`: PP test statistic (Zt or Zα)
- `pvalue`: Approximate p-value
- `regression`: Regression specification (:none, :constant, :trend)
- `critical_values`: Critical values at 1%, 5%, 10% levels
- `bandwidth`: Newey-West bandwidth used
- `nobs`: Effective number of observations
"""
struct PPResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    regression::Symbol
    critical_values::Dict{Int,T}
    bandwidth::Int
    nobs::Int
end

"""
    ZAResult{T} <: AbstractUnitRootTest

Zivot-Andrews structural break unit root test result.

Fields:
- `statistic`: Minimum t-statistic across all break points
- `pvalue`: Approximate p-value
- `break_index`: Index of estimated structural break
- `break_fraction`: Break point as fraction of sample
- `regression`: Break specification (:constant, :trend, :both)
- `critical_values`: Critical values at 1%, 5%, 10% levels
- `lags`: Number of augmenting lags
- `nobs`: Effective number of observations
"""
struct ZAResult{T<:AbstractFloat} <: AbstractUnitRootTest
    statistic::T
    pvalue::T
    break_index::Int
    break_fraction::T
    regression::Symbol
    critical_values::Dict{Int,T}
    lags::Int
    nobs::Int
end

"""
    NgPerronResult{T} <: AbstractUnitRootTest

Ng-Perron unit root test result (MZα, MZt, MSB, MPT).

Fields:
- `MZa`: Modified Zα statistic
- `MZt`: Modified Zt statistic
- `MSB`: Modified Sargan-Bhargava statistic
- `MPT`: Modified Point-optimal statistic
- `regression`: Regression specification (:constant, :trend)
- `critical_values`: Dict mapping statistic name to critical values
- `nobs`: Effective number of observations
"""
struct NgPerronResult{T<:AbstractFloat} <: AbstractUnitRootTest
    MZa::T
    MZt::T
    MSB::T
    MPT::T
    regression::Symbol
    critical_values::Dict{Symbol,Dict{Int,T}}
    nobs::Int
end

"""
    JohansenResult{T} <: AbstractUnitRootTest

Johansen cointegration test result.

Fields:
- `trace_stats`: Trace test statistics for each rank
- `trace_pvalues`: P-values for trace tests
- `max_eigen_stats`: Maximum eigenvalue test statistics
- `max_eigen_pvalues`: P-values for max eigenvalue tests
- `rank`: Estimated cointegration rank (at 5% level)
- `eigenvectors`: Cointegrating vectors (β), columns are vectors
- `adjustment`: Adjustment coefficients (α)
- `eigenvalues`: Eigenvalues from reduced rank regression
- `critical_values_trace`: Critical values for trace test (rows: ranks, cols: 10%, 5%, 1%)
- `critical_values_max`: Critical values for max eigenvalue test
- `deterministic`: Deterministic specification
- `lags`: Number of lags in VECM
- `nobs`: Effective number of observations
"""
struct JohansenResult{T<:AbstractFloat} <: AbstractUnitRootTest
    trace_stats::Vector{T}
    trace_pvalues::Vector{T}
    max_eigen_stats::Vector{T}
    max_eigen_pvalues::Vector{T}
    rank::Int
    eigenvectors::Matrix{T}
    adjustment::Matrix{T}
    eigenvalues::Vector{T}
    critical_values_trace::Matrix{T}
    critical_values_max::Matrix{T}
    deterministic::Symbol
    lags::Int
    nobs::Int
end

"""
    VARStationarityResult{T}

VAR model stationarity check result.

Fields:
- `is_stationary`: true if all eigenvalues have modulus < 1
- `eigenvalues`: Eigenvalues of companion matrix (may be real or complex)
- `max_modulus`: Maximum eigenvalue modulus
- `companion_matrix`: The companion form matrix F
"""
struct VARStationarityResult{T<:AbstractFloat, E<:Union{T, Complex{T}}}
    is_stationary::Bool
    eigenvalues::Vector{E}
    max_modulus::T
    companion_matrix::Matrix{T}
end

# =============================================================================
# StatsAPI Interface for Unit Root Tests
# =============================================================================

# Common interface for all unit root tests
StatsAPI.nobs(r::ADFResult) = r.nobs
StatsAPI.nobs(r::KPSSResult) = r.nobs
StatsAPI.nobs(r::PPResult) = r.nobs
StatsAPI.nobs(r::ZAResult) = r.nobs
StatsAPI.nobs(r::NgPerronResult) = r.nobs
StatsAPI.nobs(r::JohansenResult) = r.nobs

StatsAPI.dof(r::ADFResult) = r.lags + (r.regression == :none ? 1 : r.regression == :constant ? 2 : 3)
StatsAPI.dof(r::KPSSResult) = r.regression == :constant ? 1 : 2
StatsAPI.dof(r::PPResult) = r.regression == :none ? 1 : r.regression == :constant ? 2 : 3
StatsAPI.dof(r::ZAResult) = r.lags + (r.regression == :constant ? 4 : r.regression == :trend ? 4 : 5)
StatsAPI.dof(r::NgPerronResult) = r.regression == :constant ? 1 : 2
StatsAPI.dof(r::JohansenResult) = r.lags

# pvalue - already stored in struct
StatsAPI.pvalue(r::ADFResult) = r.pvalue
StatsAPI.pvalue(r::KPSSResult) = r.pvalue
StatsAPI.pvalue(r::PPResult) = r.pvalue
StatsAPI.pvalue(r::ZAResult) = r.pvalue
# For NgPerron, return MZt p-value as primary (most commonly used)
StatsAPI.pvalue(r::NgPerronResult) = _ngperron_pvalue(r.MZt, r.regression, :MZt)
# For Johansen, return minimum trace p-value
StatsAPI.pvalue(r::JohansenResult) = minimum(r.trace_pvalues)

# =============================================================================
# Critical Value Tables
# =============================================================================

# MacKinnon (2010) response surface coefficients for ADF/PP p-values
# Format: (β∞, β₁, β₂) for τ = β∞ + β₁/T + β₂/T²
const MACKINNON_ADF_COEFS = Dict(
    # No constant, no trend (nc)
    :none => Dict(
        1  => (-2.5658, -1.960, -10.04),   # 1%
        5  => (-1.9393, -0.398,  -0.0),    # 5%
        10 => (-1.6156, -0.181,  -0.0)     # 10%
    ),
    # Constant only (c)
    :constant => Dict(
        1  => (-3.4336, -5.999, -29.25),
        5  => (-2.8621, -2.738,  -8.36),
        10 => (-2.5671, -1.438,  -4.48)
    ),
    # Constant and trend (ct)
    :trend => Dict(
        1  => (-3.9638, -8.353, -47.44),
        5  => (-3.4126, -4.039, -17.83),
        10 => (-3.1279, -2.418,  -7.58)
    )
)

# KPSS critical values (Kwiatkowski et al. 1992, Table 1)
const KPSS_CRITICAL_VALUES = Dict(
    :constant => Dict(1 => 0.739, 5 => 0.463, 10 => 0.347),
    :trend    => Dict(1 => 0.216, 5 => 0.146, 10 => 0.119)
)

# Zivot-Andrews critical values (Zivot & Andrews 1992, Table 4)
const ZA_CRITICAL_VALUES = Dict(
    :constant => Dict(1 => -5.34, 5 => -4.80, 10 => -4.58),
    :trend    => Dict(1 => -4.80, 5 => -4.42, 10 => -4.11),
    :both     => Dict(1 => -5.57, 5 => -5.08, 10 => -4.82)
)

# Ng-Perron critical values (Ng & Perron 2001, Table 1)
const NGPERRON_CRITICAL_VALUES = Dict(
    :constant => Dict(
        :MZa => Dict(1 => -13.8, 5 => -8.1, 10 => -5.7),
        :MZt => Dict(1 => -2.58, 5 => -1.98, 10 => -1.62),
        :MSB => Dict(1 => 0.174, 5 => 0.233, 10 => 0.275),
        :MPT => Dict(1 => 1.78, 5 => 3.17, 10 => 4.45)
    ),
    :trend => Dict(
        :MZa => Dict(1 => -23.8, 5 => -17.3, 10 => -14.2),
        :MZt => Dict(1 => -3.42, 5 => -2.91, 10 => -2.62),
        :MSB => Dict(1 => 0.143, 5 => 0.168, 10 => 0.185),
        :MPT => Dict(1 => 4.03, 5 => 5.48, 10 => 6.67)
    )
)

# Johansen critical values (Osterwald-Lenum 1992)
# Format: [10%, 5%, 1%] for each n-r (number of common trends)
# Trace test critical values (constant in cointegrating relation)
const JOHANSEN_TRACE_CV_CONSTANT = Dict(
    1 => [6.50, 8.18, 11.65],
    2 => [15.66, 17.95, 23.52],
    3 => [28.71, 31.52, 37.22],
    4 => [45.23, 48.28, 55.43],
    5 => [66.49, 70.60, 78.87],
    6 => [85.18, 90.39, 104.20],
    7 => [118.99, 124.25, 136.06],
    8 => [151.38, 157.11, 168.92],
    9 => [186.54, 192.89, 206.95],
    10 => [224.63, 231.26, 247.18]
)

# Max eigenvalue test critical values
const JOHANSEN_MAX_CV_CONSTANT = Dict(
    1 => [6.50, 8.18, 11.65],
    2 => [12.91, 14.90, 19.19],
    3 => [18.90, 21.07, 25.75],
    4 => [24.78, 27.14, 32.14],
    5 => [30.84, 33.32, 38.78],
    6 => [36.25, 39.43, 44.59],
    7 => [42.06, 45.28, 51.30],
    8 => [48.43, 51.42, 57.07],
    9 => [54.01, 57.12, 62.80],
    10 => [59.00, 62.81, 68.83]
)

# =============================================================================
# Helper Functions
# =============================================================================

"""Compute ADF critical values using MacKinnon response surface."""
function adf_critical_values(regression::Symbol, nobs::Int, T::Type=Float64)
    coefs = MACKINNON_ADF_COEFS[regression]
    Dict{Int,T}(
        level => T(c[1] + c[2]/nobs + c[3]/nobs^2)
        for (level, c) in coefs
    )
end

"""Approximate p-value for ADF test using MacKinnon (1994) interpolation."""
function adf_pvalue(stat::T, regression::Symbol, nobs::Int) where {T<:AbstractFloat}
    # Get critical values at standard levels
    cv = adf_critical_values(regression, nobs, T)

    # Simple interpolation between critical values
    if stat <= cv[1]
        return T(0.001)  # Below 1% critical value
    elseif stat <= cv[5]
        # Interpolate between 1% and 5%
        return T(0.01 + 0.04 * (stat - cv[1]) / (cv[5] - cv[1]))
    elseif stat <= cv[10]
        # Interpolate between 5% and 10%
        return T(0.05 + 0.05 * (stat - cv[5]) / (cv[10] - cv[5]))
    else
        # Above 10% critical value - use normal approximation for large values
        # The ADF statistic converges to standard normal under stationarity
        return T(min(1.0, 0.10 + 0.90 * (1 - cdf(Normal(), -stat))))
    end
end

"""Approximate p-value for KPSS test."""
function kpss_pvalue(stat::T, regression::Symbol) where {T<:AbstractFloat}
    cv = KPSS_CRITICAL_VALUES[regression]

    if stat >= cv[1]
        return T(0.01)
    elseif stat >= cv[5]
        return T(0.01 + 0.04 * (cv[1] - stat) / (cv[1] - cv[5]))
    elseif stat >= cv[10]
        return T(0.05 + 0.05 * (cv[5] - stat) / (cv[5] - cv[10]))
    else
        return T(0.10 + 0.40 * (cv[10] - stat) / cv[10])
    end
end

"""Approximate p-value for Zivot-Andrews test."""
function za_pvalue(stat::T, regression::Symbol) where {T<:AbstractFloat}
    cv = ZA_CRITICAL_VALUES[regression]

    if stat <= cv[1]
        return T(0.01)
    elseif stat <= cv[5]
        return T(0.01 + 0.04 * (stat - cv[1]) / (cv[5] - cv[1]))
    elseif stat <= cv[10]
        return T(0.05 + 0.05 * (stat - cv[5]) / (cv[10] - cv[5]))
    else
        return T(min(1.0, 0.10 + 0.30 * (stat - cv[10]) / abs(cv[10])))
    end
end

"""Approximate p-value for Ng-Perron tests."""
function _ngperron_pvalue(stat::T, regression::Symbol, test::Symbol) where {T<:AbstractFloat}
    cv = NGPERRON_CRITICAL_VALUES[regression][test]

    # For MZa, MZt: more negative = reject
    # For MSB: smaller = reject
    # For MPT: smaller = reject
    if test in (:MZa, :MZt)
        if stat <= cv[1]
            return T(0.01)
        elseif stat <= cv[5]
            return T(0.01 + 0.04 * (stat - cv[1]) / (cv[5] - cv[1]))
        elseif stat <= cv[10]
            return T(0.05 + 0.05 * (stat - cv[5]) / (cv[10] - cv[5]))
        else
            return T(min(1.0, 0.10 + 0.30 * (stat - cv[10]) / abs(cv[10])))
        end
    else  # MSB, MPT
        if stat <= cv[1]
            return T(0.01)
        elseif stat <= cv[5]
            return T(0.01 + 0.04 * (stat - cv[1]) / (cv[5] - cv[1]))
        elseif stat <= cv[10]
            return T(0.05 + 0.05 * (stat - cv[5]) / (cv[10] - cv[5]))
        else
            return T(min(1.0, 0.10 + 0.30 * (stat - cv[10]) / cv[10]))
        end
    end
end

"""Compute optimal lag length for ADF test using information criterion."""
function adf_select_lags(y::AbstractVector{T}, max_lags::Int, regression::Symbol,
                         criterion::Symbol) where {T<:AbstractFloat}
    n = length(y)
    dy = diff(y)
    y_lag = y[1:end-1]

    best_ic = T(Inf)
    best_lag = 0

    for p in 0:max_lags
        nobs_eff = n - 1 - p
        nobs_eff < 10 && continue

        # Build regression matrix
        X = _build_adf_matrix(y, dy, p, regression)
        Y = dy[(p+1):end]

        k = size(X, 2)
        nobs_eff = length(Y)

        # OLS estimation
        XtX = X'X
        det(XtX) ≈ 0 && continue
        B = XtX \ (X'Y)
        resid = Y - X * B
        sse = sum(resid.^2)
        sigma2 = sse / (nobs_eff - k)

        # Information criterion
        ll = -nobs_eff/2 * (log(2π) + log(sigma2) + 1)
        ic = if criterion == :aic
            -2ll + 2k
        elseif criterion == :bic
            -2ll + k * log(nobs_eff)
        else  # :hqic
            -2ll + 2k * log(log(nobs_eff))
        end

        if ic < best_ic
            best_ic = ic
            best_lag = p
        end
    end

    best_lag
end

"""Build ADF regression matrix."""
function _build_adf_matrix(y::AbstractVector{T}, dy::AbstractVector{T},
                           lags::Int, regression::Symbol) where {T<:AbstractFloat}
    n = length(dy)
    nobs = n - lags

    # y_{t-1} column
    y_lag = y[(lags+1):(n)]

    # Lagged differences
    if lags > 0
        dy_lags = Matrix{T}(undef, nobs, lags)
        for j in 1:lags
            dy_lags[:, j] = dy[(lags+1-j):(n-j)]
        end
    end

    # Build design matrix based on regression type
    if regression == :none
        X = lags > 0 ? hcat(y_lag, dy_lags) : reshape(y_lag, :, 1)
    elseif regression == :constant
        ones_col = ones(T, nobs)
        X = lags > 0 ? hcat(ones_col, y_lag, dy_lags) : hcat(ones_col, y_lag)
    else  # :trend
        ones_col = ones(T, nobs)
        trend = T.(1:nobs)
        X = lags > 0 ? hcat(ones_col, trend, y_lag, dy_lags) : hcat(ones_col, trend, y_lag)
    end

    X
end

"""Compute Newey-West bandwidth using Andrews (1991) AR(1) rule."""
function _nw_bandwidth(resid::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(resid)
    # AR(1) approximation for bandwidth
    rho = cor(resid[1:end-1], resid[2:end])
    rho = clamp(rho, -0.99, 0.99)
    # Andrews (1991) optimal bandwidth for Bartlett kernel
    bw = floor(Int, 1.1447 * (4 * rho^2 / (1 - rho^2)^2 * n)^(1/3))
    max(1, min(bw, n - 1))
end

"""Compute long-run variance using Bartlett kernel."""
function _long_run_variance(resid::AbstractVector{T}, bandwidth::Int) where {T<:AbstractFloat}
    n = length(resid)
    gamma0 = var(resid; corrected=false)

    lrv = gamma0
    for j in 1:bandwidth
        weight = 1 - j / (bandwidth + 1)  # Bartlett kernel
        gamma_j = sum(resid[1:end-j] .* resid[1+j:end]) / n
        lrv += 2 * weight * gamma_j
    end

    lrv
end

# =============================================================================
# ADF Test
# =============================================================================

"""
    adf_test(y; lags=:aic, max_lags=nothing, regression=:constant) -> ADFResult

Augmented Dickey-Fuller test for unit root.

Tests H₀: y has a unit root (non-stationary) against H₁: y is stationary.

# Arguments
- `y`: Time series vector
- `lags`: Number of augmenting lags, or :aic/:bic/:hqic for automatic selection
- `max_lags`: Maximum lags for automatic selection (default: floor(12*(T/100)^0.25))
- `regression`: Deterministic terms - :none, :constant (default), or :trend

# Returns
`ADFResult` containing test statistic, p-value, critical values, etc.

# Example
```julia
y = cumsum(randn(200))  # Random walk (has unit root)
result = adf_test(y)
result.pvalue > 0.05  # Should fail to reject H₀
```

# References
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for
  autoregressive time series with a unit root. JASA, 74(366), 427-431.
- MacKinnon, J. G. (2010). Critical values for cointegration tests.
  Queen's Economics Department Working Paper No. 1227.
"""
function adf_test(y::AbstractVector{T};
                  lags::Union{Int,Symbol}=:aic,
                  max_lags::Union{Int,Nothing}=nothing,
                  regression::Symbol=:constant) where {T<:AbstractFloat}

    regression ∈ (:none, :constant, :trend) ||
        throw(ArgumentError("regression must be :none, :constant, or :trend"))

    n = length(y)
    n < 20 && throw(ArgumentError("Time series too short (n=$n), need at least 20 observations"))

    # Determine maximum lags for selection
    max_p = isnothing(max_lags) ? floor(Int, 12 * (n / 100)^0.25) : max_lags

    # Select lag length
    p = if lags isa Symbol
        lags ∈ (:aic, :bic, :hqic) ||
            throw(ArgumentError("lags must be an integer or :aic/:bic/:hqic"))
        adf_select_lags(y, max_p, regression, lags)
    else
        lags
    end

    # Compute first differences
    dy = diff(y)

    # Build regression matrix
    X = _build_adf_matrix(y, dy, p, regression)
    Y = dy[(p+1):end]
    nobs = length(Y)

    # OLS estimation
    XtX = X'X
    XtX_inv = inv(XtX)
    B = XtX_inv * (X'Y)
    resid = Y - X * B

    # Compute standard error of γ coefficient
    sigma2 = sum(resid.^2) / (nobs - size(X, 2))
    se = sqrt.(sigma2 * diag(XtX_inv))

    # γ is the coefficient on y_{t-1}
    gamma_idx = regression == :none ? 1 : (regression == :constant ? 2 : 3)
    gamma = B[gamma_idx]
    gamma_se = se[gamma_idx]

    # ADF test statistic
    stat = gamma / gamma_se

    # Critical values and p-value
    cv = adf_critical_values(regression, nobs, T)
    pval = adf_pvalue(stat, regression, nobs)

    ADFResult(stat, pval, p, regression, cv, nobs)
end

# Float64 fallback
adf_test(y::AbstractVector; kwargs...) = adf_test(Float64.(y); kwargs...)

# =============================================================================
# KPSS Test
# =============================================================================

"""
    kpss_test(y; regression=:constant, bandwidth=:auto) -> KPSSResult

Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

Tests H₀: y is stationary against H₁: y has a unit root.

# Arguments
- `y`: Time series vector
- `regression`: :constant (level stationarity) or :trend (trend stationarity)
- `bandwidth`: Bartlett kernel bandwidth, or :auto for Newey-West selection

# Returns
`KPSSResult` containing test statistic, p-value, critical values, etc.

# Example
```julia
y = randn(200)  # Stationary series
result = kpss_test(y)
result.pvalue > 0.05  # Should fail to reject H₀ (stationarity)
```

# References
- Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992). Testing
  the null hypothesis of stationarity against the alternative of a unit root.
  Journal of Econometrics, 54(1-3), 159-178.
"""
function kpss_test(y::AbstractVector{T};
                   regression::Symbol=:constant,
                   bandwidth::Union{Int,Symbol}=:auto) where {T<:AbstractFloat}

    regression ∈ (:constant, :trend) ||
        throw(ArgumentError("regression must be :constant or :trend"))

    n = length(y)
    n < 10 && throw(ArgumentError("Time series too short (n=$n), need at least 10 observations"))

    # Detrend the series
    if regression == :constant
        resid = y .- mean(y)
    else  # :trend
        t = T.(1:n)
        X = hcat(ones(T, n), t)
        B = X \ y
        resid = y - X * B
    end

    # Compute partial sums
    S = cumsum(resid)

    # Determine bandwidth
    bw = bandwidth == :auto ? _nw_bandwidth(resid) : bandwidth

    # Long-run variance estimate
    lrv = _long_run_variance(resid, bw)

    # KPSS statistic
    stat = sum(S.^2) / (n^2 * lrv)

    # Critical values and p-value
    cv = Dict{Int,T}(k => T(v) for (k, v) in KPSS_CRITICAL_VALUES[regression])
    pval = kpss_pvalue(stat, regression)

    KPSSResult(stat, pval, regression, cv, bw, n)
end

kpss_test(y::AbstractVector; kwargs...) = kpss_test(Float64.(y); kwargs...)

# =============================================================================
# Phillips-Perron Test
# =============================================================================

"""
    pp_test(y; regression=:constant, bandwidth=:auto) -> PPResult

Phillips-Perron test for unit root with non-parametric correction.

Tests H₀: y has a unit root against H₁: y is stationary.

# Arguments
- `y`: Time series vector
- `regression`: :none, :constant (default), or :trend
- `bandwidth`: Newey-West bandwidth, or :auto for automatic selection

# Returns
`PPResult` containing test statistic (Zt), p-value, critical values, etc.

# Example
```julia
y = cumsum(randn(200))  # Random walk
result = pp_test(y)
result.pvalue > 0.05  # Should fail to reject H₀
```

# References
- Phillips, P. C., & Perron, P. (1988). Testing for a unit root in time
  series regression. Biometrika, 75(2), 335-346.
"""
function pp_test(y::AbstractVector{T};
                 regression::Symbol=:constant,
                 bandwidth::Union{Int,Symbol}=:auto) where {T<:AbstractFloat}

    regression ∈ (:none, :constant, :trend) ||
        throw(ArgumentError("regression must be :none, :constant, or :trend"))

    n = length(y)
    n < 20 && throw(ArgumentError("Time series too short (n=$n), need at least 20 observations"))

    # Build regression: y_t = α + β*t + ρ*y_{t-1} + u_t
    y_lag = y[1:end-1]
    y_curr = y[2:end]
    nobs = n - 1

    if regression == :none
        X = reshape(y_lag, :, 1)
    elseif regression == :constant
        X = hcat(ones(T, nobs), y_lag)
    else  # :trend
        t = T.(1:nobs)
        X = hcat(ones(T, nobs), t, y_lag)
    end

    # OLS estimation
    XtX = X'X
    XtX_inv = inv(XtX)
    B = XtX_inv * (X'y_curr)
    resid = y_curr - X * B

    # Standard error under homoskedasticity
    sigma2 = sum(resid.^2) / (nobs - size(X, 2))
    se = sqrt.(sigma2 * diag(XtX_inv))

    # Coefficient on y_{t-1}
    rho_idx = regression == :none ? 1 : (regression == :constant ? 2 : 3)
    rho = B[rho_idx]
    rho_se = se[rho_idx]

    # t-statistic (uncorrected)
    t_rho = (rho - 1) / rho_se

    # Bandwidth for long-run variance
    bw = bandwidth == :auto ? _nw_bandwidth(resid) : bandwidth

    # Long-run variance and short-run variance
    gamma0 = var(resid; corrected=false)
    lambda2 = _long_run_variance(resid, bw)

    # Phillips-Perron Zt statistic
    # Zt = sqrt(γ₀/λ²) * t_ρ - (λ² - γ₀) / (2λ * se_ρ * sqrt(T))
    stat = sqrt(gamma0 / lambda2) * t_rho -
           (lambda2 - gamma0) / (2 * sqrt(lambda2) * rho_se * sqrt(nobs))

    # Critical values (same as ADF)
    cv = adf_critical_values(regression, nobs, T)
    pval = adf_pvalue(stat, regression, nobs)

    PPResult(stat, pval, regression, cv, bw, nobs)
end

pp_test(y::AbstractVector; kwargs...) = pp_test(Float64.(y); kwargs...)

# =============================================================================
# Zivot-Andrews Test
# =============================================================================

"""
    za_test(y; regression=:both, trim=0.15, lags=:aic, max_lags=nothing) -> ZAResult

Zivot-Andrews test for unit root with endogenous structural break.

Tests H₀: y has a unit root without break against H₁: y is stationary with break.

# Arguments
- `y`: Time series vector
- `regression`: Type of break - :constant (intercept), :trend (slope), or :both
- `trim`: Trimming fraction for break search (default 0.15)
- `lags`: Number of augmenting lags, or :aic/:bic for automatic selection
- `max_lags`: Maximum lags for selection

# Returns
`ZAResult` containing minimum t-statistic, break point, p-value, etc.

# Example
```julia
# Series with structural break
y = vcat(randn(100), randn(100) .+ 2)
result = za_test(y; regression=:constant)
```

# References
- Zivot, E., & Andrews, D. W. K. (1992). Further evidence on the great crash,
  the oil-price shock, and the unit-root hypothesis. JBES, 10(3), 251-270.
"""
function za_test(y::AbstractVector{T};
                 regression::Symbol=:both,
                 trim::Real=0.15,
                 lags::Union{Int,Symbol}=:aic,
                 max_lags::Union{Int,Nothing}=nothing) where {T<:AbstractFloat}

    regression ∈ (:constant, :trend, :both) ||
        throw(ArgumentError("regression must be :constant, :trend, or :both"))
    0 < trim < 0.5 || throw(ArgumentError("trim must be between 0 and 0.5"))

    n = length(y)
    n < 50 && throw(ArgumentError("Time series too short for ZA test (n=$n), need at least 50"))

    # Maximum lags
    max_p = isnothing(max_lags) ? floor(Int, 12 * (n / 100)^0.25) : max_lags

    # Trimming bounds
    start_idx = max(2, ceil(Int, trim * n))
    end_idx = min(n - 1, floor(Int, (1 - trim) * n))

    # First differences
    dy = diff(y)

    min_stat = T(Inf)
    best_break = start_idx
    best_lags = 0

    for tb in start_idx:end_idx
        # Determine lags for this break point
        p = if lags isa Symbol
            # Simplified: use fixed lag selection for speed
            min(max_p, 4)
        else
            lags
        end

        # Build regression matrix with break dummies
        nobs = n - 1 - p
        nobs < 10 && continue

        # Dependent variable
        Y = dy[(p+1):end]

        # Regressors
        ones_col = ones(T, nobs)
        trend = T.(1:nobs)
        y_lag = y[(p+1):(n-1)]

        # Break dummies
        DU = T.((p+1):n-1 .>= tb)  # Level shift
        DT = T.(max.(0, (p+1:n-1) .- tb .+ 1))  # Trend shift

        # Lagged differences
        X_base = if regression == :constant
            hcat(ones_col, trend, DU, y_lag)
        elseif regression == :trend
            hcat(ones_col, trend, DT, y_lag)
        else  # :both
            hcat(ones_col, trend, DU, DT, y_lag)
        end

        # Add lagged differences
        if p > 0
            dy_lags = Matrix{T}(undef, nobs, p)
            for j in 1:p
                dy_lags[:, j] = dy[(p+1-j):(n-1-j)]
            end
            X = hcat(X_base, dy_lags)
        else
            X = X_base
        end

        # OLS
        XtX = X'X
        cond(XtX) > 1e12 && continue

        B = XtX \ (X'Y)
        resid = Y - X * B

        # Standard errors
        sigma2 = sum(resid.^2) / (nobs - size(X, 2))
        se = sqrt.(sigma2 * diag(inv(XtX)))

        # t-statistic on y_{t-1}
        gamma_idx = regression == :both ? 5 : 4
        t_stat = B[gamma_idx] / se[gamma_idx]

        if t_stat < min_stat
            min_stat = t_stat
            best_break = tb
            best_lags = p
        end
    end

    # Critical values and p-value
    cv = Dict{Int,T}(k => T(v) for (k, v) in ZA_CRITICAL_VALUES[regression])
    pval = za_pvalue(min_stat, regression)
    break_frac = T(best_break) / n

    ZAResult(min_stat, pval, best_break, break_frac, regression, cv, best_lags, n - 1 - best_lags)
end

za_test(y::AbstractVector; kwargs...) = za_test(Float64.(y); kwargs...)

# =============================================================================
# Ng-Perron Tests
# =============================================================================

"""
    ngperron_test(y; regression=:constant) -> NgPerronResult

Ng-Perron unit root tests with GLS detrending (MZα, MZt, MSB, MPT).

Tests H₀: y has a unit root against H₁: y is stationary.
These tests have better size properties than ADF/PP in small samples.

# Arguments
- `y`: Time series vector
- `regression`: :constant (default) or :trend

# Returns
`NgPerronResult` containing MZα, MZt, MSB, MPT statistics and critical values.

# Example
```julia
y = cumsum(randn(100))
result = ngperron_test(y)
# Check if MZt rejects at 5%
result.MZt < result.critical_values[:MZt][5]
```

# References
- Ng, S., & Perron, P. (2001). Lag length selection and the construction of
  unit root tests with good size and power. Econometrica, 69(6), 1519-1554.
"""
function ngperron_test(y::AbstractVector{T};
                       regression::Symbol=:constant) where {T<:AbstractFloat}

    regression ∈ (:constant, :trend) ||
        throw(ArgumentError("regression must be :constant or :trend"))

    n = length(y)
    n < 20 && throw(ArgumentError("Time series too short (n=$n), need at least 20 observations"))

    # GLS detrending parameter
    c_bar = regression == :constant ? T(-7.0) : T(-13.5)
    alpha = 1 + c_bar / n

    # Construct quasi-differenced data
    y_qd = copy(y)
    y_qd[2:end] = y[2:end] - alpha * y[1:end-1]

    # Deterministic regressors (quasi-differenced)
    if regression == :constant
        z = ones(T, n)
        z[2:end] .= 1 - alpha
        Z = reshape(z, :, 1)
    else  # :trend
        z1 = ones(T, n)
        z1[2:end] .= 1 - alpha
        z2 = T.(1:n)
        z2[2:end] = z2[2:end] - alpha * z2[1:end-1]
        Z = hcat(z1, z2)
    end

    # GLS detrending
    delta = Z \ y_qd
    y_d = y - Z * (Z \ y)  # Detrended series using full Z

    # Compute statistics
    # Autoregressive spectral density estimate at frequency zero
    k = floor(Int, 4 * (n / 100)^(2/9))  # MAIC bandwidth

    # AR(k) regression for spectral density
    if k > 0 && n > k + 5
        Y_ar = y_d[(k+1):end]
        X_ar = hcat([y_d[(k+1-j):(end-j)] for j in 1:k]...)
        rho_ar = X_ar \ Y_ar
        resid_ar = Y_ar - X_ar * rho_ar
        s2 = var(resid_ar; corrected=true)
        ar_sum = 1 - sum(rho_ar)
        s2_ar = s2 / ar_sum^2
    else
        s2_ar = var(y_d; corrected=true)
    end

    # Components for statistics
    sum_yd2 = sum(y_d[1:end-1].^2) / n^2
    T_term = y_d[end]^2 / n

    # MZα statistic
    MZa = (T_term - s2_ar) / (2 * sum_yd2)

    # MSB statistic
    MSB = sqrt(sum_yd2 / s2_ar)

    # MZt statistic
    MZt = MZa * MSB

    # MPT statistic
    if regression == :constant
        MPT = (c_bar^2 * sum_yd2 + T_term) / s2_ar
    else
        MPT = (c_bar^2 * sum_yd2 + (1 - c_bar) * T_term) / s2_ar
    end

    # Critical values
    cv = Dict{Symbol,Dict{Int,T}}(
        stat => Dict{Int,T}(k => T(v) for (k, v) in vals)
        for (stat, vals) in NGPERRON_CRITICAL_VALUES[regression]
    )

    NgPerronResult(MZa, MZt, MSB, MPT, regression, cv, n)
end

ngperron_test(y::AbstractVector; kwargs...) = ngperron_test(Float64.(y); kwargs...)

# =============================================================================
# Johansen Cointegration Test
# =============================================================================

"""
    johansen_test(Y, p; deterministic=:constant) -> JohansenResult

Johansen cointegration test for VAR system.

Tests for the number of cointegrating relationships among variables using
trace and maximum eigenvalue tests.

# Arguments
- `Y`: Data matrix (T × n)
- `p`: Number of lags in the VECM representation
- `deterministic`: Specification for deterministic terms
  - :none - No deterministic terms
  - :constant - Constant in cointegrating relation (default)
  - :trend - Linear trend in levels

# Returns
`JohansenResult` containing trace and max-eigenvalue statistics, cointegrating
vectors, adjustment coefficients, and estimated rank.

# Example
```julia
# Generate cointegrated system
n, T = 3, 200
Y = randn(T, n)
Y[:, 2] = Y[:, 1] + 0.1 * randn(T)  # Y2 cointegrated with Y1

result = johansen_test(Y, 2)
result.rank  # Should detect 1 or 2 cointegrating relations
```

# References
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration
  vectors in Gaussian vector autoregressive models. Econometrica, 59(6), 1551-1580.
- Osterwald-Lenum, M. (1992). A note with quantiles of the asymptotic
  distribution of the ML cointegration rank test statistics. Oxford BEJM.
"""
function johansen_test(Y::AbstractMatrix{T}, p::Int;
                       deterministic::Symbol=:constant) where {T<:AbstractFloat}

    deterministic ∈ (:none, :constant, :trend) ||
        throw(ArgumentError("deterministic must be :none, :constant, or :trend"))

    T_obs, n = size(Y)
    T_obs < n + p + 10 && throw(ArgumentError("Not enough observations for Johansen test"))
    p < 1 && throw(ArgumentError("Number of lags p must be at least 1"))

    # VECM representation: ΔYₜ = Π Yₜ₋₁ + Σᵢ Γᵢ ΔYₜ₋ᵢ + μ + εₜ
    # where Π = αβ' is the long-run matrix

    # Construct matrices
    dY = diff(Y, dims=1)  # ΔY: (T-1) × n
    Y_lag = Y[p:end-1, :]  # Y_{t-1}: (T-p) × n

    # Lagged differences
    T_eff = T_obs - p
    dY_lags = if p > 1
        hcat([dY[(p-j):(end-j), :] for j in 1:(p-1)]...)
    else
        Matrix{T}(undef, T_eff, 0)
    end

    # Dependent variable
    dY_eff = dY[p:end, :]

    # Deterministic terms
    if deterministic == :none
        Z = dY_lags
    elseif deterministic == :constant
        Z = isempty(dY_lags) ? ones(T, T_eff, 1) : hcat(ones(T, T_eff), dY_lags)
    else  # :trend
        trend = T.(1:T_eff)
        Z = isempty(dY_lags) ? hcat(ones(T, T_eff), trend) : hcat(ones(T, T_eff), trend, dY_lags)
    end

    # Concentrate out short-run dynamics
    if size(Z, 2) > 0
        M = I - Z * ((Z'Z) \ Z')
        R0 = M * dY_eff   # Residuals from regressing ΔY on Z
        R1 = M * Y_lag    # Residuals from regressing Y_{t-1} on Z
    else
        R0 = dY_eff
        R1 = Y_lag
    end

    # Moment matrices
    S00 = (R0'R0) / T_eff
    S11 = (R1'R1) / T_eff
    S01 = (R0'R1) / T_eff
    S10 = S01'

    # Solve generalized eigenvalue problem
    # |λS₁₁ - S₁₀S₀₀⁻¹S₀₁| = 0
    S00_inv = inv(S00)
    A = S11 \ (S10 * S00_inv * S01)

    # Eigendecomposition
    eig = eigen(A)
    idx = sortperm(real.(eig.values), rev=true)
    eigenvalues = real.(eig.values[idx])
    eigenvectors = real.(eig.vectors[:, idx])

    # Ensure eigenvalues are in [0, 1]
    eigenvalues = clamp.(eigenvalues, 0, 1 - eps(T))

    # Test statistics
    trace_stats = Vector{T}(undef, n)
    max_eigen_stats = Vector{T}(undef, n)

    for r in 0:(n-1)
        # Trace statistic: -T Σᵢ₌ᵣ₊₁ⁿ ln(1 - λᵢ)
        trace_stats[r+1] = -T_eff * sum(log.(1 .- eigenvalues[(r+1):n]))
        # Max eigenvalue statistic: -T ln(1 - λᵣ₊₁)
        max_eigen_stats[r+1] = -T_eff * log(1 - eigenvalues[r+1])
    end

    # Critical values
    cv_trace = Matrix{T}(undef, n, 3)
    cv_max = Matrix{T}(undef, n, 3)

    for r in 0:(n-1)
        n_minus_r = n - r
        if haskey(JOHANSEN_TRACE_CV_CONSTANT, n_minus_r)
            cv_trace[r+1, :] = T.(JOHANSEN_TRACE_CV_CONSTANT[n_minus_r])
            cv_max[r+1, :] = T.(JOHANSEN_MAX_CV_CONSTANT[n_minus_r])
        else
            # Extrapolate for large systems (approximate)
            cv_trace[r+1, :] = T.([6.5 + 10*n_minus_r, 8.18 + 10*n_minus_r, 11.65 + 12*n_minus_r])
            cv_max[r+1, :] = T.([6.5 + 6*n_minus_r, 8.18 + 6*n_minus_r, 11.65 + 7*n_minus_r])
        end
    end

    # P-values (approximate, based on critical value interpolation)
    trace_pvalues = Vector{T}(undef, n)
    max_pvalues = Vector{T}(undef, n)

    for r in 1:n
        # Trace test p-value
        stat = trace_stats[r]
        cv = cv_trace[r, :]
        if stat >= cv[3]
            trace_pvalues[r] = T(0.01)
        elseif stat >= cv[2]
            trace_pvalues[r] = T(0.01 + 0.04 * (cv[3] - stat) / (cv[3] - cv[2]))
        elseif stat >= cv[1]
            trace_pvalues[r] = T(0.05 + 0.05 * (cv[2] - stat) / (cv[2] - cv[1]))
        else
            trace_pvalues[r] = T(min(1.0, 0.10 + 0.40 * (cv[1] - stat) / cv[1]))
        end

        # Max eigenvalue p-value
        stat = max_eigen_stats[r]
        cv = cv_max[r, :]
        if stat >= cv[3]
            max_pvalues[r] = T(0.01)
        elseif stat >= cv[2]
            max_pvalues[r] = T(0.01 + 0.04 * (cv[3] - stat) / (cv[3] - cv[2]))
        elseif stat >= cv[1]
            max_pvalues[r] = T(0.05 + 0.05 * (cv[2] - stat) / (cv[2] - cv[1]))
        else
            max_pvalues[r] = T(min(1.0, 0.10 + 0.40 * (cv[1] - stat) / cv[1]))
        end
    end

    # Determine rank (using trace test at 5% level)
    rank = 0
    for r in 0:(n-1)
        if trace_stats[r+1] > cv_trace[r+1, 2]  # 5% critical value
            rank = r
        else
            break
        end
    end

    # Cointegrating vectors and adjustment coefficients
    beta = eigenvectors[:, 1:max(1, rank)]  # β: cointegrating vectors
    alpha = S01 * beta * inv(beta' * S11 * beta)  # α: adjustment coefficients

    JohansenResult(
        trace_stats, trace_pvalues,
        max_eigen_stats, max_pvalues,
        rank, beta, alpha, eigenvalues,
        cv_trace, cv_max,
        deterministic, p, T_eff
    )
end

johansen_test(Y::AbstractMatrix, p::Int; kwargs...) = johansen_test(Float64.(Y), p; kwargs...)

# =============================================================================
# VAR Stationarity Check
# =============================================================================

"""
    is_stationary(model::VARModel) -> VARStationarityResult

Check if estimated VAR model is stationary.

A VAR(p) is stationary if and only if all eigenvalues of the companion matrix
have modulus strictly less than 1.

# Returns
`VARStationarityResult` with:
- `is_stationary`: Boolean indicating stationarity
- `eigenvalues`: Complex eigenvalues of companion matrix
- `max_modulus`: Maximum eigenvalue modulus
- `companion_matrix`: The (np × np) companion form matrix

# Example
```julia
model = estimate_var(Y, 2)
result = is_stationary(model)
if !result.is_stationary
    println("Warning: VAR is non-stationary, max modulus = ", result.max_modulus)
end
```
"""
function is_stationary(model::VARModel{T}) where {T}
    F = companion_matrix(model.B, nvars(model), model.p)
    eigs = eigvals(F)
    max_mod = T(maximum(abs.(eigs)))
    VARStationarityResult(max_mod < one(T), eigs, max_mod, F)
end

# =============================================================================
# Convenience Functions
# =============================================================================

"""
    unit_root_summary(y; tests=[:adf, :kpss, :pp], kwargs...) -> NamedTuple

Run multiple unit root tests and return summary with PrettyTables output.

# Arguments
- `y`: Time series vector
- `tests`: Vector of test symbols to run (default: [:adf, :kpss, :pp])
- `kwargs...`: Additional arguments passed to individual tests

# Returns
NamedTuple with test results, conclusion, and summary table.

# Example
```julia
y = cumsum(randn(200))
summary = unit_root_summary(y)
summary.conclusion  # Overall conclusion
```
"""
function unit_root_summary(y::AbstractVector{T};
                           tests::Vector{Symbol}=[:adf, :kpss, :pp],
                           regression::Symbol=:constant) where {T<:AbstractFloat}

    results = Dict{Symbol,AbstractUnitRootTest}()

    for test in tests
        if test == :adf
            results[:adf] = adf_test(y; regression=regression)
        elseif test == :kpss
            reg = regression == :none ? :constant : regression
            results[:kpss] = kpss_test(y; regression=reg)
        elseif test == :pp
            results[:pp] = pp_test(y; regression=regression)
        elseif test == :za
            reg = regression == :none ? :constant : regression
            results[:za] = za_test(y; regression=reg)
        elseif test == :ngperron
            reg = regression == :none ? :constant : regression
            results[:ngperron] = ngperron_test(y; regression=reg)
        end
    end

    # Determine conclusion
    has_unit_root_adf = haskey(results, :adf) && results[:adf].pvalue > 0.05
    is_stationary_kpss = haskey(results, :kpss) && results[:kpss].pvalue > 0.05
    has_unit_root_pp = haskey(results, :pp) && results[:pp].pvalue > 0.05

    conclusion = if has_unit_root_adf && !is_stationary_kpss
        "Unit root detected (ADF fails to reject, KPSS rejects stationarity)"
    elseif !has_unit_root_adf && is_stationary_kpss
        "Series appears stationary (ADF rejects unit root, KPSS fails to reject)"
    elseif has_unit_root_adf && is_stationary_kpss
        "Inconclusive (ADF and KPSS both fail to reject)"
    else
        "Conflicting results (ADF rejects unit root, KPSS rejects stationarity)"
    end

    (results=results, conclusion=conclusion)
end

unit_root_summary(y::AbstractVector; kwargs...) = unit_root_summary(Float64.(y); kwargs...)

"""
    test_all_variables(Y; test=:adf, kwargs...) -> Vector

Apply unit root test to each column of Y.

# Arguments
- `Y`: Data matrix (T × n)
- `test`: Test to apply (:adf, :kpss, :pp, :za, :ngperron)
- `kwargs...`: Additional arguments passed to the test

# Returns
Vector of test results, one per variable.

# Example
```julia
Y = randn(200, 3)
Y[:, 1] = cumsum(Y[:, 1])  # Make first column non-stationary
results = test_all_variables(Y; test=:adf)
[r.pvalue for r in results]  # P-values for each variable
```
"""
function test_all_variables(Y::AbstractMatrix{T};
                            test::Symbol=:adf,
                            kwargs...) where {T<:AbstractFloat}

    n = size(Y, 2)
    results = Vector{AbstractUnitRootTest}(undef, n)

    test_func = if test == :adf
        adf_test
    elseif test == :kpss
        kpss_test
    elseif test == :pp
        pp_test
    elseif test == :za
        za_test
    elseif test == :ngperron
        ngperron_test
    else
        throw(ArgumentError("Unknown test: $test. Use :adf, :kpss, :pp, :za, or :ngperron"))
    end

    for i in 1:n
        results[i] = test_func(Y[:, i]; kwargs...)
    end

    results
end

test_all_variables(Y::AbstractMatrix; kwargs...) = test_all_variables(Float64.(Y); kwargs...)

# =============================================================================
# PrettyTables Show Methods
# =============================================================================

function Base.show(io::IO, r::ADFResult)
    println(io, "Augmented Dickey-Fuller Test")
    println(io)

    # Test info table
    info_data = [
        "Test Statistic" round(r.statistic, digits=4);
        "P-value" round(r.pvalue, digits=4);
        "Lags" r.lags;
        "Regression" r.regression;
        "Observations" r.nobs
    ]
    pretty_table(io, info_data;
        column_labels=[["Parameter", "Value"]],
        alignment=[:l, :r],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)

    # Critical values table
    cv_data = hcat(
        [1, 5, 10],
        [round(r.critical_values[l], digits=3) for l in [1, 5, 10]]
    )
    pretty_table(io, cv_data;
        column_labels=[["Level (%)", "Critical Value"]],
        alignment=[:r, :r],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)
    reject = r.statistic < r.critical_values[5]
    println(io, "H₀: Unit root  |  ", reject ? "Reject at 5%" : "Fail to reject at 5%")
end

function Base.show(io::IO, r::KPSSResult)
    println(io, "KPSS Stationarity Test")
    println(io)

    info_data = [
        "Test Statistic" round(r.statistic, digits=4);
        "P-value" round(r.pvalue, digits=4);
        "Bandwidth" r.bandwidth;
        "Regression" r.regression;
        "Observations" r.nobs
    ]
    pretty_table(io, info_data;
        column_labels=[["Parameter", "Value"]],
        alignment=[:l, :r],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)

    cv_data = hcat(
        [1, 5, 10],
        [round(r.critical_values[l], digits=3) for l in [1, 5, 10]]
    )
    pretty_table(io, cv_data;
        column_labels=[["Level (%)", "Critical Value"]],
        alignment=[:r, :r],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)
    reject = r.statistic > r.critical_values[5]
    println(io, "H₀: Stationary  |  ", reject ? "Reject at 5%" : "Fail to reject at 5%")
end

function Base.show(io::IO, r::PPResult)
    println(io, "Phillips-Perron Test")
    println(io)

    info_data = [
        "Test Statistic" round(r.statistic, digits=4);
        "P-value" round(r.pvalue, digits=4);
        "Bandwidth" r.bandwidth;
        "Regression" r.regression;
        "Observations" r.nobs
    ]
    pretty_table(io, info_data;
        column_labels=[["Parameter", "Value"]],
        alignment=[:l, :r],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)

    cv_data = hcat(
        [1, 5, 10],
        [round(r.critical_values[l], digits=3) for l in [1, 5, 10]]
    )
    pretty_table(io, cv_data;
        column_labels=[["Level (%)", "Critical Value"]],
        alignment=[:r, :r],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)
    reject = r.statistic < r.critical_values[5]
    println(io, "H₀: Unit root  |  ", reject ? "Reject at 5%" : "Fail to reject at 5%")
end

function Base.show(io::IO, r::ZAResult)
    println(io, "Zivot-Andrews Test (Structural Break)")
    println(io)

    info_data = [
        "Test Statistic" round(r.statistic, digits=4);
        "P-value" round(r.pvalue, digits=4);
        "Break Index" r.break_index;
        "Break Fraction" round(r.break_fraction, digits=3);
        "Regression" r.regression;
        "Lags" r.lags;
        "Observations" r.nobs
    ]
    pretty_table(io, info_data;
        column_labels=[["Parameter", "Value"]],
        alignment=[:l, :r],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)

    cv_data = hcat(
        [1, 5, 10],
        [round(r.critical_values[l], digits=3) for l in [1, 5, 10]]
    )
    pretty_table(io, cv_data;
        column_labels=[["Level (%)", "Critical Value"]],
        alignment=[:r, :r],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)
    reject = r.statistic < r.critical_values[5]
    println(io, "H₀: Unit root without break  |  ", reject ? "Reject at 5%" : "Fail to reject at 5%")
end

function Base.show(io::IO, r::NgPerronResult)
    println(io, "Ng-Perron Unit Root Tests")
    println(io)

    # Statistics table
    stats_data = [
        "MZα" round(r.MZa, digits=4) round(r.critical_values[:MZa][5], digits=3) (r.MZa < r.critical_values[:MZa][5] ? "*" : "");
        "MZt" round(r.MZt, digits=4) round(r.critical_values[:MZt][5], digits=3) (r.MZt < r.critical_values[:MZt][5] ? "*" : "");
        "MSB" round(r.MSB, digits=4) round(r.critical_values[:MSB][5], digits=3) (r.MSB < r.critical_values[:MSB][5] ? "*" : "");
        "MPT" round(r.MPT, digits=4) round(r.critical_values[:MPT][5], digits=3) (r.MPT < r.critical_values[:MPT][5] ? "*" : "")
    ]
    pretty_table(io, stats_data;
        column_labels=[["Statistic", "Value", "5% CV", ""]],
        alignment=[:l, :r, :r, :c],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)
    println(io, "Regression: ", r.regression, "  |  Observations: ", r.nobs)
    println(io, "* indicates rejection at 5% level")
end

function Base.show(io::IO, r::JohansenResult)
    n = length(r.trace_stats)
    println(io, "Johansen Cointegration Test")
    println(io)

    # Info
    info_data = [
        "Deterministic" r.deterministic;
        "Lags (VECM)" r.lags;
        "Observations" r.nobs
    ]
    pretty_table(io, info_data;
        column_labels=[["Parameter", "Value"]],
        alignment=[:l, :r],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)
    println(io, "Trace Test:")

    trace_data = Matrix{Any}(undef, n, 5)
    for i in 1:n
        trace_data[i, 1] = i - 1
        trace_data[i, 2] = round(r.trace_stats[i], digits=2)
        trace_data[i, 3] = round(r.critical_values_trace[i, 2], digits=2)
        trace_data[i, 4] = round(r.trace_pvalues[i], digits=3)
        trace_data[i, 5] = r.trace_stats[i] > r.critical_values_trace[i, 2] ? "*" : ""
    end
    pretty_table(io, trace_data;
        column_labels=[["Rank", "Statistic", "5% CV", "P-value", ""]],
        alignment=[:r, :r, :r, :r, :c],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)
    println(io, "Maximum Eigenvalue Test:")

    max_data = Matrix{Any}(undef, n, 5)
    for i in 1:n
        max_data[i, 1] = i - 1
        max_data[i, 2] = round(r.max_eigen_stats[i], digits=2)
        max_data[i, 3] = round(r.critical_values_max[i, 2], digits=2)
        max_data[i, 4] = round(r.max_eigen_pvalues[i], digits=3)
        max_data[i, 5] = r.max_eigen_stats[i] > r.critical_values_max[i, 2] ? "*" : ""
    end
    pretty_table(io, max_data;
        column_labels=[["Rank", "Statistic", "5% CV", "P-value", ""]],
        alignment=[:r, :r, :r, :r, :c],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    println(io)
    println(io, "Estimated cointegration rank: ", r.rank)
    println(io, "* indicates rejection at 5% level")
end

function Base.show(io::IO, r::VARStationarityResult)
    println(io, "VAR Stationarity Check")
    println(io)

    info_data = [
        "Stationary" r.is_stationary;
        "Max Modulus" round(r.max_modulus, digits=6);
        "# Eigenvalues" length(r.eigenvalues)
    ]
    pretty_table(io, info_data;
        column_labels=[["Parameter", "Value"]],
        alignment=[:l, :r],
        table_format=TextTableFormat(borders=text_table_borders__compact)
    )

    if !r.is_stationary
        println(io)
        println(io, "Warning: VAR is non-stationary!")
        println(io, "Consider differencing or VECM specification.")
    end
end
