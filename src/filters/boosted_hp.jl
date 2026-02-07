"""
Boosted HP filter (Phillips & Shi 2021).

Iteratively re-applies the HP filter to the cyclical component until
a data-driven stopping criterion is met, improving trend estimation
for series with stochastic trends.

References:
- Phillips, Peter C. B., and Zhentao Shi. 2021.
  "Boosting: Why You Can Use the HP Filter."
  *International Economic Review* 62 (2): 521–570.
- Mei, Ziwei, Peter C. B. Phillips, and Zhentao Shi. 2024.
  "The boosted HP filter is more general than you might think."
  *Journal of Applied Econometrics* 39 (7): 1260–1281.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    boosted_hp(y::AbstractVector; lambda=1600.0, stopping=:BIC, max_iter=100, sig_p=0.05) -> BoostedHPResult

Apply the boosted HP filter (Phillips & Shi 2021) for improved trend estimation.

Iteratively re-filters the cyclical component of the standard HP filter.
At each iteration, the cycle is decomposed again and the newly estimated
"trend of the cycle" is added back to the overall trend.

# Stopping criteria
- `:ADF` — Stop when the ADF test rejects the null of a unit root in the cycle
  at significance level `sig_p` (recommended for detecting stationarity)
- `:BIC` — Stop when the BIC of the cycle's AR(1) regression increases
  (selects the iteration minimizing information loss)
- `:fixed` — Run all `max_iter` iterations

# Arguments
- `y::AbstractVector`: Time series data (length ≥ 3)

# Keywords
- `lambda::Real=1600.0`: HP smoothing parameter
- `stopping::Symbol=:BIC`: Stopping criterion (`:ADF`, `:BIC`, or `:fixed`)
- `max_iter::Int=100`: Maximum number of boosting iterations
- `sig_p::Real=0.05`: Significance level for ADF stopping criterion

# Returns
- `BoostedHPResult{T}` with fields `trend`, `cycle`, `lambda`, `iterations`,
  `stopping`, `bic_path`, `adf_pvalues`, `T_obs`

# Examples
```julia
y = cumsum(randn(200))
result = boosted_hp(y)                              # BIC stopping (default)
result = boosted_hp(y; stopping=:ADF, sig_p=0.05)   # ADF stopping
result = boosted_hp(y; stopping=:fixed, max_iter=5)  # fixed iterations
```

# References
- Phillips, P. C. B., & Shi, Z. (2021). *IER* 62(2): 521–570.
- Mei, Z., Phillips, P. C. B., & Shi, Z. (2024). *JAE* 39(7): 1260–1281.
"""
function boosted_hp(y::AbstractVector{T}; lambda::Real=T(1600),
                    stopping::Symbol=:BIC, max_iter::Int=100,
                    sig_p::Real=T(0.05)) where {T<:AbstractFloat}
    T_obs = length(y)
    T_obs < 3 && throw(ArgumentError("Boosted HP requires at least 3 observations, got $T_obs"))
    lambda < 0 && throw(ArgumentError("lambda must be non-negative, got $lambda"))
    max_iter < 1 && throw(ArgumentError("max_iter must be positive, got $max_iter"))
    stopping ∈ (:ADF, :BIC, :fixed) || throw(ArgumentError(
        "stopping must be :ADF, :BIC, or :fixed, got :$stopping"))

    lam = T(lambda)
    sig = T(sig_p)
    yv = Vector{T}(y)

    # Build HP smoother matrix and factorize once
    A = _hp_penalty(T_obs, lam)
    F = cholesky(A)

    # Initial HP decomposition
    tau = Vector{T}(F \ yv)
    cyc = yv .- tau

    bic_path = T[]
    adf_pvals = T[]
    best_cyc = copy(cyc)
    best_tau = copy(tau)
    best_iter = 1

    if stopping == :BIC
        bic_val = _cycle_bic(cyc)
        push!(bic_path, bic_val)
        best_bic = bic_val
    elseif stopping == :ADF
        pval = _adf_pvalue(cyc)
        push!(adf_pvals, pval)
    end

    n_iter = 1
    for iter in 2:max_iter
        # Re-filter the cycle: extract "trend of cycle"
        cyc_trend = Vector{T}(F \ cyc)
        cyc = cyc .- cyc_trend
        tau = yv .- cyc
        n_iter = iter

        if stopping == :BIC
            bic_val = _cycle_bic(cyc)
            push!(bic_path, bic_val)
            if bic_val < best_bic
                best_bic = bic_val
                best_cyc = copy(cyc)
                best_tau = copy(tau)
                best_iter = iter
            else
                # BIC increased: stop at previous iteration
                break
            end
        elseif stopping == :ADF
            pval = _adf_pvalue(cyc)
            push!(adf_pvals, pval)
            if pval < sig
                # Cycle is stationary: done
                best_cyc = copy(cyc)
                best_tau = copy(tau)
                best_iter = iter
                break
            end
        end
        # :fixed continues to max_iter
    end

    if stopping == :fixed
        best_cyc = cyc
        best_tau = tau
        best_iter = n_iter
    elseif stopping == :ADF && isempty(adf_pvals) == false && all(p -> p >= sig, adf_pvals)
        # ADF never rejected: use last iteration
        best_cyc = cyc
        best_tau = tau
        best_iter = n_iter
    end

    BoostedHPResult(best_tau, best_cyc, lam, best_iter, stopping,
                     bic_path, adf_pvals, T_obs)
end

# Float64 fallback for non-float input
boosted_hp(y::AbstractVector; kwargs...) = boosted_hp(Float64.(y); kwargs...)

# =============================================================================
# Internal helpers
# =============================================================================

"""BIC for AR(1) fit of cycle: BIC = T ln(σ²) + 2 ln(T)."""
function _cycle_bic(cyc::Vector{T}) where {T<:AbstractFloat}
    n = length(cyc)
    n < 3 && return T(Inf)

    # Fit AR(1): c_t = a + b c_{t-1} + e_t
    y_dep = @view cyc[2:end]
    y_lag = @view cyc[1:end-1]
    n_eff = n - 1
    x_mean = mean(y_lag)
    y_mean = mean(y_dep)
    cov_xy = dot(y_dep .- y_mean, y_lag .- x_mean)
    var_x = dot(y_lag .- x_mean, y_lag .- x_mean)
    b = var_x > zero(T) ? cov_xy / var_x : zero(T)
    a = y_mean - b * x_mean
    resid = y_dep .- a .- b .* y_lag
    sigma2 = dot(resid, resid) / n_eff
    sigma2 <= zero(T) && return T(-Inf)
    T(n_eff) * log(sigma2) + T(2) * log(T(n_eff))
end

"""Get ADF p-value for cycle stationarity check."""
function _adf_pvalue(cyc::Vector{T}) where {T<:AbstractFloat}
    n = length(cyc)
    n < 5 && return one(T)  # too short to test
    try
        result = adf_test(cyc; regression=:constant)
        return T(result.pvalue)
    catch
        return one(T)  # if test fails, assume non-stationary
    end
end
