"""
Type definitions for Local Projection methods.

Implements types for:
- Core LP estimation (Jordà 2005)
- LP with IV (Stock & Watson 2018)
- Smooth LP with B-splines (Barnichon & Brownlees 2019)
- State-dependent LP (Auerbach & Gorodnichenko 2013)
- Propensity score LP (Angrist et al. 2018)

Note: GMM types are defined in gmm.jl
Note: Covariance estimator types are defined in covariance_estimators.jl
"""

using LinearAlgebra, StatsAPI

# =============================================================================
# Abstract Types
# =============================================================================

"""Abstract supertype for Local Projection models."""
abstract type AbstractLPModel <: StatsAPI.RegressionModel end

"""Abstract supertype for LP impulse response results."""
abstract type AbstractLPImpulseResponse <: AbstractImpulseResponse end

# Note: AbstractCovarianceEstimator and its subtypes (NeweyWestEstimator, WhiteEstimator,
# DriscollKraayEstimator) are now defined in covariance_estimators.jl

# =============================================================================
# Core LP Model Type (Jordà 2005)
# =============================================================================

"""
    LPModel{T} <: AbstractLPModel

Local Projection model estimated via OLS with robust standard errors (Jordà 2005).

The LP regression for horizon h:
    y_{t+h} = α_h + β_h * shock_t + Γ_h * controls_t + ε_{t+h}

Fields:
- Y: Response data matrix (T_obs × n_vars)
- shock_var: Index of shock variable in Y
- response_vars: Indices of response variables (default: all)
- horizon: Maximum IRF horizon H
- lags: Number of control lags included
- B: Vector of coefficient matrices, one per horizon h=0,...,H
- residuals: Vector of residual matrices per horizon
- vcov: Vector of robust covariance matrices per horizon
- T_eff: Effective sample sizes per horizon
- cov_estimator: Covariance estimator used
"""
struct LPModel{T<:AbstractFloat} <: AbstractLPModel
    Y::Matrix{T}
    shock_var::Int
    response_vars::Vector{Int}
    horizon::Int
    lags::Int
    B::Vector{Matrix{T}}
    residuals::Vector{Matrix{T}}
    vcov::Vector{Matrix{T}}
    T_eff::Vector{Int}
    cov_estimator::AbstractCovarianceEstimator

    function LPModel(Y::Matrix{T}, shock_var::Int, response_vars::Vector{Int},
                     horizon::Int, lags::Int, B::Vector{Matrix{T}},
                     residuals::Vector{Matrix{T}}, vcov::Vector{Matrix{T}},
                     T_eff::Vector{Int}, cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert 1 <= shock_var <= n "shock_var must be in 1:$n"
        @assert all(1 .<= response_vars .<= n) "response_vars must be in 1:$n"
        @assert horizon >= 0 "horizon must be non-negative"
        @assert lags >= 0 "lags must be non-negative"
        @assert length(B) == horizon + 1 "B must have H+1 elements"
        @assert length(residuals) == horizon + 1 "residuals must have H+1 elements"
        @assert length(vcov) == horizon + 1 "vcov must have H+1 elements"
        @assert length(T_eff) == horizon + 1 "T_eff must have H+1 elements"
        new{T}(Y, shock_var, response_vars, horizon, lags, B, residuals, vcov, T_eff, cov_estimator)
    end
end

# Convenience constructor with type promotion
function LPModel(Y::AbstractMatrix, shock_var::Int, response_vars::Vector{Int},
                 horizon::Int, lags::Int, B::Vector{<:AbstractMatrix},
                 residuals::Vector{<:AbstractMatrix}, vcov::Vector{<:AbstractMatrix},
                 T_eff::Vector{Int}, cov_estimator::AbstractCovarianceEstimator)
    T = promote_type(eltype(Y), eltype(first(B)))
    LPModel(Matrix{T}(Y), shock_var, response_vars, horizon, lags,
            [Matrix{T}(b) for b in B], [Matrix{T}(r) for r in residuals],
            [Matrix{T}(v) for v in vcov], T_eff, cov_estimator)
end

# Accessors
nvars(model::LPModel) = size(model.Y, 2)
nlags(model::LPModel) = model.lags
nhorizons(model::LPModel) = model.horizon + 1
nresponse(model::LPModel) = length(model.response_vars)

# =============================================================================
# LP Impulse Response Type
# =============================================================================

"""
    LPImpulseResponse{T} <: AbstractLPImpulseResponse

LP-based impulse response function with confidence intervals from robust standard errors.

Fields:
- values: Point estimates (H+1 × n_response)
- ci_lower: Lower CI bounds
- ci_upper: Upper CI bounds
- se: Standard errors
- horizon: Maximum horizon
- response_vars: Names of response variables
- shock_var: Name of shock variable
- cov_type: Covariance estimator type
- conf_level: Confidence level used
"""
struct LPImpulseResponse{T<:AbstractFloat} <: AbstractLPImpulseResponse
    values::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    se::Matrix{T}
    horizon::Int
    response_vars::Vector{String}
    shock_var::String
    cov_type::Symbol
    conf_level::T

    function LPImpulseResponse{T}(values::Matrix{T}, ci_lower::Matrix{T}, ci_upper::Matrix{T},
                                   se::Matrix{T}, horizon::Int, response_vars::Vector{String},
                                   shock_var::String, cov_type::Symbol, conf_level::T) where {T<:AbstractFloat}
        @assert size(values) == size(ci_lower) == size(ci_upper) == size(se)
        @assert size(values, 1) == horizon + 1
        @assert length(response_vars) == size(values, 2)
        @assert 0 < conf_level < 1
        new{T}(values, ci_lower, ci_upper, se, horizon, response_vars, shock_var, cov_type, conf_level)
    end
end

# =============================================================================
# LP-IV Model Type (Stock & Watson 2018)
# =============================================================================

"""
    LPIVModel{T} <: AbstractLPModel

Local Projection with Instrumental Variables (Stock & Watson 2018).
Uses 2SLS estimation at each horizon.

Fields:
- Y: Response data matrix
- shock_var: Index of endogenous shock variable
- response_vars: Indices of response variables
- instruments: Instrument matrix (T × n_instruments)
- horizon: Maximum horizon
- lags: Number of control lags
- B: 2SLS coefficient matrices per horizon
- residuals: Residuals per horizon
- vcov: Robust covariance matrices per horizon
- first_stage_F: First-stage F-statistics per horizon (for weak IV test)
- first_stage_coef: First-stage coefficients per horizon
- T_eff: Effective sample sizes
- cov_estimator: Covariance estimator used
"""
struct LPIVModel{T<:AbstractFloat} <: AbstractLPModel
    Y::Matrix{T}
    shock_var::Int
    response_vars::Vector{Int}
    instruments::Matrix{T}
    horizon::Int
    lags::Int
    B::Vector{Matrix{T}}
    residuals::Vector{Matrix{T}}
    vcov::Vector{Matrix{T}}
    first_stage_F::Vector{T}
    first_stage_coef::Vector{Vector{T}}
    T_eff::Vector{Int}
    cov_estimator::AbstractCovarianceEstimator

    function LPIVModel{T}(Y::Matrix{T}, shock_var::Int, response_vars::Vector{Int},
                          instruments::Matrix{T}, horizon::Int, lags::Int,
                          B::Vector{Matrix{T}}, residuals::Vector{Matrix{T}},
                          vcov::Vector{Matrix{T}}, first_stage_F::Vector{T},
                          first_stage_coef::Vector{Vector{T}}, T_eff::Vector{Int},
                          cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert 1 <= shock_var <= n "shock_var must be in 1:$n"
        @assert all(1 .<= response_vars .<= n) "response_vars must be in 1:$n"
        @assert size(instruments, 1) == size(Y, 1) "instruments must have same T as Y"
        @assert size(instruments, 2) >= 1 "need at least one instrument"
        @assert length(first_stage_F) == horizon + 1
        new{T}(Y, shock_var, response_vars, instruments, horizon, lags, B, residuals,
               vcov, first_stage_F, first_stage_coef, T_eff, cov_estimator)
    end
end

n_instruments(model::LPIVModel) = size(model.instruments, 2)

# =============================================================================
# B-Spline Basis Type
# =============================================================================

"""
    BSplineBasis{T} <: Any

B-spline basis for smooth LP (Barnichon & Brownlees 2019).

Fields:
- degree: Spline degree (typically 3 for cubic)
- n_interior_knots: Number of interior knots
- knots: Full knot vector including boundary knots
- basis_matrix: Precomputed basis matrix at horizon points (H+1 × n_basis)
- horizons: Horizon points where basis is evaluated
"""
struct BSplineBasis{T<:AbstractFloat}
    degree::Int
    n_interior_knots::Int
    knots::Vector{T}
    basis_matrix::Matrix{T}
    horizons::Vector{Int}

    function BSplineBasis{T}(degree::Int, n_interior_knots::Int, knots::Vector{T},
                             basis_matrix::Matrix{T}, horizons::Vector{Int}) where {T<:AbstractFloat}
        @assert degree >= 0 "degree must be non-negative"
        @assert n_interior_knots >= 0 "n_interior_knots must be non-negative"
        n_basis = n_interior_knots + degree + 1
        @assert size(basis_matrix, 2) == n_basis "basis_matrix columns must equal n_basis"
        @assert size(basis_matrix, 1) == length(horizons) "basis_matrix rows must equal length(horizons)"
        new{T}(degree, n_interior_knots, knots, basis_matrix, horizons)
    end
end

n_basis(basis::BSplineBasis) = basis.n_interior_knots + basis.degree + 1

# =============================================================================
# Smooth LP Model Type (Barnichon & Brownlees 2019)
# =============================================================================

"""
    SmoothLPModel{T} <: AbstractLPModel

Smooth Local Projection with B-spline basis (Barnichon & Brownlees 2019).

The IRF is parameterized as: β(h) = Σ_j θ_j B_j(h)
where B_j are B-spline basis functions.

Fields:
- Y: Response data matrix
- shock_var: Shock variable index
- response_vars: Response variable indices
- horizon: Maximum horizon
- lags: Number of control lags
- spline_basis: B-spline basis configuration
- theta: Spline coefficients (n_basis × n_response)
- vcov_theta: Covariance of theta (vectorized)
- lambda: Smoothing penalty parameter
- irf_values: Smoothed IRF point estimates (H+1 × n_response)
- irf_se: Standard errors of smoothed IRF
- residuals: Pooled residuals
- T_eff: Effective sample size
- cov_estimator: Covariance estimator used
"""
struct SmoothLPModel{T<:AbstractFloat} <: AbstractLPModel
    Y::Matrix{T}
    shock_var::Int
    response_vars::Vector{Int}
    horizon::Int
    lags::Int
    spline_basis::BSplineBasis{T}
    theta::Matrix{T}
    vcov_theta::Matrix{T}
    lambda::T
    irf_values::Matrix{T}
    irf_se::Matrix{T}
    residuals::Matrix{T}
    T_eff::Int
    cov_estimator::AbstractCovarianceEstimator

    function SmoothLPModel{T}(Y::Matrix{T}, shock_var::Int, response_vars::Vector{Int},
                              horizon::Int, lags::Int, spline_basis::BSplineBasis{T},
                              theta::Matrix{T}, vcov_theta::Matrix{T}, lambda::T,
                              irf_values::Matrix{T}, irf_se::Matrix{T}, residuals::Matrix{T},
                              T_eff::Int, cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert 1 <= shock_var <= n
        @assert all(1 .<= response_vars .<= n)
        @assert lambda >= 0 "lambda must be non-negative"
        @assert size(theta, 1) == n_basis(spline_basis)
        @assert size(theta, 2) == length(response_vars)
        new{T}(Y, shock_var, response_vars, horizon, lags, spline_basis, theta,
               vcov_theta, lambda, irf_values, irf_se, residuals, T_eff, cov_estimator)
    end
end

# =============================================================================
# State Transition Type
# =============================================================================

"""
    StateTransition{T} <: Any

Smooth state transition function for state-dependent LP.

F(z_t) = exp(-γ(z_t - c)) / (1 + exp(-γ(z_t - c)))

Fields:
- state_var: State variable values (standardized)
- gamma: Transition smoothness parameter (higher = sharper)
- threshold: Transition threshold c
- method: Transition function type (:logistic, :exponential, :indicator)
- F_values: Precomputed transition function values
"""
struct StateTransition{T<:AbstractFloat}
    state_var::Vector{T}
    gamma::T
    threshold::T
    method::Symbol
    F_values::Vector{T}

    function StateTransition(state_var::Vector{T}, gamma::T, threshold::T,
                             method::Symbol=:logistic) where {T<:AbstractFloat}
        @assert gamma > 0 "gamma must be positive"
        method ∉ (:logistic, :exponential, :indicator) &&
            throw(ArgumentError("method must be :logistic, :exponential, or :indicator"))

        # Compute F values
        F_values = if method == :logistic
            @. exp(-gamma * (state_var - threshold)) / (1 + exp(-gamma * (state_var - threshold)))
        elseif method == :exponential
            @. 1 - exp(-gamma * (state_var - threshold)^2)
        else  # :indicator
            T.(state_var .>= threshold)
        end

        new{T}(state_var, gamma, threshold, method, F_values)
    end
end

# =============================================================================
# State-Dependent LP Model Type (Auerbach & Gorodnichenko 2013)
# =============================================================================

"""
    StateLPModel{T} <: AbstractLPModel

State-dependent Local Projection (Auerbach & Gorodnichenko 2013).

Model: y_{t+h} = F(z_t)[α_E + β_E * shock_t + ...] + (1-F(z_t))[α_R + β_R * shock_t + ...]

F(z) is a smooth transition function, typically logistic.
State E = expansion (high z), State R = recession (low z).

Fields:
- Y: Response data matrix
- shock_var: Shock variable index
- response_vars: Response variable indices
- horizon: Maximum horizon
- lags: Number of control lags
- state: StateTransition configuration
- B_expansion: Coefficients in expansion state (per horizon)
- B_recession: Coefficients in recession state (per horizon)
- residuals: Residuals per horizon
- vcov_expansion: Covariance in expansion (per horizon)
- vcov_recession: Covariance in recession (per horizon)
- vcov_diff: Covariance of difference (per horizon)
- T_eff: Effective sample sizes
- cov_estimator: Covariance estimator used
"""
struct StateLPModel{T<:AbstractFloat} <: AbstractLPModel
    Y::Matrix{T}
    shock_var::Int
    response_vars::Vector{Int}
    horizon::Int
    lags::Int
    state::StateTransition{T}
    B_expansion::Vector{Matrix{T}}
    B_recession::Vector{Matrix{T}}
    residuals::Vector{Matrix{T}}
    vcov_expansion::Vector{Matrix{T}}
    vcov_recession::Vector{Matrix{T}}
    vcov_diff::Vector{Matrix{T}}
    T_eff::Vector{Int}
    cov_estimator::AbstractCovarianceEstimator

    function StateLPModel{T}(Y::Matrix{T}, shock_var::Int, response_vars::Vector{Int},
                             horizon::Int, lags::Int, state::StateTransition{T},
                             B_expansion::Vector{Matrix{T}}, B_recession::Vector{Matrix{T}},
                             residuals::Vector{Matrix{T}}, vcov_expansion::Vector{Matrix{T}},
                             vcov_recession::Vector{Matrix{T}}, vcov_diff::Vector{Matrix{T}},
                             T_eff::Vector{Int}, cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert 1 <= shock_var <= n
        @assert all(1 .<= response_vars .<= n)
        @assert length(state.state_var) == size(Y, 1) "state_var must have same length as Y"
        @assert length(B_expansion) == horizon + 1
        @assert length(B_recession) == horizon + 1
        new{T}(Y, shock_var, response_vars, horizon, lags, state, B_expansion,
               B_recession, residuals, vcov_expansion, vcov_recession, vcov_diff,
               T_eff, cov_estimator)
    end
end

# =============================================================================
# Propensity Score Configuration
# =============================================================================

"""
    PropensityScoreConfig{T} <: Any

Configuration for propensity score estimation and IPW.

Fields:
- method: Propensity model (:logit, :probit)
- trimming: (lower, upper) bounds for propensity scores
- normalize: Normalize weights to sum to 1 within groups
"""
struct PropensityScoreConfig{T<:AbstractFloat}
    method::Symbol
    trimming::Tuple{T,T}
    normalize::Bool

    function PropensityScoreConfig{T}(method::Symbol, trimming::Tuple{T,T},
                                       normalize::Bool) where {T<:AbstractFloat}
        method ∉ (:logit, :probit) && throw(ArgumentError("method must be :logit or :probit"))
        @assert 0 <= trimming[1] < trimming[2] <= 1 "trimming must be in [0,1]"
        new{T}(method, trimming, normalize)
    end
end

function PropensityScoreConfig(; method::Symbol=:logit,
                                 trimming::Tuple{<:Real,<:Real}=(0.01, 0.99),
                                 normalize::Bool=true)
    PropensityScoreConfig{Float64}(method, (Float64(trimming[1]), Float64(trimming[2])), normalize)
end

# =============================================================================
# Propensity Score LP Model Type (Angrist et al. 2018)
# =============================================================================

"""
    PropensityLPModel{T} <: AbstractLPModel

Local Projection with Inverse Propensity Weighting (Angrist et al. 2018).

Estimates Average Treatment Effect (ATE) at each horizon using IPW.

Fields:
- Y: Response data matrix
- treatment: Binary treatment indicator vector
- response_vars: Response variable indices
- covariates: Covariate matrix for propensity model
- horizon: Maximum horizon
- propensity_scores: Estimated propensity scores P(D=1|X)
- ipw_weights: Inverse propensity weights
- B: IPW-weighted regression coefficients per horizon
- residuals: Residuals per horizon
- vcov: Robust covariance matrices per horizon
- ate: Average treatment effects per horizon (for each response var)
- ate_se: Standard errors of ATE
- config: Propensity score configuration
- T_eff: Effective sample sizes
- cov_estimator: Covariance estimator used
"""
struct PropensityLPModel{T<:AbstractFloat} <: AbstractLPModel
    Y::Matrix{T}
    treatment::Vector{Bool}
    response_vars::Vector{Int}
    covariates::Matrix{T}
    horizon::Int
    propensity_scores::Vector{T}
    ipw_weights::Vector{T}
    B::Vector{Matrix{T}}
    residuals::Vector{Matrix{T}}
    vcov::Vector{Matrix{T}}
    ate::Matrix{T}  # (H+1) × n_response
    ate_se::Matrix{T}
    config::PropensityScoreConfig{T}
    T_eff::Vector{Int}
    cov_estimator::AbstractCovarianceEstimator

    function PropensityLPModel{T}(Y::Matrix{T}, treatment::AbstractVector{Bool}, response_vars::Vector{Int},
                                  covariates::Matrix{T}, horizon::Int, propensity_scores::Vector{T},
                                  ipw_weights::Vector{T}, B::Vector{Matrix{T}},
                                  residuals::Vector{Matrix{T}}, vcov::Vector{Matrix{T}},
                                  ate::Matrix{T}, ate_se::Matrix{T}, config::PropensityScoreConfig{T},
                                  T_eff::Vector{Int}, cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert length(treatment) == size(Y, 1)
        @assert all(1 .<= response_vars .<= n)
        @assert size(covariates, 1) == size(Y, 1)
        @assert length(propensity_scores) == size(Y, 1)
        @assert all(0 .<= propensity_scores .<= 1)
        @assert size(ate, 1) == horizon + 1
        @assert size(ate, 2) == length(response_vars)
        new{T}(Y, collect(Bool, treatment), response_vars, covariates, horizon, propensity_scores,
               ipw_weights, B, residuals, vcov, ate, ate_se, config, T_eff, cov_estimator)
    end
end

n_treated(model::PropensityLPModel) = sum(model.treatment)
n_control(model::PropensityLPModel) = sum(.!model.treatment)

# =============================================================================
# Structural LP Type (Plagborg-Møller & Wolf 2021)
# =============================================================================

"""
    StructuralLP{T} <: AbstractFrequentistResult

Structural Local Projection result combining VAR-based identification with LP estimation.

Estimates multi-shock IRFs by computing orthogonalized structural shocks from a VAR model
and using them as regressors in LP regressions (Plagborg-Møller & Wolf 2021).

Fields:
- `irf`: 3D impulse responses (H × n × n) — reuses `ImpulseResponse{T}`
- `structural_shocks`: Structural shocks (T_eff × n)
- `var_model`: Underlying VAR model used for identification
- `Q`: Rotation/identification matrix
- `method`: Identification method used (:cholesky, :sign, :long_run, :fastica, etc.)
- `lags`: Number of LP control lags
- `cov_type`: HAC estimator type
- `se`: Standard errors (H × n × n)
- `lp_models`: Individual LP model per shock
"""
struct StructuralLP{T<:AbstractFloat} <: AbstractFrequentistResult
    irf::ImpulseResponse{T}
    structural_shocks::Matrix{T}
    var_model::VARModel{T}
    Q::Matrix{T}
    method::Symbol
    lags::Int
    cov_type::Symbol
    se::Array{T,3}
    lp_models::Vector{LPModel{T}}
end

# Accessors
nvars(slp::StructuralLP) = nvars(slp.var_model)

# =============================================================================
# LP Forecast Type
# =============================================================================

"""
    LPForecast{T}

Direct multi-step LP forecast result.

Each horizon h uses its own regression coefficients directly (no recursion),
producing ŷ_{T+h} = α_h + β_h·shock_h + Γ_h·controls_T.

Fields:
- `forecasts`: Point forecasts (H × n_response)
- `ci_lower`: Lower CI bounds (H × n_response)
- `ci_upper`: Upper CI bounds (H × n_response)
- `se`: Standard errors (H × n_response)
- `horizon`: Maximum forecast horizon
- `response_vars`: Response variable indices
- `shock_var`: Shock variable index
- `shock_path`: Assumed shock trajectory
- `conf_level`: Confidence level
- `ci_method`: CI method (:analytical, :bootstrap, :none)
"""
struct LPForecast{T<:AbstractFloat}
    forecasts::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    se::Matrix{T}
    horizon::Int
    response_vars::Vector{Int}
    shock_var::Int
    shock_path::Vector{T}
    conf_level::T
    ci_method::Symbol

    function LPForecast(forecasts::Matrix{T}, ci_lower::Matrix{T}, ci_upper::Matrix{T},
                        se::Matrix{T}, horizon::Int, response_vars::Vector{Int},
                        shock_var::Int, shock_path::Vector{T}, conf_level::T,
                        ci_method::Symbol) where {T<:AbstractFloat}
        @assert size(forecasts) == size(ci_lower) == size(ci_upper) == size(se)
        @assert size(forecasts, 1) == horizon
        @assert size(forecasts, 2) == length(response_vars)
        @assert length(shock_path) == horizon
        @assert 0 < conf_level < 1
        @assert ci_method ∈ (:analytical, :bootstrap, :none)
        new{T}(forecasts, ci_lower, ci_upper, se, horizon, response_vars,
               shock_var, shock_path, conf_level, ci_method)
    end
end

# =============================================================================
# StatsAPI Interface for LP Models
# =============================================================================

StatsAPI.coef(model::LPModel) = model.B
StatsAPI.coef(model::LPModel, h::Int) = model.B[h + 1]
StatsAPI.residuals(model::LPModel) = model.residuals
StatsAPI.residuals(model::LPModel, h::Int) = model.residuals[h + 1]
StatsAPI.vcov(model::LPModel) = model.vcov
StatsAPI.vcov(model::LPModel, h::Int) = model.vcov[h + 1]
StatsAPI.nobs(model::LPModel) = size(model.Y, 1)
StatsAPI.nobs(model::LPModel, h::Int) = model.T_eff[h + 1]
StatsAPI.dof(model::LPModel) = sum(length(b) for b in model.B)
StatsAPI.islinear(::LPModel) = true

# For LPIVModel
StatsAPI.coef(model::LPIVModel) = model.B
StatsAPI.coef(model::LPIVModel, h::Int) = model.B[h + 1]
StatsAPI.residuals(model::LPIVModel) = model.residuals
StatsAPI.vcov(model::LPIVModel) = model.vcov
StatsAPI.nobs(model::LPIVModel) = size(model.Y, 1)
StatsAPI.islinear(::LPIVModel) = true

# For SmoothLPModel
StatsAPI.coef(model::SmoothLPModel) = model.theta
StatsAPI.residuals(model::SmoothLPModel) = model.residuals
StatsAPI.vcov(model::SmoothLPModel) = model.vcov_theta
StatsAPI.nobs(model::SmoothLPModel) = size(model.Y, 1)
StatsAPI.islinear(::SmoothLPModel) = true

# For StateLPModel
StatsAPI.nobs(model::StateLPModel) = size(model.Y, 1)
StatsAPI.islinear(::StateLPModel) = true

# For PropensityLPModel
StatsAPI.coef(model::PropensityLPModel) = model.B
StatsAPI.residuals(model::PropensityLPModel) = model.residuals
StatsAPI.vcov(model::PropensityLPModel) = model.vcov
StatsAPI.nobs(model::PropensityLPModel) = size(model.Y, 1)
StatsAPI.islinear(::PropensityLPModel) = true

