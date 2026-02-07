"""
Type definitions and StatsAPI interface for Stochastic Volatility models.
"""

using MCMCChains

# =============================================================================
# SV Model Type
# =============================================================================

"""
    SVModel{T} <: AbstractVolatilityModel

Stochastic Volatility model (Taylor 1986), estimated via Bayesian MCMC:
    yₜ = exp(hₜ/2) εₜ,       εₜ ~ N(0,1)
    hₜ = μ + φ(hₜ₋₁ - μ) + σ_η ηₜ,  ηₜ ~ N(0,1)

# Fields
- `y::Vector{T}`: Original data
- `chain::Chains`: Full MCMC chain
- `mu_post::Vector{T}`: Posterior draws of μ (log-variance level)
- `phi_post::Vector{T}`: Posterior draws of φ (persistence)
- `sigma_eta_post::Vector{T}`: Posterior draws of σ_η (volatility of volatility)
- `volatility_mean::Vector{T}`: Posterior mean of exp(hₜ) at each time t
- `volatility_quantiles::Matrix{T}`: Quantiles of exp(hₜ) (T × n_quantiles)
- `quantile_levels::Vector{T}`: Quantile levels (e.g., [0.025, 0.5, 0.975])
- `dist::Symbol`: Error distribution (:normal or :studentt)
- `leverage::Bool`: Whether leverage effect was estimated
- `n_samples::Int`: Number of posterior samples
"""
struct SVModel{T<:AbstractFloat} <: AbstractVolatilityModel
    y::Vector{T}
    chain::Chains
    mu_post::Vector{T}
    phi_post::Vector{T}
    sigma_eta_post::Vector{T}
    volatility_mean::Vector{T}
    volatility_quantiles::Matrix{T}
    quantile_levels::Vector{T}
    dist::Symbol
    leverage::Bool
    n_samples::Int
end

# Accessors needed by _show_volatility_model
# SVModel doesn't have omega/alpha/beta, so we define mu/omega for display compatibility
function persistence(m::SVModel{T}) where {T}
    mean(m.phi_post)
end

function unconditional_variance(m::SVModel{T}) where {T}
    mu = mean(m.mu_post)
    exp(mu)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.nobs(m::SVModel) = length(m.y)
StatsAPI.coef(m::SVModel) = [mean(m.mu_post), mean(m.phi_post), mean(m.sigma_eta_post)]
StatsAPI.residuals(m::SVModel) = m.y ./ sqrt.(m.volatility_mean)
StatsAPI.predict(m::SVModel) = m.volatility_mean
StatsAPI.islinear(::SVModel) = false

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::SVModel{T}) where {T}
    mu_m = mean(m.mu_post)
    phi_m = mean(m.phi_post)
    se_m = mean(m.sigma_eta_post)

    mu_s = std(m.mu_post)
    phi_s = std(m.phi_post)
    se_s = std(m.sigma_eta_post)

    mu_q = quantile(m.mu_post, [0.025, 0.5, 0.975])
    phi_q = quantile(m.phi_post, [0.025, 0.5, 0.975])
    se_q = quantile(m.sigma_eta_post, [0.025, 0.5, 0.975])

    title = "Stochastic Volatility Model"
    m.leverage && (title *= " (with leverage)")
    m.dist == :studentt && (title *= " (Student-t)")

    data = Any[
        "μ"   _fmt(mu_m)  _fmt(mu_s)  _fmt(mu_q[1])  _fmt(mu_q[2])  _fmt(mu_q[3]);
        "φ"   _fmt(phi_m) _fmt(phi_s) _fmt(phi_q[1])  _fmt(phi_q[2])  _fmt(phi_q[3]);
        "σ_η" _fmt(se_m)  _fmt(se_s)  _fmt(se_q[1])  _fmt(se_q[2])  _fmt(se_q[3])
    ]

    _pretty_table(io, data;
        title = title,
        column_labels = ["Parameter", "Mean", "Std", "2.5%", "50%", "97.5%"],
        alignment = [:l, :r, :r, :r, :r, :r],
    )

    info_data = Any[
        "Observations" string(nobs(m));
        "Posterior samples" string(m.n_samples);
        "Distribution" string(m.dist);
        "Leverage" string(m.leverage)
    ]
    _pretty_table(io, info_data;
        column_labels = ["Info", "Value"],
        alignment = [:l, :r],
    )
end
