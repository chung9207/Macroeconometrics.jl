"""
Stochastic Volatility model estimation via Turing.jl MCMC.

Three variants:
1. Basic SV (Taylor 1986)
2. SV with leverage (correlated innovations)
3. SV with Student-t errors
"""

using Turing, MCMCChains

# =============================================================================
# Turing Model: Basic SV
# =============================================================================

@model function sv_model(y::Vector{T}) where {T<:AbstractFloat}
    n = length(y)

    # Priors
    mu ~ Normal(T(0), T(10))
    phi_raw ~ Beta(T(20), T(1.5))
    phi = T(2) * phi_raw - one(T)  # Map to (-1, 1)
    sigma_eta ~ truncated(Normal(T(0), T(1)), T(0), T(Inf))

    # Latent log-volatilities
    h = Vector{T}(undef, n)
    # Stationary initialization
    h_var = sigma_eta^2 / (one(T) - phi^2 + T(1e-8))
    h[1] ~ Normal(mu, sqrt(max(h_var, T(1e-8))))

    for t in 2:n
        h[t] ~ Normal(mu + phi * (h[t-1] - mu), sigma_eta)
    end

    # Observation equation
    for t in 1:n
        vol = exp(h[t] / 2)
        y[t] ~ Normal(T(0), vol)
    end
end

# =============================================================================
# Turing Model: SV with Leverage
# =============================================================================

@model function sv_leverage_model(y::Vector{T}) where {T<:AbstractFloat}
    n = length(y)

    mu ~ Normal(T(0), T(10))
    phi_raw ~ Beta(T(20), T(1.5))
    phi = T(2) * phi_raw - one(T)
    sigma_eta ~ truncated(Normal(T(0), T(1)), T(0), T(Inf))
    rho ~ Uniform(T(-1), T(1))

    h = Vector{T}(undef, n)
    h_var = sigma_eta^2 / (one(T) - phi^2 + T(1e-8))
    h[1] ~ Normal(mu, sqrt(max(h_var, T(1e-8))))

    for t in 2:n
        # Leverage: conditional mean of h_t depends on z_{t-1}
        z_prev = y[t-1] / exp(h[t-1] / 2)
        h_mean = mu + phi * (h[t-1] - mu) + rho * sigma_eta * z_prev
        h[t] ~ Normal(h_mean, sigma_eta * sqrt(max(one(T) - rho^2, T(1e-8))))
    end

    for t in 1:n
        vol = exp(h[t] / 2)
        y[t] ~ Normal(T(0), vol)
    end
end

# =============================================================================
# Turing Model: SV with Student-t
# =============================================================================

@model function sv_studentt_model(y::Vector{T}) where {T<:AbstractFloat}
    n = length(y)

    mu ~ Normal(T(0), T(10))
    phi_raw ~ Beta(T(20), T(1.5))
    phi = T(2) * phi_raw - one(T)
    sigma_eta ~ truncated(Normal(T(0), T(1)), T(0), T(Inf))
    nu ~ truncated(Normal(T(10), T(5)), T(2.01), T(Inf))

    h = Vector{T}(undef, n)
    h_var = sigma_eta^2 / (one(T) - phi^2 + T(1e-8))
    h[1] ~ Normal(mu, sqrt(max(h_var, T(1e-8))))

    for t in 2:n
        h[t] ~ Normal(mu + phi * (h[t-1] - mu), sigma_eta)
    end

    for t in 1:n
        vol = exp(h[t] / 2)
        y[t] ~ LocationScale(T(0), vol, TDist(nu))
    end
end

# =============================================================================
# Public API
# =============================================================================

"""
    estimate_sv(y; n_samples=2000, n_adapts=1000, sampler=:nuts,
                dist=:normal, leverage=false,
                quantile_levels=[0.025, 0.5, 0.975]) -> SVModel

Estimate a Stochastic Volatility model via Bayesian MCMC.

# Model
    yₜ = exp(hₜ/2) εₜ
    hₜ = μ + φ(hₜ₋₁ - μ) + σ_η ηₜ

# Arguments
- `y`: Time series vector
- `n_samples`: Number of posterior samples (default 2000)
- `n_adapts`: Number of adaptation steps for NUTS (default 1000)
- `sampler`: MCMC sampler (:nuts, :hmc, etc.)
- `dist`: Error distribution (:normal or :studentt)
- `leverage`: Whether to include leverage effect (correlated innovations)
- `quantile_levels`: Quantile levels for volatility posterior

# Example
```julia
y = randn(200) .* exp.(cumsum(0.1 .* randn(200)) ./ 2)
model = estimate_sv(y; n_samples=1000)
println("φ = ", mean(model.phi_post))
```
"""
function estimate_sv(y::AbstractVector{T};
                     n_samples::Int=2000, n_adapts::Int=1000,
                     sampler::Symbol=:nuts,
                     dist::Symbol=:normal, leverage::Bool=false,
                     quantile_levels::Vector{<:Real}=[0.025, 0.5, 0.975]) where {T<:AbstractFloat}
    n = length(y)
    n < 20 && throw(ArgumentError("Need at least 20 observations for SV model, got $n"))
    y_vec = Vector{T}(y)

    # Select model
    if leverage && dist == :normal
        turing_model = sv_leverage_model(y_vec)
    elseif dist == :studentt
        turing_model = sv_studentt_model(y_vec)
    else
        turing_model = sv_model(y_vec)
    end

    # Get sampler
    samp = get_sampler(sampler, n_adapts, NamedTuple())

    # Run MCMC
    chain = sample(turing_model, samp, n_samples; progress=false)

    # Extract parameters
    mu_post = T.(vec(chain[:mu].data))
    phi_raw_post = T.(vec(chain[:phi_raw].data))
    phi_post = T(2) .* phi_raw_post .- one(T)
    sigma_eta_post = T.(vec(chain[:sigma_eta].data))

    # Extract latent volatilities
    ql = T.(quantile_levels)
    vol_mean = Vector{T}(undef, n)
    vol_quantiles = Matrix{T}(undef, n, length(ql))

    for t in 1:n
        h_sym = Symbol("h[$t]")
        h_draws = T.(vec(chain[h_sym].data))
        vol_draws = exp.(h_draws)
        vol_mean[t] = mean(vol_draws)
        for (j, q) in enumerate(ql)
            vol_quantiles[t, j] = quantile(vol_draws, q)
        end
    end

    SVModel(y_vec, chain, mu_post, phi_post, sigma_eta_post,
            vol_mean, vol_quantiles, ql, dist, leverage, n_samples)
end

estimate_sv(y::AbstractVector; kwargs...) = estimate_sv(Float64.(y); kwargs...)
