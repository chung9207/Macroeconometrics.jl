"""
Stochastic Volatility posterior predictive forecasting.
"""

"""
    forecast(m::SVModel, h; conf_level=0.95) -> VolatilityForecast

Posterior predictive forecast of volatility from an SV model.

For each MCMC draw (μ, φ, σ_η), simulates the log-volatility path forward
h_{T+1}, ..., h_{T+h} and returns quantiles of exp(hₜ).

# Arguments
- `m`: Fitted SVModel
- `h`: Forecast horizon
- `conf_level`: Confidence level for intervals (default 0.95)
"""
function forecast(m::SVModel{T}, h::Int; conf_level::T=T(0.95)) where {T}
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))

    n_draws = m.n_samples
    n_obs = length(m.y)
    paths = Matrix{T}(undef, n_draws, h)

    for s in 1:n_draws
        mu = m.mu_post[s]
        phi = m.phi_post[s]
        sigma_eta = m.sigma_eta_post[s]

        # Get last latent state from chain
        h_sym = Symbol("h[$n_obs]")
        h_last = T(m.chain[h_sym].data[s])

        h_prev = h_last
        for t in 1:h
            h_t = mu + phi * (h_prev - mu) + sigma_eta * randn(T)
            paths[s, t] = exp(h_t)
            h_prev = h_t
        end
    end

    _build_volatility_forecast(paths, h, conf_level, :sv)
end
