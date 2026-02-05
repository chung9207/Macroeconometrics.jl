# API Reference

This section provides the complete API documentation for **MacroEconometricModels.jl**.

The API documentation is organized into the following pages:

- **[Types](@ref api_types)**: Core type definitions for models, results, and estimators
- **[Functions](@ref api_functions)**: Function documentation organized by module

## Quick Reference Tables

### ARIMA Estimation Functions

| Function | Description |
|----------|-------------|
| `estimate_ar(y, p; method=:ols)` | AR(p) via OLS or MLE |
| `estimate_ma(y, q; method=:css_mle)` | MA(q) via CSS, MLE, or CSS-MLE |
| `estimate_arma(y, p, q; method=:css_mle)` | ARMA(p,q) via CSS, MLE, or CSS-MLE |
| `estimate_arima(y, p, d, q; method=:css_mle)` | ARIMA(p,d,q) via differencing + ARMA |
| `forecast(model, h; conf_level=0.95)` | Multi-step forecasting with confidence intervals |
| `select_arima_order(y, max_p, max_q)` | Grid search for optimal ARMA order |
| `auto_arima(y)` | Automatic ARIMA order selection |
| `ic_table(y, max_p, max_q)` | Information criteria comparison table |

### Multivariate Estimation Functions

| Function | Description |
|----------|-------------|
| `estimate_var(Y, p)` | Estimate VAR(p) via OLS |
| `estimate_bvar(Y, p; ...)` | Estimate Bayesian VAR with MCMC |
| `estimate_lp(Y, shock_var, H; ...)` | Standard Local Projection |
| `estimate_lp_iv(Y, shock_var, Z, H; ...)` | LP with instrumental variables |
| `estimate_smooth_lp(Y, shock_var, H; ...)` | Smooth LP with B-splines |
| `estimate_state_lp(Y, shock_var, state_var, H; ...)` | State-dependent LP |
| `estimate_propensity_lp(Y, treatment, covariates, H; ...)` | LP with propensity scores |
| `doubly_robust_lp(Y, treatment, covariates, H; ...)` | Doubly robust LP estimator |
| `estimate_factors(X, r; ...)` | Static factor model via PCA |
| `estimate_dynamic_factors(X, r, p; ...)` | Dynamic factor model |
| `estimate_gdfm(X, q; ...)` | Generalized dynamic factor model |
| `estimate_gmm(moment_fn, theta0, data; ...)` | GMM estimation |

### Structural Analysis Functions

| Function | Description |
|----------|-------------|
| `irf(model, H; ...)` | Compute impulse response functions |
| `fevd(model, H; ...)` | Forecast error variance decomposition |
| `identify_cholesky(model)` | Cholesky identification |
| `identify_sign(model; ...)` | Sign restriction identification |
| `identify_long_run(model)` | Blanchard-Quah identification |
| `identify_narrative(model; ...)` | Narrative sign restrictions |

### Unit Root Test Functions

| Function | Description |
|----------|-------------|
| `adf_test(y; ...)` | Augmented Dickey-Fuller unit root test |
| `kpss_test(y; ...)` | KPSS stationarity test |
| `pp_test(y; ...)` | Phillips-Perron unit root test |
| `za_test(y; ...)` | Zivot-Andrews structural break test |
| `ngperron_test(y; ...)` | Ng-Perron unit root tests (MZα, MZt, MSB, MPT) |
| `johansen_test(Y, p; ...)` | Johansen cointegration test |
| `is_stationary(model)` | Check VAR model stationarity |
| `unit_root_summary(y; ...)` | Run multiple tests with summary |
| `test_all_variables(Y; ...)` | Apply test to all columns |

### LP IRF Extraction

| Function | Description |
|----------|-------------|
| `lp_irf(model; ...)` | Extract IRF from LPModel |
| `lp_iv_irf(model; ...)` | Extract IRF from LPIVModel |
| `smooth_lp_irf(model; ...)` | Extract smoothed IRF |
| `state_irf(model; ...)` | Extract state-dependent IRFs |
| `propensity_irf(model; ...)` | Extract ATE impulse response |

### Factor Model Functions

| Function | Description |
|----------|-------------|
| `estimate_factors(X, r; ...)` | Estimate r-factor model |
| `ic_criteria(X, r_max)` | Bai-Ng information criteria |
| `scree_plot_data(model)` | Data for scree plot |

### Diagnostic Functions

| Function | Description |
|----------|-------------|
| `optimize_hyperparameters(Y, p; ...)` | Optimize Minnesota prior |
| `weak_instrument_test(model; ...)` | Test for weak instruments |
| `sargan_test(model, h)` | Overidentification test |
| `test_regime_difference(model; ...)` | Test regime differences |
| `propensity_diagnostics(model)` | Propensity score diagnostics |
| `j_test(model)` | Hansen J-test for GMM |
| `gmm_summary(model)` | Summary statistics for GMM |

### Covariance Functions

| Function | Description |
|----------|-------------|
| `newey_west(X, residuals; ...)` | Newey-West HAC estimator |
| `white_vcov(X, residuals; ...)` | White heteroskedasticity-robust |
| `driscoll_kraay(X, residuals; ...)` | Driscoll-Kraay panel-robust |
| `long_run_variance(x; ...)` | Long-run variance estimate |
| `long_run_covariance(X; ...)` | Long-run covariance matrix |
| `optimal_bandwidth_nw(residuals)` | Automatic bandwidth selection |

### Utility Functions

| Function | Description |
|----------|-------------|
| `construct_var_matrices(Y, p)` | Build VAR design matrices |
| `companion_matrix(B, n, p)` | VAR companion form |
| `robust_inv(A)` | Robust matrix inverse |
| `safe_cholesky(A; ...)` | Stable Cholesky decomposition |
