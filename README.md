# MacroEconometricModels.jl

[![CI](https://github.com/chung9207/MacroEconometricModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/chung9207/MacroEconometricModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/chung9207/MacroEconometricModels.jl/graph/badge.svg)](https://codecov.io/gh/chung9207/MacroEconometricModels.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://chung9207.github.io/MacroEconometricModels.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18439170.svg)](https://doi.org/10.5281/zenodo.18439170)

A Julia package for macroeconomic time series analysis.

## Features

### Estimation
- **Vector Autoregression (VAR)** - OLS estimation with lag order selection (AIC, BIC, HQ)
- **Bayesian VAR (BVAR)** - Minnesota priors with hyperparameter optimization (Giannone, Lenza & Primiceri 2015)
- **Local Projections (LP)** - Jordà (2005) with extensions:
  - HAC standard errors (Newey-West, White, Driscoll-Kraay)
  - Instrumental Variables (Stock & Watson 2018)
  - Smooth IRF via B-splines (Barnichon & Brownlees 2019)
  - State-dependent LP (Auerbach & Gorodnichenko 2013)
  - Propensity Score Matching (Angrist et al. 2018)
- **Factor Models** - Static, Dynamic, and Generalized Dynamic Factor Models
- **Generalized Method of Moments (GMM)**

### Structural Identification
- Cholesky decomposition (recursive)
- Sign restrictions (Rubio-Ramírez, Waggoner & Zha 2010)
- Narrative restrictions (Antolín-Díaz & Rubio-Ramírez 2018)
- Long-run restrictions (Blanchard & Quah 1989)

### Innovation Accounting
- **Impulse Response Functions (IRF)** - Bootstrap and Bayesian credible intervals
- **Forecast Error Variance Decomposition (FEVD)**
- **Historical Decomposition (HD)** - Decompose observed movements into shock contributions
- **Summary Tables** - Publication-quality output with `summary()`, `table()`, `print_table()`

### Hypothesis Tests
- **Unit Root Tests** - ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron
- **Cointegration** - Johansen test (trace and max-eigenvalue)

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricModels")
```

## Quick Start

```julia
using MacroEconometricModels
using Random

# Generate synthetic VAR(1) data
Random.seed!(42)
T, n = 200, 3
A = [0.5 0.1 0.0; 0.0 0.6 0.1; 0.1 0.0 0.4]
Y = zeros(T, n)
for t in 2:T
    Y[t, :] = A * Y[t-1, :] + randn(n)
end

# Estimate VAR model
model = estimate_var(Y, 2)

# Innovation accounting
irf_result = irf(model, 20; ci_type=:bootstrap)
fevd_result = fevd(model, 20)
hd_result = historical_decomposition(model, 198)

# Publication-quality tables
df = table(irf_result, 1, 1; horizons=[1, 4, 8, 12, 20])
print_table(stdout, fevd_result, 1)

# Unit root tests
adf_test(Y[:, 1]; lags=:aic)
kpss_test(Y[:, 1])

# Bayesian estimation
chain = estimate_bvar(Y, 2; prior=:minnesota)

# Local Projections
lp_result = estimate_lp(Y, 1, 20; cov_type=:newey_west)
```

## Documentation

Full documentation available at [https://chung9207.github.io/MacroEconometricModels.jl/dev/](https://chung9207.github.io/MacroEconometricModels.jl/dev/)

## References

- Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative Sign Restrictions for SVARs. *American Economic Review*.
- Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto regressions. *Journal of Applied Econometrics*.
- Blanchard, O. J., & Quah, D. (1989). The Dynamic Effects of Aggregate Demand and Supply Disturbances. *American Economic Review*.
- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior Selection for Vector Autoregressions. *Review of Economics and Statistics*.
- Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections. *American Economic Review*.
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural vector autoregressions. *Review of Economic Studies*.
- Stock, J. H., & Watson, M. W. (2018). Identification and Estimation of Dynamic Causal Effects. *The Economic Journal*.

## License

MIT
