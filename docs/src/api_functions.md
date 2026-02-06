# [API Functions](@id api_functions)

This page documents all functions in **MacroEconometricModels.jl**, organized by module.

---

## ARIMA Models

### Estimation

```@docs
estimate_ar
estimate_ma
estimate_arma
estimate_arima
```

### Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["arima_forecast.jl"]
Order   = [:function]
```

### Order Selection

```@docs
select_arima_order
auto_arima
ic_table
```

---

## VAR Estimation

### Frequentist Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["estimation.jl"]
Order   = [:function]
```

### Bayesian Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["bayesian.jl"]
Order   = [:function]
```

### Prior Specification

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["priors.jl"]
Order   = [:function]
```

---

## Structural Identification

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["identification.jl"]
Order   = [:function]
```

---

## Innovation Accounting

### Impulse Response Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["irf.jl"]
Order   = [:function]
```

### Forecast Error Variance Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["fevd.jl"]
Order   = [:function]
```

### Historical Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["hd.jl"]
Order   = [:function]
```

### Summary Tables

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["summary.jl"]
Order   = [:function]
```

---

## Local Projections

### Core LP Estimation and Covariance

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp_core.jl"]
Order   = [:function]
```

### LP Extensions (IV, Smooth, State-Dependent, Propensity)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp_extensions.jl"]
Order   = [:function]
```

---

## Factor Models

### Static Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["staticfactor.jl"]
Order   = [:function]
```

### Dynamic Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["dynamicfactor.jl"]
Order   = [:function]
```

### Generalized Dynamic Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["generalizedfactor.jl"]
Order   = [:function]
```

---

## GMM Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["gmm.jl"]
Order   = [:function]
```

---

## Unit Root and Cointegration Tests

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["unitroot.jl"]
Order   = [:function]
```

---

## Non-Gaussian SVAR Identification

### Normality Tests

```@docs
jarque_bera_test
mardia_test
doornik_hansen_test
henze_zirkler_test
normality_test_suite
```

### ICA-based Identification

```@docs
identify_fastica
identify_jade
identify_sobi
identify_dcov
identify_hsic
```

### Non-Gaussian ML Identification

```@docs
identify_student_t
identify_mixture_normal
identify_pml
identify_skew_normal
identify_nongaussian_ml
```

### Heteroskedasticity Identification

```@docs
identify_markov_switching
identify_garch
identify_smooth_transition
identify_external_volatility
```

### Identifiability Tests

```@docs
test_identification_strength
test_shock_gaussianity
test_gaussian_vs_nongaussian
test_shock_independence
test_overidentification
```

---

## Utility Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["utils.jl"]
Order   = [:function]
```
