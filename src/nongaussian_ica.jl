"""
ICA-based SVAR identification: FastICA, JADE, SOBI, dCov, HSIC.

Recovers structural shocks under the assumption that at most one shock is Gaussian.
The unmixing matrix W satisfies ε_t = W u_t where ε_t are independent non-Gaussian shocks.

References:
- Hyvärinen, A. (1999). "Fast and robust fixed-point algorithms for independent component analysis."
- Cardoso, J.-F. & Souloumiac, A. (1993). "Blind beamforming for non-Gaussian signals."
- Belouchrani, A. et al. (1997). "A blind source separation technique using second-order statistics."
- Lanne, M., Meitz, M. & Saikkonen, P. (2017). "Identification and estimation of non-Gaussian SVAR."
- Székely, G. J., Rizzo, M. L. & Bakirov, N. K. (2007). "Measuring and testing dependence by correlation of distances."
- Gretton, A. et al. (2005). "Measuring statistical dependence with Hilbert-Schmidt norms."
"""

using LinearAlgebra, Statistics, Random
import Optim

# =============================================================================
# Result Type
# =============================================================================

"""
    ICASVARResult{T} <: AbstractNonGaussianSVAR

Result from ICA-based SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix (n × n): u_t = B₀ ε_t
- `W::Matrix{T}` — unmixing matrix (n × n): ε_t = W u_t
- `Q::Matrix{T}` — rotation matrix for `compute_Q` integration
- `shocks::Matrix{T}` — recovered structural shocks (T_eff × n)
- `method::Symbol` — `:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`
- `converged::Bool`
- `iterations::Int`
- `objective::T` — final objective value
"""
struct ICASVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    W::Matrix{T}
    Q::Matrix{T}
    shocks::Matrix{T}
    method::Symbol
    converged::Bool
    iterations::Int
    objective::T
end

function Base.show(io::IO, r::ICASVARResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Method"     string(r.method);
        "Variables"  n;
        "Converged"  r.converged ? "Yes" : "No";
        "Iterations" r.iterations;
        "Objective"  _fmt(r.objective)
    ]
    pretty_table(io, spec;
        title = "ICA-SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
        table_format = _TABLE_FORMAT
    )
    _matrix_table(io, r.B0, "Structural Impact Matrix (B₀)";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Shock $j" for j in 1:n])
end

# =============================================================================
# Whitening
# =============================================================================

"""Pre-whiten data via PCA: Z = W_white * U' such that Cov(Z) = I."""
function _whiten(U::Matrix{T}) where {T<:AbstractFloat}
    mu = mean(U, dims=1)
    Uc = U .- mu
    Sigma = Symmetric(Uc' * Uc / size(Uc, 1))
    E = eigen(Sigma)
    idx = sortperm(E.values, rev=true)
    vals = E.values[idx]
    vecs = E.vectors[:, idx]

    # Only keep components with positive eigenvalues
    k = sum(vals .> eps(T) * maximum(vals) * 100)
    D_inv_sqrt = Diagonal(T(1) ./ sqrt.(vals[1:k]))
    W_white = D_inv_sqrt * vecs[:, 1:k]'
    dewhiten = vecs[:, 1:k] * Diagonal(sqrt.(vals[1:k]))

    Z = Matrix{T}((W_white * Uc')')  # T × k
    (Z, Matrix{T}(W_white), Matrix{T}(dewhiten))
end

# =============================================================================
# Givens Rotation Parameterization
# =============================================================================

"""Convert n(n-1)/2 Givens angles to n × n orthogonal matrix."""
function _givens_to_orthogonal(angles::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    Q = Matrix{T}(I, n, n)
    idx = 1
    for i in 1:n-1
        for j in (i+1):n
            c, s = cos(angles[idx]), sin(angles[idx])
            G = Matrix{T}(I, n, n)
            G[i, i], G[j, j] = c, c
            G[i, j], G[j, i] = -s, s
            Q = Q * G
            idx += 1
        end
    end
    Q
end

"""Extract n(n-1)/2 Givens angles from orthogonal matrix (approximate)."""
function _orthogonal_to_givens(Q::AbstractMatrix{T}, n::Int) where {T<:AbstractFloat}
    n_angles = n * (n - 1) ÷ 2
    angles = zeros(T, n_angles)
    R = copy(Q)
    idx = n_angles
    for i in (n-1):-1:1
        for j in n:-1:(i+1)
            angles[idx] = atan(R[j, i], R[i, i])
            c, s = cos(angles[idx]), sin(angles[idx])
            G = Matrix{T}(I, n, n)
            G[i, i], G[j, j] = c, c
            G[i, j], G[j, i] = s, -s
            R = G * R
            idx -= 1
        end
    end
    angles
end

# =============================================================================
# Contrast Functions for FastICA
# =============================================================================

"""Log-cosh contrast: G(u) = log(cosh(u)), g(u) = tanh(u), g'(u) = 1 - tanh²(u)."""
function _contrast_logcosh(u::T) where {T<:AbstractFloat}
    t = tanh(u)
    (log(cosh(u)), t, one(T) - t^2)
end

"""Exponential contrast: G(u) = -exp(-u²/2), g(u) = u·exp(-u²/2), g'(u) = (1-u²)·exp(-u²/2)."""
function _contrast_exp(u::T) where {T<:AbstractFloat}
    e = exp(-u^2 / 2)
    (-e, u * e, (one(T) - u^2) * e)
end

"""Kurtosis contrast: G(u) = u⁴/4, g(u) = u³, g'(u) = 3u²."""
function _contrast_kurtosis(u::T) where {T<:AbstractFloat}
    (u^4 / 4, u^3, 3 * u^2)
end

_get_contrast(sym::Symbol) = sym == :logcosh ? _contrast_logcosh :
                              sym == :exp ? _contrast_exp :
                              sym == :kurtosis ? _contrast_kurtosis :
                              throw(ArgumentError("Unknown contrast: $sym"))

# =============================================================================
# FastICA Algorithms
# =============================================================================

"""FastICA deflation: extract components one at a time."""
function _fastica_deflation(Z::Matrix{T}, n::Int; contrast::Symbol=:logcosh,
                            max_iter::Int=200, tol::T=T(1e-6)) where {T<:AbstractFloat}
    T_obs = size(Z, 1)
    g_func = _get_contrast(contrast)
    W = zeros(T, n, n)
    total_iter = 0

    for p in 1:n
        w = randn(T, n)
        w /= norm(w)

        for iter in 1:max_iter
            total_iter += 1
            # w_new = E[Z g(w'Z)] - E[g'(w'Z)] w
            proj = Z * w  # T_obs × 1
            g_vals = similar(proj)
            gp_vals = similar(proj)
            @inbounds for i in 1:T_obs
                _, g, gp = g_func(proj[i])
                g_vals[i] = g
                gp_vals[i] = gp
            end

            w_new = (Z' * g_vals) / T_obs - mean(gp_vals) * w

            # Orthogonalize against previous components
            for k in 1:(p-1)
                w_new -= dot(w_new, @view(W[k, :])) * @view(W[k, :])
            end
            w_new /= norm(w_new)

            converged = abs(abs(dot(w, w_new)) - one(T)) < tol
            w = w_new
            converged && break
        end
        W[p, :] = w
    end
    W, total_iter
end

"""FastICA symmetric: extract all components simultaneously."""
function _fastica_symmetric(Z::Matrix{T}, n::Int; contrast::Symbol=:logcosh,
                            max_iter::Int=200, tol::T=T(1e-6)) where {T<:AbstractFloat}
    T_obs = size(Z, 1)
    g_func = _get_contrast(contrast)

    # Initialize W with random orthogonal matrix
    W = Matrix{T}(qr(randn(T, n, n)).Q)
    total_iter = 0

    for iter in 1:max_iter
        total_iter = iter
        W_old = copy(W)

        for p in 1:n
            w = @view W[p, :]
            proj = Z * w
            g_vals = similar(proj)
            gp_vals = similar(proj)
            @inbounds for i in 1:T_obs
                _, g, gp = g_func(proj[i])
                g_vals[i] = g
                gp_vals[i] = gp
            end
            W[p, :] = (Z' * g_vals) / T_obs - mean(gp_vals) * w
        end

        # Symmetric decorrelation: W = (W W')^{-1/2} W
        F = eigen(Symmetric(W * W'))
        D_inv_sqrt = Diagonal(1.0 ./ sqrt.(max.(F.values, eps(T))))
        W = F.vectors * D_inv_sqrt * F.vectors' * W

        # Check convergence
        max_change = maximum(abs.(abs.(diag(W * W_old')) .- one(T)))
        max_change < tol && break
    end
    W, total_iter
end

# =============================================================================
# ICA to SVAR Conversion
# =============================================================================

"""Convert ICA unmixing matrix to SVAR representation: B₀, Q, shocks."""
function _ica_to_svar(W_ica::Matrix{T}, model::VARModel{T}) where {T<:AbstractFloat}
    n = nvars(model)
    L = safe_cholesky(model.Sigma)

    # Full unmixing: W_full * u_t = ε_t, so B₀ = W_full⁻¹
    # From whitened: W_ica * W_white * u_t = ε_t
    # W_full = W_ica * W_white (if Z = W_white * U')
    # But we want B₀ = L * Q where Q is orthogonal

    # Compute B₀ = W_full⁻¹
    B0_raw = robust_inv(W_ica)

    # Extract Q: Q = L⁻¹ B₀
    L_inv = robust_inv(Matrix(L))
    Q_raw = L_inv * B0_raw

    # Enforce orthogonality via polar decomposition
    F = svd(Q_raw)
    Q = F.U * F.Vt

    # Recompute B₀ from L and Q for consistency
    B0 = Matrix(L) * Q

    # Structural shocks
    shocks = (robust_inv(B0) * model.U')'

    # Normalize: make diagonal of B₀ positive (sign convention)
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] *= -one(T)
            Q[:, j] *= -one(T)
            shocks[:, j] *= -one(T)
        end
    end

    (B0, Q, shocks)
end

# =============================================================================
# Public API: FastICA
# =============================================================================

"""
    identify_fastica(model::VARModel; contrast=:logcosh, approach=:deflation,
                     max_iter=200, tol=1e-6) -> ICASVARResult

Identify SVAR via FastICA (Hyvärinen 1999).

Recovers independent non-Gaussian structural shocks by maximizing non-Gaussianity
of the recovered sources.

Arguments:
- `contrast` — non-Gaussianity measure: `:logcosh` (default, robust), `:exp`, `:kurtosis`
- `approach` — `:deflation` (one-by-one) or `:symmetric` (simultaneous)
- `max_iter` — maximum iterations per component
- `tol` — convergence tolerance

**Reference**: Hyvärinen (1999)
"""
function identify_fastica(model::VARModel{T}; contrast::Symbol=:logcosh,
                          approach::Symbol=:deflation, max_iter::Int=200,
                          tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    Z, W_white, dewhiten = _whiten(model.U)

    if approach == :deflation
        W_ica, iters = _fastica_deflation(Z, n; contrast=contrast, max_iter=max_iter, tol=tol)
    else
        W_ica, iters = _fastica_symmetric(Z, n; contrast=contrast, max_iter=max_iter, tol=tol)
    end

    # Full unmixing in original space
    W_full = W_ica * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)

    # Compute objective (sum of negentropies)
    g_func = _get_contrast(contrast)
    obj = zero(T)
    for j in 1:n
        s = @view shocks[:, j]
        obj += abs(mean(x -> g_func(x)[1], s) - mean(x -> g_func(x)[1], randn(T, length(s))))
    end

    ICASVARResult{T}(B0, W_full, Q, shocks, :fastica, iters < max_iter * n, iters, obj)
end

# =============================================================================
# JADE: Joint Approximate Diagonalization of Eigenmatrices
# =============================================================================

"""Compute fourth-order cumulant matrices for JADE."""
function _jade_cumulant_matrices(Z::Matrix{T}) where {T<:AbstractFloat}
    T_obs, n = size(Z)
    matrices = Matrix{T}[]

    for i in 1:n, j in i:n
        C = zeros(T, n, n)
        @inbounds for t in 1:T_obs
            z = @view Z[t, :]
            C .+= z[i] * z[j] * (z * z') / T_obs
        end
        # Subtract Gaussian cumulant: δ_{ij} I + e_i e_j' + e_j e_i'
        C[i, j] -= one(T)
        C[j, i] -= one(T)
        for k in 1:n
            C[k, k] -= (i == j ? one(T) : zero(T))
        end
        push!(matrices, C)
    end
    matrices
end

"""Joint diagonalization via Jacobi rotations (Cardoso & Souloumiac 1993)."""
function _joint_diagonalization(matrices::Vector{Matrix{T}}; max_iter::Int=100,
                                tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = size(matrices[1], 1)
    V = Matrix{T}(I, n, n)
    M = [copy(m) for m in matrices]

    total_iter = 0
    for iter in 1:max_iter
        total_iter = iter
        max_rotation = zero(T)

        for p in 1:n-1, q in (p+1):n
            # Compute optimal Givens rotation angle
            h = zeros(T, 3)
            for m in M
                ip, iq = m[p, p] - m[q, q], m[p, q] + m[q, p]
                h[1] += ip * iq
                h[2] += ip^2 - iq^2
                h[3] += 2 * ip * iq  # redundant with h[1] but keeps notation clear
            end

            theta = atan(2 * h[1], h[2]) / 2

            # Apply rotation
            c, s = cos(theta), sin(theta)
            if abs(s) > tol
                max_rotation = max(max_rotation, abs(s))
                for m in M
                    # Rotate rows
                    for k in 1:n
                        a, b = m[p, k], m[q, k]
                        m[p, k] = c * a + s * b
                        m[q, k] = -s * a + c * b
                    end
                    # Rotate columns
                    for k in 1:n
                        a, b = m[k, p], m[k, q]
                        m[k, p] = c * a + s * b
                        m[k, q] = -s * a + c * b
                    end
                end

                # Update V
                for k in 1:n
                    a, b = V[k, p], V[k, q]
                    V[k, p] = c * a + s * b
                    V[k, q] = -s * a + c * b
                end
            end
        end

        max_rotation < tol && break
    end

    V, total_iter
end

"""
    identify_jade(model::VARModel; max_iter=100, tol=1e-6) -> ICASVARResult

Identify SVAR via JADE (Joint Approximate Diagonalization of Eigenmatrices).

Uses fourth-order cumulant matrices and joint diagonalization via Jacobi rotations.

**Reference**: Cardoso & Souloumiac (1993)
"""
function identify_jade(model::VARModel{T}; max_iter::Int=100,
                        tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    Z, W_white, dewhiten = _whiten(model.U)

    cum_matrices = _jade_cumulant_matrices(Z)
    V, iters = _joint_diagonalization(cum_matrices; max_iter=max_iter, tol=tol)

    W_full = V' * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)

    # Objective: sum of squared off-diagonals of rotated cumulant matrices
    obj = zero(T)
    for m in cum_matrices
        Rm = V' * m * V
        for i in 1:n, j in 1:n
            i != j && (obj += Rm[i, j]^2)
        end
    end

    ICASVARResult{T}(B0, W_full, Q, shocks, :jade, iters < max_iter, iters, obj)
end

# =============================================================================
# SOBI: Second-Order Blind Identification
# =============================================================================

"""Compute autocovariance matrix at lag τ."""
function _sobi_autocovariance(Z::Matrix{T}, lag::Int) where {T<:AbstractFloat}
    T_obs, n = size(Z)
    R = zeros(T, n, n)
    T_eff = T_obs - lag
    @inbounds for t in 1:T_eff
        z_t = @view Z[t + lag, :]
        z_0 = @view Z[t, :]
        R .+= z_t * z_0'
    end
    R / T_eff
end

"""
    identify_sobi(model::VARModel; lags=1:12, max_iter=100, tol=1e-6) -> ICASVARResult

Identify SVAR via SOBI (Second-Order Blind Identification).

Uses autocovariance matrices at multiple lags and joint diagonalization.
Exploits temporal structure rather than higher-order statistics.

**Reference**: Belouchrani et al. (1997)
"""
function identify_sobi(model::VARModel{T}; lags::AbstractRange=1:12,
                        max_iter::Int=100, tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    Z, W_white, dewhiten = _whiten(model.U)

    auto_covs = [_sobi_autocovariance(Z, lag) for lag in lags]
    V, iters = _joint_diagonalization(auto_covs; max_iter=max_iter, tol=tol)

    W_full = V' * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)

    # Objective: sum of squared off-diagonals
    obj = zero(T)
    for m in auto_covs
        Rm = V' * m * V
        for i in 1:n, j in 1:n
            i != j && (obj += Rm[i, j]^2)
        end
    end

    ICASVARResult{T}(B0, W_full, Q, shocks, :sobi, iters < max_iter, iters, obj)
end

# =============================================================================
# Distance Covariance
# =============================================================================

"""Compute distance covariance between two vectors."""
function _distance_covariance(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)

    # Compute distance matrices
    A = [abs(x[i] - x[j]) for i in 1:n, j in 1:n]
    B = [abs(y[i] - y[j]) for i in 1:n, j in 1:n]

    # Double-center
    A_row = mean(A, dims=2)
    A_col = mean(A, dims=1)
    A_grand = mean(A)
    B_row = mean(B, dims=2)
    B_col = mean(B, dims=1)
    B_grand = mean(B)

    @inbounds for i in 1:n, j in 1:n
        A[i, j] = A[i, j] - A_row[i] - A_col[j] + A_grand
        B[i, j] = B[i, j] - B_row[i] - B_col[j] + B_grand
    end

    # dCov² = (1/n²) Σᵢⱼ Aᵢⱼ Bᵢⱼ
    dcov2 = sum(A .* B) / n^2
    max(dcov2, zero(T))
end

"""Objective: sum of pairwise distance covariances (to minimize)."""
function _dcov_objective(angles::AbstractVector{T}, Z::Matrix{T}, n::Int) where {T<:AbstractFloat}
    Q = _givens_to_orthogonal(angles, n)
    S = Z * Q'  # rotated sources

    obj = zero(T)
    for i in 1:n-1, j in (i+1):n
        obj += _distance_covariance(@view(S[:, i]), @view(S[:, j]))
    end
    obj
end

"""
    identify_dcov(model::VARModel; max_iter=200, tol=1e-6) -> ICASVARResult

Identify SVAR by minimizing pairwise distance covariance between recovered shocks.

Distance covariance (Székely et al. 2007) is zero iff the variables are independent,
making it a natural criterion for ICA.

**Reference**: Matteson & Tsay (2017)
"""
function identify_dcov(model::VARModel{T}; max_iter::Int=200,
                        tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    Z, W_white, dewhiten = _whiten(model.U)

    n_angles = n * (n - 1) ÷ 2
    angles0 = zeros(T, n_angles)

    result = Optim.optimize(a -> _dcov_objective(a, Z, n), angles0,
                            Optim.NelderMead(),
                            Optim.Options(iterations=max_iter, g_tol=tol))

    Q_opt = _givens_to_orthogonal(Optim.minimizer(result), n)
    W_full = Q_opt * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)

    ICASVARResult{T}(B0, W_full, Q, shocks, :dcov, Optim.converged(result),
                     Optim.iterations(result), T(Optim.minimum(result)))
end

# =============================================================================
# HSIC (Hilbert-Schmidt Independence Criterion)
# =============================================================================

"""Compute HSIC statistic between two vectors using Gaussian kernel."""
function _hsic_statistic(x::AbstractVector{T}, y::AbstractVector{T};
                          sigma::T=T(1.0)) where {T<:AbstractFloat}
    n = length(x)
    s2 = 2 * sigma^2

    # Gaussian kernel matrices
    K = [exp(-(x[i] - x[j])^2 / s2) for i in 1:n, j in 1:n]
    L = [exp(-(y[i] - y[j])^2 / s2) for i in 1:n, j in 1:n]

    # Center kernel matrices
    H = I - ones(T, n, n) / n
    Kc = H * K * H
    Lc = H * L * H

    tr(Kc * Lc) / (n - 1)^2
end

"""Objective: sum of pairwise HSIC (to minimize)."""
function _hsic_objective(angles::AbstractVector{T}, Z::Matrix{T}, n::Int;
                          sigma::T=T(1.0)) where {T<:AbstractFloat}
    Q = _givens_to_orthogonal(angles, n)
    S = Z * Q'

    obj = zero(T)
    for i in 1:n-1, j in (i+1):n
        obj += _hsic_statistic(@view(S[:, i]), @view(S[:, j]); sigma=sigma)
    end
    obj
end

"""
    identify_hsic(model::VARModel; kernel=:gaussian, sigma=1.0,
                  max_iter=200, tol=1e-6) -> ICASVARResult

Identify SVAR by minimizing pairwise HSIC between recovered shocks.

HSIC with a characteristic kernel (Gaussian) is zero iff variables are independent.

**Reference**: Gretton et al. (2005)
"""
function identify_hsic(model::VARModel{T}; kernel::Symbol=:gaussian,
                        sigma::T=T(1.0), max_iter::Int=200,
                        tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    Z, W_white, dewhiten = _whiten(model.U)

    # Use median heuristic for bandwidth if not specified
    if sigma == one(T)
        dists = T[]
        T_obs = size(Z, 1)
        n_sample = min(T_obs, 200)
        idx = randperm(T_obs)[1:n_sample]
        for i in idx, j in idx
            i < j && push!(dists, norm(@view(Z[i, :]) .- @view(Z[j, :])))
        end
        sigma = max(median(dists), eps(T))
    end

    n_angles = n * (n - 1) ÷ 2
    angles0 = zeros(T, n_angles)

    result = Optim.optimize(a -> _hsic_objective(a, Z, n; sigma=sigma), angles0,
                            Optim.NelderMead(),
                            Optim.Options(iterations=max_iter, g_tol=tol))

    Q_opt = _givens_to_orthogonal(Optim.minimizer(result), n)
    W_full = Q_opt * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)

    ICASVARResult{T}(B0, W_full, Q, shocks, :hsic, Optim.converged(result),
                     Optim.iterations(result), T(Optim.minimum(result)))
end
