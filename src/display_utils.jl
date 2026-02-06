"""
Shared PrettyTables formatting utilities for publication-quality display.

Provides a unified borderless table format, backend switching (text/LaTeX/HTML),
and common formatting helpers used across all show methods in the package.
"""

using PrettyTables

# =============================================================================
# Display Backend Configuration
# =============================================================================

# Global display backend (:text, :latex, :html)
const _DISPLAY_BACKEND = Ref{Symbol}(:text)

"""
    set_display_backend(backend::Symbol)

Set the PrettyTables output backend. Options: `:text` (default), `:latex`, `:html`.

# Examples
```julia
set_display_backend(:latex)   # all show() methods now emit LaTeX
set_display_backend(:html)    # switch to HTML tables
set_display_backend(:text)    # back to terminal-friendly text
```
"""
function set_display_backend(backend::Symbol)
    backend ∈ (:text, :latex, :html) || throw(ArgumentError("backend must be :text, :latex, or :html, got :$backend"))
    _DISPLAY_BACKEND[] = backend
end

"""
    get_display_backend() -> Symbol

Return the current PrettyTables display backend (`:text`, `:latex`, or `:html`).
"""
get_display_backend() = _DISPLAY_BACKEND[]

# Shared borderless text table format (Stata-style)
const _TEXT_TABLE_FORMAT = TextTableFormat(
    borders = text_table_borders__borderless,
    horizontal_line_after_column_labels = true
)

"""
    _pretty_table(io::IO, data; kwargs...)

Central PrettyTables wrapper that respects the global display backend.

For `:text` backend, applies `_TEXT_TABLE_FORMAT` automatically.
For `:latex` and `:html` backends, omits text-only formatting options.
"""
function _pretty_table(io::IO, data; kwargs...)
    be = _DISPLAY_BACKEND[]
    if be == :text
        pretty_table(io, data; backend = :text, table_format = _TEXT_TABLE_FORMAT, kwargs...)
    elseif be == :latex
        pretty_table(io, data; backend = :latex, kwargs...)
    else
        pretty_table(io, data; backend = :html, kwargs...)
    end
end

# =============================================================================
# Formatting Helpers
# =============================================================================

_fmt(x::Real; digits::Int=4) = round(x, digits=digits)
_fmt_pct(x::Real; digits::Int=1) = string(round(x * 100, digits=digits), "%")

function _format_pvalue(pval::Real)
    pval < 0.001 && return "<0.001"
    pval > 0.999 && return ">0.999"
    return string(round(pval, digits=4))
end

function _significance_stars(pvalue::Real)
    pvalue < 0.01 && return "***"
    pvalue < 0.05 && return "**"
    pvalue < 0.10 && return "*"
    return ""
end

"""Select representative horizons for display."""
function _select_horizons(H::Int)
    H <= 5 && return collect(1:H)
    H <= 12 && return [1, 4, 8, H]
    H <= 24 && return [1, 4, 8, 12, H]
    return [1, 4, 8, 12, 24, H]
end

"""Print a labeled matrix as a PrettyTables table."""
function _matrix_table(io::IO, M::AbstractMatrix, title::String;
                       row_labels=nothing, col_labels=nothing, digits::Int=4)
    n, m = size(M)
    row_labels = something(row_labels, ["[$i]" for i in 1:n])
    col_labels = something(col_labels, ["[$j]" for j in 1:m])
    data = Matrix{Any}(undef, n, m + 1)
    for i in 1:n
        data[i, 1] = row_labels[i]
        for j in 1:m
            data[i, j+1] = round(M[i, j], digits=digits)
        end
    end
    _pretty_table(io, data;
        title = title,
        column_labels = vcat([""], col_labels),
        alignment = vcat([:l], fill(:r, m))
    )
end
