using Macroeconometrics
using Documenter

DocMeta.setdocmeta!(Macroeconometrics, :DocTestSetup, :(using Macroeconometrics); recursive=true)

makedocs(;
    modules=[Macroeconometrics],
    authors="Wookyung Chung <mirimtl@protonmail.com>",
    repo="https://github.com/chung9207/Macroeconometrics.jl/blob/{commit}{path}#{line}",
    sitename="Macroeconometrics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chung9207.github.io/Macroeconometrics.jl",
        edit_link="main",
        assets=String[],
        mathengine=Documenter.MathJax3(),
    ),
    pages=[
        "Home" => "index.md",
        "Theory" => [
            "VAR & BVAR" => "manual.md",
            "Local Projections" => "lp.md",
            "Factor Models" => "factormodels.md",
        ],
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    checkdocs=:exports,
    warnonly=[:missing_docs, :cross_references, :autodocs_block, :docs_block],
)

deploydocs(;
    repo="github.com/chung9207/Macroeconometrics.jl",
    devbranch="main",
)
