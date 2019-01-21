# ptgibbs.jl

Companion Julia module for pGMCM R package

**NOTE: this package was developed in Julia v1.0.2, and likely is not back-compatible with older versions**

## Installation

To install, launch Julia, then enter the following:

```{julia}
using Pkg
Pkg.add(PackageSpec(url="https://github.com/hillarykoch/ptgibbs.jl"))
```

Load and test the package by then executing the following from inside Julia (optional):
```{julia}
using ptgibbs
Pkg.test("ptgibbs")
```
If things went as they should, you should see something akin to this:
```console
Testing ptgibbs

Computing for burn-in...100%|███████████████████████████|
Computing for main Markov chain...100%|█████████████████|

Test Summary:                  | Pass  Total
reasonable parameter estimates |   10     10
Testing ptgibbs tests passed
```
