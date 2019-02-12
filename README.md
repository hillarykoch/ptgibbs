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

Running the MCMC... 100%|███████████████████████████|

Test Summary:                  | Pass  Total
reasonable parameter estimates |   19     19
Test Summary:            | Pass  Total
unbiased random matrices |   11     11
Test Summary:                         | Pass  Total
0 and 1 constraints correctly imposed |    7      7
Testing ptgibbs tests passed
```


---------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------------


<sub><sup>*This package was originally designed to support parallel tempering, as there is a Metropolis proposal in the algorithm. Currently, this feature is no longer supported, though it may return in the future.*</sub></sup>
