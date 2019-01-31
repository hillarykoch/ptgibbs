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
Test Summary:            | Pass  Total
unbiased random matrices |   11     11
Test Summary:                         | Pass  Total
0 and 1 constraints correctly imposed |    7      7
Testing ptgibbs tests passed
```


---------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------------


<sub><sup>*This package was originally designed with a Metropolis proposal for the mixing weights in the model (now, it is Gibbs). Since all updates are Gibbs, there is no need for the ''tempered'' portion of the algorithm. A tempered and standard version are nevertheless supported, in case of offering versions with Metropolis proposals in the future.*</sub></sup>
