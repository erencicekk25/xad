# Black-Scholes Greeks Sample

Prices a European call and put with the Black-Scholes formula and computes
their Greeks with XAD, once in forward mode and once in adjoint mode. Both
results are checked against the closed-form Greeks.

The pricer is written once, as a template on the number type, so the same code
runs with `double`, `xad::FReal<double>` and `xad::AReal<double>`.

## Formula

    d1   = (ln(S / K) + (r + sigma^2 / 2) T) / (sigma sqrt(T))
    d2   = d1 - sigma sqrt(T)
    call = S N(d1) - K exp(-r T) N(d2)
    put  = K exp(-r T) N(-d2) - S N(-d1)

with spot `S`, strike `K`, volatility `sigma`, rate `r`, time to maturity `T`
and the standard normal CDF `N`.

| Greek | Derivative                                    |
| ----- | --------------------------------------------- |
| delta | dV/dS                                         |
| vega  | dV/dsigma                                     |
| rho   | dV/dr                                         |
| theta | -dV/dT (value change as time passes)          |

## Forward and adjoint mode

- **Forward mode** (`FReal`) seeds one input per pass. A pass gives the
  sensitivity of both prices to that input, so four passes give all Greeks.
- **Adjoint mode** (`AReal`) records the pricing once. One reverse sweep per
  price gives that price's sensitivities to all four inputs at once.

With many inputs and few outputs, as is typical for pricing, adjoint mode
needs far fewer passes.

## Output

    Call
                     closed form     forward (FReal)     adjoint (AReal)
       price      8.021352235143      8.021352235143      8.021352235143
       delta      0.542228333585      0.542228333585      0.542228333585
        vega     39.670523808427     39.670523808427     39.670523808427
         rho     46.201481123337     46.201481123337     46.201481123337
       theta     -6.277126437010     -6.277126437010     -6.277126437010

The program exits with a non-zero status if any value differs from the closed
form by more than `1e-12` (relative, or absolute for values below 1).
