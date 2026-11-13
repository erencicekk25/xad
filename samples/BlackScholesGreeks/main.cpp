/*******************************************************************************

   Prices a European call and put with the Black-Scholes formula and computes
   their Greeks (delta, vega, rho and theta) with XAD in two ways:

     - forward mode (FReal): one pass per input; each pass gives the
       sensitivities of both prices to that input;
     - adjoint mode (AReal): one recording, then one reverse sweep per price
       gives that price's sensitivities to all inputs at once.

   Both are checked against the closed-form Greeks.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

*******************************************************************************/

#include <XAD/XAD.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

// Standard normal cumulative distribution function. Callers with an XAD type pass
// T explicitly, so that an expression such as -d2 converts to T instead of being
// deduced as an expression-template type.
template <typename T>
T normalCdf(const T& x)
{
    using std::erfc;
    using std::sqrt;
    return 0.5 * erfc(-x / sqrt(2.0));
}

// Prices of a European call and put on the same underlying.
template <typename T>
struct OptionPrices
{
    T call;
    T put;
};

// The Black-Scholes formula, written once for any number type: double, FReal or AReal.
// The strike is a contract term, not a market input, so it stays a passive double.
template <typename T>
OptionPrices<T> blackScholes(const T& spot, const T& vol, const T& rate, const T& maturity,
                             double strike)
{
    using std::exp;
    using std::log;
    using std::sqrt;

    T stdDev = vol * sqrt(maturity);
    T d1 = (log(spot / strike) + (rate + 0.5 * vol * vol) * maturity) / stdDev;
    T d2 = d1 - stdDev;
    T discountedStrike = strike * exp(-rate * maturity);

    OptionPrices<T> prices;
    prices.call = spot * normalCdf<T>(d1) - discountedStrike * normalCdf<T>(d2);
    prices.put = discountedStrike * normalCdf<T>(-d2) - spot * normalCdf<T>(-d1);
    return prices;
}

// Market data and contract.
struct Inputs
{
    double spot;      // S
    double vol;       // sigma
    double rate;      // r, continuously compounded
    double maturity;  // T, in years
    double strike;    // K
};

// A price and its sensitivities. Theta is the change in value as time passes,
// i.e. minus the derivative with respect to the time to maturity.
struct Greeks
{
    double price;
    double delta;
    double vega;
    double rho;
    double theta;
};

// Closed-form Black-Scholes Greeks, used as the reference.
void closedForm(const Inputs& in, Greeks& call, Greeks& put)
{
    const double pi = 3.14159265358979323846;
    double sqrtT = std::sqrt(in.maturity);
    double d1 = (std::log(in.spot / in.strike) + (in.rate + 0.5 * in.vol * in.vol) * in.maturity) /
                (in.vol * sqrtT);
    double d2 = d1 - in.vol * sqrtT;
    double pdf = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * pi);
    double df = std::exp(-in.rate * in.maturity);
    double timeDecay = -in.spot * pdf * in.vol / (2.0 * sqrtT);

    call.price = in.spot * normalCdf(d1) - in.strike * df * normalCdf(d2);
    call.delta = normalCdf(d1);
    call.vega = in.spot * pdf * sqrtT;
    call.rho = in.strike * in.maturity * df * normalCdf(d2);
    call.theta = timeDecay - in.rate * in.strike * df * normalCdf(d2);

    put.price = in.strike * df * normalCdf(-d2) - in.spot * normalCdf(-d1);
    put.delta = normalCdf(d1) - 1.0;
    put.vega = call.vega;
    put.rho = -in.strike * in.maturity * df * normalCdf(-d2);
    put.theta = timeDecay + in.rate * in.strike * df * normalCdf(-d2);
}

// Forward mode: seed one input per pass. A pass gives d(call)/d(input) and
// d(put)/d(input) together, so four passes cover the four Greeks.
void forwardMode(const Inputs& in, Greeks& call, Greeks& put)
{
    typedef xad::FReal<double> AD;

    double* callGreeks[] = {&call.delta, &call.vega, &call.rho, &call.theta};
    double* putGreeks[] = {&put.delta, &put.vega, &put.rho, &put.theta};

    for (std::size_t input = 0; input < 4; ++input)
    {
        AD spot = in.spot, vol = in.vol, rate = in.rate, maturity = in.maturity;
        AD* seeded[] = {&spot, &vol, &rate, &maturity};
        derivative(*seeded[input]) = 1.0;

        OptionPrices<AD> prices = blackScholes(spot, vol, rate, maturity, in.strike);

        *callGreeks[input] = derivative(prices.call);
        *putGreeks[input] = derivative(prices.put);
        call.price = value(prices.call);
        put.price = value(prices.put);
    }

    // theta = -dV/dT
    call.theta = -call.theta;
    put.theta = -put.theta;
}

// Adjoint mode: record the pricing once, then one reverse sweep per price
// gives that price's sensitivities to all four inputs.
void adjointMode(const Inputs& in, Greeks& call, Greeks& put)
{
    typedef xad::AReal<double> AD;
    xad::Tape<double> tape;

    AD spot = in.spot, vol = in.vol, rate = in.rate, maturity = in.maturity;
    tape.registerInput(spot);
    tape.registerInput(vol);
    tape.registerInput(rate);
    tape.registerInput(maturity);
    tape.newRecording();

    OptionPrices<AD> prices = blackScholes(spot, vol, rate, maturity, in.strike);
    tape.registerOutput(prices.call);
    tape.registerOutput(prices.put);

    Greeks* results[] = {&call, &put};
    AD* outputs[] = {&prices.call, &prices.put};
    for (std::size_t i = 0; i < 2; ++i)
    {
        tape.clearDerivatives();
        derivative(*outputs[i]) = 1.0;
        tape.computeAdjoints();

        results[i]->price = value(*outputs[i]);
        results[i]->delta = derivative(spot);
        results[i]->vega = derivative(vol);
        results[i]->rho = derivative(rate);
        results[i]->theta = -derivative(maturity);
    }
}

bool matches(double actual, double expected)
{
    const double tolerance = 1e-12;
    return std::fabs(actual - expected) <= tolerance * std::max(1.0, std::fabs(expected));
}

// Prints one row per value and returns the number of mismatches.
int report(const char* name, const Greeks& reference, const Greeks& fwd, const Greeks& adj)
{
    const char* labels[] = {"price", "delta", "vega", "rho", "theta"};
    double Greeks::*fields[] = {&Greeks::price, &Greeks::delta, &Greeks::vega, &Greeks::rho,
                                &Greeks::theta};
    int failures = 0;

    std::cout << "\n"
              << name << "\n"
              << std::setw(8) << "" << std::setw(20) << "closed form" << std::setw(20)
              << "forward (FReal)" << std::setw(20) << "adjoint (AReal)"
              << "\n";
    for (std::size_t i = 0; i < 5; ++i)
    {
        double ref = reference.*fields[i];
        double f = fwd.*fields[i];
        double a = adj.*fields[i];
        bool ok = matches(f, ref) && matches(a, ref);
        if (!ok)
            ++failures;
        std::cout << std::setw(8) << labels[i] << std::setw(20) << ref << std::setw(20) << f
                  << std::setw(20) << a << (ok ? "" : "   <-- mismatch") << "\n";
    }
    return failures;
}

int main()
{
    Inputs in;
    in.spot = 100.0;
    in.vol = 0.2;
    in.rate = 0.05;
    in.maturity = 1.0;
    in.strike = 105.0;

    Greeks refCall, refPut, fwdCall, fwdPut, adjCall, adjPut;
    closedForm(in, refCall, refPut);
    forwardMode(in, fwdCall, fwdPut);
    adjointMode(in, adjCall, adjPut);

    std::cout << "Black-Scholes with S = " << in.spot << ", K = " << in.strike
              << ", sigma = " << in.vol << ", r = " << in.rate << ", T = " << in.maturity << "\n"
              << std::fixed << std::setprecision(12);

    int failures = report("Call", refCall, fwdCall, adjCall);
    failures += report("Put", refPut, fwdPut, adjPut);

    if (failures != 0)
    {
        std::cout << "\n"
                  << failures << " value(s) differ from the closed form by more than 1e-12\n";
        return 1;
    }
    std::cout << "\nForward and adjoint Greeks match the closed form to 1e-12.\n";
    return 0;
}
