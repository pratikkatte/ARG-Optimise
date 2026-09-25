#!/usr/bin/env python3
"""Exact rational two-locus posterior moments from a separate 3-state system.

The main reference solver uses 13 labeled physical states and mutation marks.
Here the joint prior TMRCA Laplace transform uses only three unlabelled states,
and derivative linear equations yield exact rational answers independently.
"""
from fractions import Fraction as F
from functools import lru_cache
import json
import math
from pathlib import Path


def solve(matrix,rhs):
    table=[list(row)+[value] for row,value in zip(matrix,rhs)]
    for i in range(3):
        pivot=table[i][i]
        table[i]=[x/pivot for x in table[i]]
        for j in range(3):
            if j!=i:
                multiple=table[j][i]
                table[j]=[x-multiple*y for x,y in zip(table[j],table[i])]
    return tuple(row[-1] for row in table)


def derive():
    kappa=c=F(1,2);a=b=2*kappa
    matrix=((1+2*c+a+b,-2*c,F(0)),(-F(1),3+c+a+b,-c),(F(0),-F(4),6+a+b))
    @lru_cache(None)
    def derivative(m,n):
        if m==n==0:
            h=1/(1+a)+1/(1+b)
            rhs=[F(1),h,h]
        else:
            h=(F((-1)**m*math.factorial(m),(1+a)**(m+1)) if n==0 else
               F((-1)**n*math.factorial(n),(1+b)**(n+1)) if m==0 else F(0))
            rhs=[F(0),h,h]
        if m:
            rhs=[x-m*y for x,y in zip(rhs,derivative(m-1,n))]
        if n:
            rhs=[x-n*y for x,y in zip(rhs,derivative(m,n-1))]
        return solve(matrix,rhs)
    mixed=derivative(1,1)[0]
    z=kappa*kappa*mixed
    mean=-derivative(2,1)[0]/mixed
    second=derivative(3,1)[0]/mixed
    cross=derivative(2,2)[0]/mixed
    zero_recombination=2*kappa*kappa/(1+2*c+4*kappa)**3/z
    values=dict(evidence=z,tmrca_mean=mean,tmrca_variance=second-mean*mean,
                tmrca_covariance=cross-mean*mean,zero_recombination_probability=zero_recombination)
    # After a first recombination, T_j=t+U_j. Integrating the suffix gives
    # exp(-rate*t) * (C0+C1*t+C2*t**2), an exact three-Gamma mixture.
    rate=1+2*c+4*kappa
    coefficients=(derivative(1,1)[1],-derivative(1,0)[1]-derivative(0,1)[1],derivative(0,0)[1])
    masses=tuple(value*math.factorial(j)/rate**(j+1) for j,value in enumerate(coefficients))
    assert 2*c*kappa*kappa*sum(masses)==z*(1-zero_recombination)
    first_recombination=dict(component_shapes=[1,2,3],common_rate=float(rate),
        weights=[dict(rational=str(value/sum(masses)),decimal=float(value/sum(masses))) for value in masses],
        unnormalized_polynomial_coefficients=[str(value) for value in coefficients],
        interpretation='Exact posterior waiting time conditional on either specified first recombination action; evaluation only.')
    return dict(method='exact Fraction arithmetic; derivatives of independent three-state prior Laplace transform',
        time_units='2Ne',kappa=float(kappa),link_rate=float(c),
        first_recombination_time=first_recombination,
        values={name:dict(rational=str(value),decimal=float(value)) for name,value in values.items()})


if __name__=='__main__':
    result=derive()
    path=Path(__file__).resolve().parents[2]/'paper/outputs/poc/reference/analytic_moments.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
