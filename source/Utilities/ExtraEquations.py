"""Class containing equations."""

from operator import (
    add,
    sub,
    mul
)

import numpy as np
from math import (
    cos,
    sin,
    tan,
    floor
)
import cupy

@cupy.fuse()
def protected_div(x1, x2):
    """Calculate the travel time between two points including wait time at the end node."""
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

def add_all(a, b, c, d, e, f):
    """Calculate the travel time between two points including wait time at the end node."""
    x = type(f)
    if x == np.float64 or x == int or x == float:
        return protected_div((a + b + c - d - e), f)
    else:
        return (a + b + c - d - e) / f
    
def add_all_no_health(a, b, c, d, e):
    """Calculate the travel time between two points including wait time at the end node."""
    return a + b + c - d - e

def add_all_with_health(a, b, c, d):
    """Calculate the travel time between two points including wait time at the end node."""
    return (a + b - c) / d

def add_all_with_health_and_level(a, al, b, c, cl, d):
    """Calculate the travel time between two points including wait time at the end node."""
    return (a + b + al - c - cl) / d

def add_all_with_health_and_level_ind_mult(a, al, b, c, cl, d):
    """Calculate the travel time between two points including wait time at the end node."""
    return ((a * al) + (b * al) - (c * cl)) / (d * cl)

def add_all_with_health_and_level_ind_mult_add(a, al1, al2, b, c, cl1, cl2, d):
    """Calculate the travel time between two points including wait time at the end node."""
    return ((cl2 * (c + d + cl1)) - (al2 * (a + b + al1))) / (cl2 * (d + cl1))
    
def new_attacker_value(A, AL):
    """New method for calculating the attacker value"""
    return 0.000162305879817839*AL*(A*AL + 4*A + AL) + 1.4572514429509
    # 1.4572514429509 + 0.000162305879817839*AL^2 + 0.000649223519271356*A*AL  + 0.000162305879817839*A*AL^2

def new_defender_value(D, DL):
    """Return the new defender value"""
    return -2.45761538863145 + 0.027454688044022*(D**2 - 978*D + DL**2 - 1271*DL - 13260/sin(699*D))/(D*DL)

def newest_attack_value(A, AL):
    """The newest attacker value"""
    return sub(AL, protected_div(AL, protected_div(-87, A)))

def newest_attacker_level_value(AL):
    """The newest attacker value"""
    return protected_div(459, sub(AL, sin(cos(-579))))

def newest_power_value(P):
    """The newest attacker value"""
    return tan(protected_div(sub(P, -961), add(add(P, 214), P)))

def newest_defense_value(D):
    """The newest attacker value"""
    return mul(D, cos(cos(tan(mul(mul(D, 636), tan(D))))))

def newest_defender_level_value(DL):
    """The newest attacker value"""
    return tan(mul(protected_div(mul(sin(990), add(DL, cos(add(DL, tan(DL))))), 57), DL))

def newest_health_value(H, DL):
    """The newest attacker value"""
    return mul(DL, sub(sub(-114, DL), H))

def old_health_value(H, DL):
    """The newest attacker value"""
    return floor((2 * H * DL) / 100) + DL + 10

def newest_ratio(A, AL, P, D, DL, H):
    """The final values"""
    return add_all(
        a=newest_attack_value(A, AL),
        b=newest_attacker_level_value(AL),
        c=newest_power_value(P),
        d=newest_defense_value(D),
        e=newest_defender_level_value(DL),
        f=newest_health_value(H, DL)
    )

def newest_damage(A, AL, P, D, DL):
    """The next ratio without health"""
    return add_all_no_health(
        -701,
        AL,
        sub(P, P),
        sub(protected_div(-617, D), tan(289)),
        add(-21, DL)
    )

def current_ratio(A, AL, P, D, DL, H):
    """The current method of telling the ratio"""
    return add_all(
        add(AL, AL),
        sub(AL, protected_div(-658, mul(AL, AL))),
        mul(cos(sin(728)), P),
        D,
        add(DL, DL),
        old_health_value(H, DL)
    )
