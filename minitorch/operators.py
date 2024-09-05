"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, List

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TASK 0.1
def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal."""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal (1/x)."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second argument."""
    return d / x


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second argument."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second argument."""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TASK 0.3


def map(fn: Callable[[float], float], lst: List[float]) -> List[float]:
    """Applies a function to each element of the list."""
    return [fn(x) for x in lst]


def zipWith(
    fn: Callable[[float, float], float], lst1: List[float], lst2: List[float]
) -> List[float]:
    """Combines two lists element-wise using a given function."""
    return [fn(x, y) for x, y in zip(lst1, lst2)]


def reduce(fn: Callable[[float, float], float], lst: List[float]) -> float:
    """Reduces a list to a single value by repeatedly applying a function."""
    if not lst:
        raise ValueError("Cannot reduce an empty list without an initial value")
    result = lst[0]
    for x in lst[1:]:
        result = fn(result, x)
    return result


def negList(lst: List[float]) -> List[float]:
    """Negate all elements in a list using map."""
    return map(neg, lst)


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Add corresponding elements from two lists using zipWith."""
    return zipWith(add, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Sum all elements in a list using reduce."""
    if not lst:
        return 0.0
    return reduce(add, lst)


def prod(lst: List[float]) -> float:
    """Calculate the product of all elements in a list using reduce."""
    return reduce(mul, lst)
