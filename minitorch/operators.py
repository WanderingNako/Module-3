"""Collection of the core mathematical operators used throughout the code base."""
import math
from typing import Callable, Iterable
# ## Task 0.1

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


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiplies two numbers together."""
    return x * y

def id(x: float) -> float:
    """Returns the input."""
    return x

def add(x: float, y: float) -> float:
    """Adds two numbers together."""
    return x + y

def neg(x: float) -> float:
    """Negates a number."""
    return -x

def lt(x: float, y: float) -> float:
    """Returns 1.0 if x < y else 0.0."""
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    """Returns 1.0 if x == y else 0.0."""
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    """Returns the maximum of x and y."""
    return x if x > y else y

def is_close(x: float, y: float) -> bool:
    """Returns True if x and y are within 1e-2 of each other."""
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    """Returns the sigmoid of x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
    
def relu(x: float) -> float:
    """Returns the ReLU of x."""
    return x if x > 0 else 0.0

def log(x: float) -> float:
    """Returns the natural logarithm of x."""
    return math.log(x)

def exp(x: float) -> float:
    """Returns the exponential of x."""
    return math.exp(x)

def inv(x: float) -> float:
    """Returns the inverse of x."""
    return 1.0 / x

def log_back(x: float, d: float) -> float:
    """Returns the backward pass of log."""
    return d / x

def inv_back(x: float, d: float) -> float:
    """Returns the backward pass of inv."""
    return -d / (x * x)

def relu_back(x: float, d: float) -> float:
    """Returns the backward pass of relu."""
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


# TODO: Implement for Task 0.3.
def map(ls: Iterable[float], fn: Callable[[float], float]) -> Iterable[float]:
    """Applies fn to each element in ls."""
    return [fn(x) for x in ls]

def zipWith(ls1: Iterable[float], ls2: Iterable[float], fn: Callable[[float, float], float]) -> Iterable[float]:
    """Applies fn to each pair of elements in ls1 and ls2."""
    return [fn(x, y) for x, y in zip(ls1, ls2)]

def reduce(ls: Iterable[float], fn: Callable[[float, float], float], start: float) -> float:
    """Reduces ls into a single value using fn, starting with start."""
    result = start
    for x in ls:
        result = fn(result, x)
    return result

def negList(ls: Iterable[float]) -> Iterable[float]:
    """Returns a list with each element negated."""
    return map(ls, neg)

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Returns a list with each element the sum of the elements in ls1 and ls2."""
    return zipWith(ls1, ls2, add)

def sum(ls: Iterable[float]) -> float:
    """Returns the sum of the elements in ls."""
    return reduce(ls, add, 0.0)

def prod(ls: Iterable[float]) -> float:
    """Returns the product of the elements in ls."""
    return reduce(ls, mul, 1.0)