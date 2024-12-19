"""
delayed - A simple library for delayed computation in Python.

This module provides a simple framework for defining and managing delayed computations
in Python. It allows you to define functions that are not immediately evaluated, but
instead are computed only when the result is needed. This can be useful for complex
computations that are expensive or time-consuming, or for building complex computation
graphs with dependencies.

The main class is `Delayed`, which represents a delayed computation. You can define
a delayed computation by wrapping a function with the `@delayed` decorator, which
returns a `Delayed` object. You can then call the `compute()` method on the `Delayed`
object to compute the result.

Example usage:
```python
@delayed
def add(a, b):
    return a + b

@delayed
def multiply(a, b):
    return a * b

@delayed
def power(a, b):
    return a ** b

# Define a computation graph
result = add(multiply(2, 3), power(4, 5))

# Compute the result
print(result.compute()) # will take some time (not really in this case)
print(result.compute()) # will be instant, as the result is cached
```

The `Delayed` class also provides methods for analyzing the computation graph, visualizing
the dependencies, and clearing cached results. You can also access the computation graph
directly using the `get_dependency_graph()` method. You can visualize the computation graph
with the `visualize()` method, which uses NetworkX and Matplotlib to draw the graph.


Areas for improvement:
- Add support for parallel computation
- Add memory load handling and breaking cache (e.g. if memory usage exceeds a threshold)
- Add more sophisticated graph analysis and visualization features
- Benchmarking and profiling tools
- Test suites!
- Context manager for temporary cache control...
- Memoization for repeated objects....
- Serialiation support for saving/loading delayed objects and pipelines
- Autodetecting the need to recompute based on changes in dependencies and
  the ability to update an intermediate node and have other nodes recognize this. 
  - Use the graph to determine which nodes need to be recomputed!! Can use the Dependent part of the graph...
- Implement operations on delayed objects including all base operators
  - Arithmetic (+, -, *, /, //, %, **)
  - Comparison (<, >, <=, >=, ==, !=)
  - Bitwise (&, |, ^, <<, >>)
  - Boolean (and, or, not)
- To do the above, will also need to implement a way to implement delayed inline (with lambda?)
  - Actually, can always just wrap the function in delayed inline too...
    - Like: delayed(some_function, cache_data=True)(arg1, arg2, **kwargs)
- If a single delayed object is called without caching, then the whole tree that depends on it
  will automatically recompute because of the way updates are handled! Think more about this...
"""

from .delayed import delayed
