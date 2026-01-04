# JAX helper functions for dimwit library

import jax
import jax.numpy as jnp
from jax import vmap

import builtins
builtins.jax = jax
builtins.jnp = jnp

def vmap(f, dims):
    """
    Applies a function `f` to a tensor using JAX's vmap functionality.
    
    It is wrapped in a Python function to ensure that the function, as otherwise
    jax will crash upon inspection.
    """
                
    def python_wrapper(x):
        return f(x)
            
    return jax.vmap(python_wrapper, in_axes=dims)
           
def zipvmap(f, dims):
    def python_wrapper(*args):
        return f(args)
    return lambda jax_inputs_tuple: jax.vmap(python_wrapper, in_axes=dims)(*jax_inputs_tuple)

def apply_over_axes(f, axis):
    """
    Applies a function `f` over specified axes using JAX's vmap functionality.
    
    Args:
        f: Function that takes one argument (x)
        axis: Axis or tuple of axes to map over
    
    It is wrapped in a Python function to ensure that the function, as otherwise
    jax will crash upon inspection.
    """
                
    # Wrap the ScalaPy function in a pure Python wrapper
    def python_wrapper(x):
        return f(x)
            
    # Create vmap with the wrapper
    return jnp.apply_over_axes(python_wrapper, axis)

def vmap2(f, dims):
    """
    Applies a function `f` to two tensors using JAX's vmap functionality.
    
    Args:
        f: Function that takes two arguments (x, y)
        dims: Either an integer (same axis for both inputs) or tuple (axis1, axis2)
    
    It is wrapped in a Python function to ensure that the function, as otherwise
    jax will crash upon inspection.
    """
                
    # Wrap the ScalaPy function in a pure Python wrapper
    def python_wrapper(x, y):
        return f(x, y)
    
    # Handle dims parameter - can be int or tuple
    if isinstance(dims, int):
        # Same axis for both inputs
        in_axes = (dims, dims)
    else:
        # Different axes for each input
        in_axes = dims
            
    # Create vmap with the wrapper
    return jax.vmap(python_wrapper, in_axes=in_axes)



def grad(f):
    """
    Computes the gradient of a function `f` with respect to its arguments.
    
    This is a simple wrapper around JAX's grad function.
    Only works for scalar-output functions.
    """
    from jax import grad as jax_grad
    def python_wrapper(*args):
        # Remove debug print that might cause issues
        return f(*args)
    
    return jax_grad(python_wrapper)

def value_and_grad(f):
    """
    Computes both the value and gradient of a function `f` with respect to its arguments.
    
    This is more efficient than computing value and gradient separately.
    Only works for scalar-output functions.
    """
    from jax import value_and_grad as jax_value_and_grad
    def python_wrapper(*args):
        return f(*args)
    
    return jax_value_and_grad(python_wrapper)

def jacfwd(f):
    """
    Computes the Jacobian of a function `f` using forward-mode differentiation.
    
    Works for vector-output functions. Efficient when output dimension > input dimension.
    """
    from jax import jacfwd as jax_jacfwd
    def python_wrapper(x):
        return f(x)
    
    return jax_jacfwd(python_wrapper)

def jacrev(f):
    """
    Computes the Jacobian of a function `f` using reverse-mode differentiation.
    
    Works for vector-output functions. Efficient when input dimension > output dimension.
    """
    from jax import jacrev as jax_jacrev
    def python_wrapper(x):
        return f(x)
    
    return jax_jacrev(python_wrapper)

def jacobian(f):
    from jax import jacobian as jax_jacobian
    def python_wrapper(x):
        return f(x)
    return jax_jacobian(python_wrapper)

def jit(f):
    """
    Just-in-time compiles a function for faster execution.
    
    The first call will be slower due to compilation, but subsequent calls
    with the same shapes will be much faster.
    """
    from jax import jit as jax_jit
    def python_wrapper(*args):
        return f(*args)
    
    return jax_jit(python_wrapper)

def jit_fn(f):
    """
    Universal JIT wrapper that works with any function.
    Simply wraps the function in a Python wrapper and JIT compiles it.
    
    This is the simplest and most flexible approach - works with:
    - Regular functions
    - vmap'ed functions
    - grad functions
    - Any combination
    
    Args:
        f: Any function to JIT compile
    
    Returns:
        JIT compiled function
    """
    from jax import jit as jax_jit
    def python_wrapper(*args):
        return f(*args)
    return jax_jit(python_wrapper)

def jit_update_fn(f, donate_argnums=None):
    """
    JIT wrapper with buffer donation for update functions.
    Donates specified arguments to allow JAX to reuse their memory for the output.
    Use this for training loops where you don't need the old parameters.
    
    Args:
        f: Update function that takes params and returns updated params
        donate_argnums: Tuple of argument indices to donate (default: (0,))
    
    Returns:
        JIT compiled function with buffer donation for specified arguments
    """
    from jax import jit as jax_jit
    if donate_argnums is None:
        donate_argnums = (0,)
    def python_wrapper(*args):
        return f(*args)
    return jax_jit(python_wrapper, donate_argnums=donate_argnums)


def serialize_pytree(pytree):
    """
    Serialize a JAX PyTree to a base64-encoded string.
    
    Uses JAX's tree_util to flatten the PyTree structure, then pickles
    the flattened values and tree definition. Handles JAX PRNG keys by
    converting them to their raw data representation before pickling.
    
    Args:
        pytree: Any JAX PyTree structure (nested dict/list/tuple of arrays)
    
    Returns:
        Base64-encoded string containing the pickled PyTree data
    """
    import pickle
    import base64
    from jax import tree_util
    from jax import random
    import numpy as np
    
    # Flatten the PyTree to separate structure from data
    flat_values, tree_def = tree_util.tree_flatten(pytree)
    
    # Convert JAX arrays to numpy and extract PRNG key data
    flat_numpy = []
    for v in flat_values:
        # Check if this is a JAX PRNG key
        if hasattr(v, 'dtype') and 'key' in str(type(v)).lower():
            # Extract raw key data for newer JAX versions
            try:
                key_data = random.key_data(v)
                flat_numpy.append(('prng_key', np.asarray(key_data)))
            except:
                # Fallback for older JAX versions
                flat_numpy.append(('prng_key', np.asarray(v)))
        else:
            # Regular array
            flat_numpy.append(('array', np.asarray(v)))
    
    # Package the data with tree structure
    data = {'values': flat_numpy, 'treedef': tree_def}
    
    # Pickle and encode to base64
    pickled = pickle.dumps(data)
    return base64.b64encode(pickled).decode('utf-8')


def deserialize_pytree(b64_string):
    """
    Deserialize a JAX PyTree from a base64-encoded string.
    
    Reconstructs the PyTree from the pickled data, converting numpy arrays
    back to JAX arrays and reconstructing PRNG keys properly.
    
    Args:
        b64_string: Base64-encoded string from serialize_pytree
    
    Returns:
        Reconstructed JAX PyTree
    """
    import pickle
    import base64
    from jax import tree_util
    from jax import random
    
    # Decode from base64 and unpickle
    pickled = base64.b64decode(b64_string.encode('utf-8'))
    data = pickle.loads(pickled)
    
    # Convert numpy arrays back to JAX arrays and reconstruct keys
    flat_jax = []
    for value_type, value in data['values']:
        if value_type == 'prng_key':
            # Reconstruct PRNG key from raw data
            try:
                # For newer JAX versions that use typed keys
                key = random.wrap_key_data(value)
            except:
                # Fallback: convert to JAX array (works for older versions)
                key = jnp.asarray(value)
            flat_jax.append(key)
        else:
            # Regular array
            flat_jax.append(jnp.asarray(value))
    
    # Reconstruct the PyTree structure
    return tree_util.tree_unflatten(data['treedef'], flat_jax)

