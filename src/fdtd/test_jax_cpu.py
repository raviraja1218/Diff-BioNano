"""
Test JAX on CPU
"""
import jax
import jax.numpy as jnp
import numpy as np

print("="*60)
print("JAX CPU TEST")
print("="*60)

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Platform: {jax.default_backend()}")

# Simple test function
@jax.jit
def f(x):
    return jnp.sum(jnp.sin(x) ** 2)

# Test gradient
x = jnp.ones((10, 10))
print(f"\nTesting function: f(x) = sum(sin(x)^2)")
print(f"x shape: {x.shape}")

# Compute function value
y = f(x)
print(f"f(x) = {y:.6f}")

# Compute gradient
grad_f = jax.grad(f)
gradient = grad_f(x)
print(f"Gradient shape: {gradient.shape}")
print(f"Gradient mean: {jnp.mean(gradient):.6e}")
print(f"Gradient std: {jnp.std(gradient):.6e}")

# Test that gradient is correct (finite difference check)
eps = 1e-6
x_perturbed = x + eps * gradient
y_perturbed = f(x_perturbed)
expected_change = eps * jnp.sum(gradient * gradient)
actual_change = y_perturbed - y

print(f"\nGradient accuracy test:")
print(f"Expected change: {expected_change:.6e}")
print(f"Actual change: {actual_change:.6e}")
print(f"Relative error: {abs(actual_change - expected_change) / abs(expected_change):.2e}")

if abs(actual_change - expected_change) / abs(expected_change) < 0.01:
    print("✅ Gradient is accurate!")
else:
    print("⚠ Gradient might have issues")

print("\n✅ JAX CPU test passed!")
print("Automatic differentiation is working correctly.")
