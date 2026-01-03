package dimwit.jax

import dimwit.tensor.{Tensor, Shape, Labels}
import dimwit.jax.{Jax, JaxDType}
import dimwit.autodiff.ToPyTree
import me.shadaj.scalapy.py

object Jit:

  // TODO replace with TupledFunction when available (no longer experimental)
  def jit[T1, T2, R](f: (T1, T2) => R)(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], outTree: ToPyTree[R]): (T1, T2) => R =
    val jitF = jit(f.tupled)
    (t1, t2) => jitF((t1, t2))
  def jit[T1, T2, T3, R](f: (T1, T2, T3) => R)(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], t3Tree: ToPyTree[T3], outTree: ToPyTree[R]): (T1, T2, T3) => R =
    val jitF = jit(f.tupled)
    (t1, t2, t3) => jitF((t1, t2, t3))

  def jit[InPyTree: ToPyTree, OutPyTree: ToPyTree](
      f: InPyTree => OutPyTree
  ): InPyTree => OutPyTree =

    // Python function that accepts a pytree
    val fpy = (pyTreePy: Jax.PyDynamic) =>
      val pyTree = ToPyTree[InPyTree].fromPyTree(pyTreePy)
      val result = f(pyTree)
      val tt = ToPyTree[OutPyTree].toPyTree(result)
      tt

    // Apply JIT compilation
    val jitted = Jax.jax_helper.jit_fn(fpy)

    // Return a function that converts Scala types to pytree and applies jitted function
    (pyTree: InPyTree) =>
      val pyTreePy = ToPyTree[InPyTree].toPyTree(pyTree)
      val res = jitted(pyTreePy)
      ToPyTree[OutPyTree].fromPyTree(res)

  /** JIT compiles an update function with buffer donation.
    *
    * This enables JAX to reuse the input buffer for the output, significantly reducing memory usage in training loops where the old parameters are no longer needed.
    *
    * WARNING: After calling this function, the input argument is "donated" and should not be used again. JAX may reuse its memory for the output.
    *
    * Example:
    * {{{
    * val jitStep = jitUpdate(gradientStep)
    * params = jitStep(params)  // Old params memory is reused
    * }}}
    *
    * @param f
    *   Update function that takes input and returns updated version
    * @return
    *   JIT compiled function with buffer donation
    */
  def jitUpdate[T: ToPyTree](f: T => T): T => T =

    // Python function that accepts a pytree
    val fpy = (pyTreePy: Jax.PyDynamic) =>
      val pyTree = ToPyTree[T].fromPyTree(pyTreePy)
      val result = f(pyTree)
      val tt = ToPyTree[T].toPyTree(result)
      tt

    // Apply JIT compilation with buffer donation
    val jitted = Jax.jax_helper.jit_update_fn(fpy)

    // Return a function that converts Scala types to pytree and applies jitted function
    (pyTree: T) =>
      val pyTreePy = ToPyTree[T].toPyTree(pyTree)
      val res = jitted(pyTreePy)
      ToPyTree[T].fromPyTree(res)
