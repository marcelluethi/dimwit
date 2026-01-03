import scala.annotation.targetName

import dimwit.jax.Jax
import me.shadaj.scalapy.py
import dimwit.autodiff.TensorTree
import dimwit.tensor.Tensor
import dimwit.tensor.Labels

package object dimwit:

  import scala.compiletime.ops.string.+

  object StringLabelMath:
    infix type *[A <: String, B <: String] = A + "*" + B

  trait Prime[T]
  object Prime:
    given [L](using label: Label[L]): Label[Prime[L]] with
      val name: String = s"${label.name}'"

    type RemovePrimes[T <: Tuple] <: Tuple = T match
      case EmptyTuple       => EmptyTuple
      case Prime[l] *: tail => l *: RemovePrimes[tail]
      case h *: tail        => h *: RemovePrimes[tail]

    extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])
      def dropPrimes: Tensor[RemovePrimes[T], V] =
        given newLabels: Labels[RemovePrimes[T]] with
          val names: List[String] =
            val oldLabels = summon[Labels[T]]
            oldLabels.names.toList.map(_.replace("'", ""))
        Tensor[RemovePrimes[T], V](tensor.jaxValue)

  def gc(): Unit =
    System.gc()
    Jax.gc()

  /** Executes a block of code with automatic cleanup of Python objects created within the block.
    *
    * This is useful for preventing memory leaks in training loops or other scenarios where temporary Python objects (like loss values, intermediate tensors) are created repeatedly. All Python objects created within the block are freed immediately when the block exits.
    *
    * Example:
    * {{{
    * // In a training loop - clean up temporary loss evaluation
    * withLocalCleanup {
    *   val loss = jitLoss(params)
    *   println(s"Loss: $loss")
    * } // loss and its Python objects are freed here
    * }}}
    *
    * WARNING: Do not return Python objects from the block - they will be invalid after cleanup. Only use this for temporary evaluations that don't need to persist.
    */

  def withLocalCleanup[T](f: => T): T = f

  @targetName("On")
  infix trait ~[A, B]
  object `~`:
    given [A, B](using labelA: Label[A], labelB: Label[B]): Label[A ~ B] with
      val name: String = s"${labelA.name}_on_${labelB.name}"

  @targetName("Combined")
  infix trait |*|[A, B]
  object `|*|`:
    given [A, B](using labelA: Label[A], labelB: Label[B]): Label[A |*| B] with
      val name: String = s"${labelA.name}*${labelB.name}"

  // Export tensor and related types
  export dimwit.tensor.{Tensor, Tensor0, Tensor1, Tensor2, Tensor3}
  export dimwit.tensor.{Shape, Shape0, Shape1, Shape2, Shape3}
  export dimwit.tensor.{DType, Device}
  export dimwit.tensor.{VType, ExecutionType, Label, Labels, Axis, AxisIndex, AxisIndices, Dim}

  // Export operations
  export dimwit.tensor.TensorOps.*

  // Export automatic differentiation
  export dimwit.autodiff.{Autodiff, TensorTree, FloatTensorTree, ToPyTree}

  // Export Just-in-Time compilation
  export dimwit.jax.Jit.jit
  export dimwit.jax.Jit.jitUpdate

  object Conversions:
    export dimwit.tensor.Tensor0.{float2FloatTensor, int2IntTensor, int2FloatTensor, boolean2BooleanTensor}
