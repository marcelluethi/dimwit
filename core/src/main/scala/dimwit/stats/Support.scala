package dimwit.stats

import dimwit.*
import dimwit.jax.Jax
import me.shadaj.scalapy.py.SeqConverters

/** The set of values one event `Tensor[T, V]` of a distribution can take.
  *
  * The case (e.g. [[Support.Interval]]) is static and can be matched at trace time, e.g. by a PPL choosing a bijector
  * to unconstrained space. The bounds are tensors, so they may be traced and depend on parameters.
  *
  * Continuous bounds are treated as open (e.g. a bijector from unconstrained space never reaches them), but `contains`
  * accepts the boundary: it has measure zero, and floating point rounding can put samples onto it.
  * Discrete bounds are inclusive.
  *
  * The built-in cases are sealed, so a match on them is checked for exhaustiveness. Supports DimWit does not model
  * extend [[Support.Custom]].
  */
sealed trait Support[T <: Tuple, V]:

  /** true iff the whole event lies in the support */
  def contains(x: Tensor[T, V]): Tensor0[Bool]

  /** whether the support is discrete, i.e. cannot be reached by a bijector from unconstrained space */
  def isDiscrete: Boolean

object Support:

  /** A support whose constraint holds for each element independently, so it can be broadcast to a larger event shape. */
  sealed trait Elementwise[T <: Tuple, V] extends Support[T, V]:
    private[stats] def broadcastTo[O <: Tuple: Labels](shape: Shape[O]): Support[O, V]

  private def broadcastBound[T <: Tuple, O <: Tuple: Labels, V](bound: Tensor[T, V], shape: Shape[O]): Tensor[O, V] =
    Tensor(Jax.jnp.broadcast_to(bound.jaxValue, shape.dimensions.toPythonProxy))

  // ---- continuous, elementwise ----

  /** all real numbers (finite values) */
  final case class Real[T <: Tuple: Labels, V: IsFloating]() extends Elementwise[T, V]:
    def isDiscrete = false
    def contains(x: Tensor[T, V]): Tensor0[Bool] = x.isfinite.all
    private[stats] def broadcastTo[O <: Tuple: Labels](shape: Shape[O]): Real[O, V] = Real()

  /** x > lower elementwise */
  final case class GreaterThan[T <: Tuple: Labels, V: IsFloating](lower: Tensor[T, V]) extends Elementwise[T, V]:
    def isDiscrete = false
    def contains(x: Tensor[T, V]): Tensor0[Bool] = (x >= lower).all
    private[stats] def broadcastTo[O <: Tuple: Labels](shape: Shape[O]): GreaterThan[O, V] = GreaterThan(broadcastBound(lower, shape))

  /** x < upper elementwise */
  final case class LessThan[T <: Tuple: Labels, V: IsFloating](upper: Tensor[T, V]) extends Elementwise[T, V]:
    def isDiscrete = false
    def contains(x: Tensor[T, V]): Tensor0[Bool] = (x <= upper).all
    private[stats] def broadcastTo[O <: Tuple: Labels](shape: Shape[O]): LessThan[O, V] = LessThan(broadcastBound(upper, shape))

  /** lower < x < upper elementwise */
  final case class Interval[T <: Tuple: Labels, V: IsFloating](lower: Tensor[T, V], upper: Tensor[T, V]) extends Elementwise[T, V]:
    def isDiscrete = false
    def contains(x: Tensor[T, V]): Tensor0[Bool] = ((x >= lower) and (x <= upper)).all
    private[stats] def broadcastTo[O <: Tuple: Labels](shape: Shape[O]): Interval[O, V] = Interval(broadcastBound(lower, shape), broadcastBound(upper, shape))

  // ---- continuous, event-level ----

  /** x_i >= 0 and sum_i x_i = 1 along L (up to `tolerance`) */
  final case class Simplex[L: Label, V: IsFloating](tolerance: Float = 1e-5f) extends Support[Tuple1[L], V]:
    def isDiscrete = false
    def contains(x: Tensor1[L, V]): Tensor0[Bool] =
      val nonNegative = (x >= Tensor.like(x).fill(0f)).all
      val sumsToOne = x.sum.approxEquals(Tensor0(VType[V])(1f), tolerance)
      nonNegative and sumsToOne

  // ---- discrete ----

  /** true and false */
  final case class Binary[T <: Tuple: Labels]() extends Elementwise[T, Bool]:
    def isDiscrete = true
    def contains(x: Tensor[T, Bool]): Tensor0[Bool] = Tensor0(true)
    private[stats] def broadcastTo[O <: Tuple: Labels](shape: Shape[O]): Binary[O] = Binary()

  /** lower <= x <= upper elementwise, both inclusive */
  final case class IntegerInterval[T <: Tuple: Labels, V: IsInteger](lower: Tensor[T, V], upper: Tensor[T, V]) extends Elementwise[T, V]:
    def isDiscrete = true
    def contains(x: Tensor[T, V]): Tensor0[Bool] = ((x >= lower) and (x <= upper)).all
    private[stats] def broadcastTo[O <: Tuple: Labels](shape: Shape[O]): IntegerInterval[O, V] =
      IntegerInterval(broadcastBound(lower, shape), broadcastBound(upper, shape))

  /** x >= lower elementwise, unbounded above */
  final case class IntegerGreaterThanOrEqual[T <: Tuple: Labels, V: IsInteger](lower: Tensor[T, V]) extends Elementwise[T, V]:
    def isDiscrete = true
    def contains(x: Tensor[T, V]): Tensor0[Bool] = (x >= lower).all
    private[stats] def broadcastTo[O <: Tuple: Labels](shape: Shape[O]): IntegerGreaterThanOrEqual[O, V] =
      IntegerGreaterThanOrEqual(broadcastBound(lower, shape))

  /** non-negative counts along L that sum to n */
  final case class MultinomialCounts[L: Label, V: IsInteger](n: Tensor0[V]) extends Support[Tuple1[L], V]:
    def isDiscrete = true
    def contains(x: Tensor1[L, V]): Tensor0[Bool] =
      val nonNegative = (x >= Tensor.like(x).fill(0)).all
      nonNegative and Tensor.arrayEqual(x.sum, n)

  // ---- convenience constructors ----

  /** x > 0 elementwise */
  def positive[T <: Tuple: Labels, V: IsFloating](shape: Shape[T]): GreaterThan[T, V] =
    GreaterThan(Tensor(shape, VType[V]).fill(0f))

  /** 0 < x < 1 elementwise */
  def unitInterval[T <: Tuple: Labels, V: IsFloating](shape: Shape[T]): Interval[T, V] =
    Interval(Tensor(shape, VType[V]).fill(0f), Tensor(shape, VType[V]).fill(1f))

  /** x >= 0 elementwise */
  def nonNegativeInteger[T <: Tuple: Labels, V: IsInteger](shape: Shape[T]): IntegerGreaterThanOrEqual[T, V] =
    IntegerGreaterThanOrEqual(Tensor(shape, VType[V]).fill(0))

  // ---- extension point ----

  /** Extension point for supports DimWit does not model, e.g. positive-definite matrices or ordered vectors.
    * Unlike the built-in cases, it can be extended outside DimWit. Code matching on Support handles it generically,
    * e.g. through a registry or a type class keyed by the custom type.
    */
  trait Custom[T <: Tuple, V] extends Support[T, V]:

    /** The support of a tensor of independent draws with this support, needed by
      * [[IndependentDistribution.fromUnivariate]]. None if the constraint cannot be applied per element.
      */
    def broadcastTo[O <: Tuple: Labels](shape: Shape[O]): Option[Support[O, V]] = None

  /** The support of a tensor of independent draws from a univariate distribution with support `s`. */
  def broadcast[T <: Tuple: Labels, V](s: Support[EmptyTuple, V], shape: Shape[T]): Support[T, V] = s match
    case e: Elementwise[EmptyTuple, V] => e.broadcastTo(shape)
    case c: Custom[EmptyTuple, V]      =>
      c.broadcastTo(shape).getOrElse(throw IllegalArgumentException(s"The custom support $c does not define broadcastTo, so it cannot be broadcast to $shape."))
