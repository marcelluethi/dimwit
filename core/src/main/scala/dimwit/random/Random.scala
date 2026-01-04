package dimwit.random

import dimwit.tensor.*
import dimwit.tensor.TensorOps.*
import dimwit.jax.{Jax, JaxDType}
import me.shadaj.scalapy.py.SeqConverters

/** JAX-based random number generation with proper key management.
  *
  * JAX uses a functional approach to randomness where:
  *   - Random keys must be explicitly managed
  *   - Keys are split to generate independent random streams
  *   - This ensures reproducibility and parallelizability
  *
  * This object provides low-level sampling primitives using JAX. For statistical modeling, prefer using distribution classes in dimwit.distributions.
  */
object Random:

  /** A random key for generating random numbers */
  case class Key(jaxKey: Jax.PyDynamic):

    /** Split this key into multiple independent keys */
    def split(num: Int): Seq[Key] =
      val splitKeys = Jax.jrandom.split(jaxKey, num)
      (0 until num).map(i => Key(splitKeys.__getitem__(i)))

    /** Split this key into multiple independent keys stored in a tensor */
    def splitToTensor[L: Label](axis: Axis[L], num: Int): Tensor1[L, Key] =
      val splitKeys = Jax.jrandom.split(jaxKey, num)
      Tensor[Tuple1[L], Key](splitKeys)

    /** Split into exactly 2 keys (common case) */
    def split2(): (Key, Key) =
      val keys = split(2)
      (keys(0), keys(1))

    /** Generate a tensor of samples by splitting the key along the given axis and applying f to each sub-key ^ */
    def splitvmap[L: Label, T <: Tuple: Labels, V](axis: Axis[L], n: Int)(f: Key => Tensor[T, V]): Tensor[L *: T, V] =
      this.splitToTensor(axis, n).vmap(axis)(k => f(k.item))

    /** Generate a new key by splitting */
    def next(): Key = split2()._2

    override def equals(other: Any): Boolean =
      other match
        case that: Key => Jax.jnp.array_equal(this.jaxKey, that.jaxKey).item().as[Boolean]
        case _         => false

    override def hashCode(): Int = jaxKey.tobytes().hashCode()

  object Key:
    /** Create a random key from an integer seed */
    def apply(seed: Int): Key = Key(Jax.jrandom.key(seed))

    /** Create a random key from current time */
    def fromTime(): Key = Key(System.currentTimeMillis().toInt)

    /** Create a random key from Scala's random */
    def random(): Key = Key(scala.util.Random.nextInt())

  // Enable .item on Tensor0[Key] to extract the Key
  // Note that implementing a Reader instance and using
  // the standard jax.item does not work, as Key is
  // not a primitive type in  JAX.
  extension (tensorKey: Tensor0[Key])
    def item: Key = Key(tensorKey.jaxValue)

  /** Generate a random permutation of indices from 0 to n-1.
    *
    * @param axis
    *   The axis label for the result
    * @param n
    *   The length of the permutation
    * @param key
    *   The random key
    * @return
    *   A 1D tensor containing a random permutation of [0, 1, ..., n-1]
    */
  def permutation[L: Label](axis: Axis[L], n: Int)(key: Key): Tensor1[L, Int] =
    Tensor.fromPy(VType[Int])(Jax.jrandom.permutation(key.jaxKey, n))

  /** Shuffle a tensor along a given axis by randomly permuting elements.
    *
    * @param tensor
    *   The tensor to shuffle
    * @param axis
    *   The axis along which to shuffle
    * @param key
    *   The random key
    * @return
    *   A new tensor with elements shuffled along the specified axis
    */
  def shuffle[T <: Tuple: Labels, V, L: Label](
      tensor: Tensor[T, V],
      axis: Axis[L],
      key: Key
  )(using axisIndex: AxisIndex[T, L]): Tensor[T, V] =
    val axisSize = tensor.shape.dimensions(axisIndex.value)
    val indices = Tensor.fromPy[Tuple1[L], Int](VType[Int])(
      Jax.jrandom.permutation(key.jaxKey, axisSize)
    )
    Tensor(Jax.jnp.take(tensor.jaxValue, indices.jaxValue, axis = axisIndex.value))
