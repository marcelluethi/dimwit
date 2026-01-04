package dimwit.random

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.tensor.TestUtil.*
import me.shadaj.scalapy.py

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class RandomSuite extends AnyFunSuite with Matchers:
  trait A derives Label
  trait B derives Label
  trait Samples derives Label

  test("splitToTensor creates tensor of correct shape"):
    val key = Random.Key(42)
    val n = 5
    val tensorKeys = key.splitToTensor(Axis[Samples], n)
    tensorKeys.shape should equal(Shape(Axis[Samples] -> n))

  test("splitToTensor creates same keys as manual split"):

    val key = Random.Key(42)
    val n = 5
    val tensorKeys = key.splitToTensor(Axis[Samples], n)

    val splitKeys = key.split(n)
    for i <- 0 until n do
      val tensorKey = tensorKeys.slice(Axis[Samples] -> i).item
      val splitKey = splitKeys(i)
      tensorKey should equal(splitKey)

  test("item returns the jax key"):
    val key = Random.Key(123)
    val tensor0Key = Tensor0[Random.Key](key.jaxKey)
    val extractedKey = tensor0Key.item

    // The extracted key should have the same underlying JAX key
    extractedKey should equal(key)

  test("splitvmap generates same random numbers as individual calls"):
    val key = Random.Key(456)
    val n = 3

    // Generate random numbers using splitvmap
    val vmapResults = key.splitvmap(Axis[Samples], n) { k =>
      Tensor0.randn(k)
    }

    // Generate random numbers using individual calls
    val splitKeys = key.split(n)
    val individualResults = Tensor1.fromArray(Axis[Samples], VType[Float])(
      splitKeys.map(k => Tensor0.randn(k).item).toArray
    )

    vmapResults should approxEqual(individualResults)

  test("permutation generates all indices from 0 to n-1"):
    val key = Random.Key(789)
    val n = 10
    val perm = Random.permutation(Axis[A], n)(key)

    perm.shape should equal(Shape(Axis[A] -> n))

    // Check that all indices from 0 to n-1 are present by using JAX sort
    val sortedJax = Jax.jnp.sort(perm.jaxValue)
    val expected = Tensor1.fromArray(Axis[A], VType[Int])((0 until n).toArray)
    val sortedPerm = Tensor.fromPy[Tuple1[A], Int](VType[Int])(sortedJax)
    sortedPerm should equal(expected)

  test("permutation with different keys produces different orders"):
    val key1 = Random.Key(100)
    val key2 = Random.Key(200)
    val n = 10

    val perm1 = Random.permutation(Axis[A], n)(key1)
    val perm2 = Random.permutation(Axis[A], n)(key2)

    // Permutations should be different
    Jax.jnp.array_equal(perm1.jaxValue, perm2.jaxValue).item().as[Boolean] should be(false)

  test("shuffle preserves tensor shape"):
    val key = Random.Key(123)
    val tensor = Tensor.fromArray(
      Shape(Axis[A] -> 5, Axis[B] -> 3),
      VType[Float]
    )(Array(
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f
    ))

    val shuffled = Random.shuffle(tensor, Axis[A], key)

    shuffled.shape should equal(tensor.shape)

  test("shuffle preserves all elements"):
    val key = Random.Key(456)
    val tensor = Tensor1.fromArray(Axis[A], VType[Float])(
      Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
    )

    val shuffled = Random.shuffle(tensor, Axis[A], key)

    // Sort both using JAX and they should be equal
    val sortedOriginal = Tensor.fromPy[Tuple1[A], Float](VType[Float])(
      Jax.jnp.sort(tensor.jaxValue)
    )
    val sortedShuffled = Tensor.fromPy[Tuple1[A], Float](VType[Float])(
      Jax.jnp.sort(shuffled.jaxValue)
    )

    sortedOriginal should approxEqual(sortedShuffled)

  test("shuffle with different keys produces different orders"):
    val key1 = Random.Key(111)
    val key2 = Random.Key(222)
    val tensor = Tensor1.fromArray(Axis[A], VType[Float])(
      Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f)
    )

    val shuffled1 = Random.shuffle(tensor, Axis[A], key1)
    val shuffled2 = Random.shuffle(tensor, Axis[A], key2)

    // Shuffled tensors should be different
    Jax.jnp.array_equal(shuffled1.jaxValue, shuffled2.jaxValue).item().as[Boolean] should be(false)

  test("shuffle along specific axis in 2D tensor"):
    val key = Random.Key(333)
    trait C derives Label

    val tensor = Tensor.fromArray(
      Shape(Axis[A] -> 3, Axis[C] -> 4),
      VType[Float]
    )(Array(
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f
    ))

    // Shuffle along first axis - columns should stay intact
    val shuffledA = Random.shuffle(tensor, Axis[A], key)

    shuffledA.shape should equal(tensor.shape)

    // Verify each row appears exactly once by sorting along the shuffled axis
    val sortedOriginal = Tensor.fromPy[(A, C), Float](VType[Float])(
      Jax.jnp.sort(tensor.jaxValue, axis = 0)
    )
    val sortedShuffled = Tensor.fromPy[(A, C), Float](VType[Float])(
      Jax.jnp.sort(shuffledA.jaxValue, axis = 0)
    )
    sortedOriginal should approxEqual(sortedShuffled)

  test("shuffle is deterministic with same key"):
    val key = Random.Key(999)
    val tensor = Tensor1.fromArray(Axis[A], VType[Int])(
      Array(10, 20, 30, 40, 50)
    )

    val shuffled1 = Random.shuffle(tensor, Axis[A], key)
    val shuffled2 = Random.shuffle(tensor, Axis[A], key)

    shuffled1 should equal(shuffled2)
