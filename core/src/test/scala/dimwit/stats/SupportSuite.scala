package dimwit.stats

import dimwit.*
import dimwit.random.Random

class SupportSuite extends DimwitTest:

  trait K derives Label
  trait Samples derives Label

  def vec(values: Float*): Tensor1[K, Float32] = Tensor1(Axis[K]).fromArray(values.toArray)
  def ivec(values: Int*): Tensor1[K, Int32] = Tensor1(Axis[K]).fromArray(values.toArray)

  /** contract 1: every sample lies in the support */
  def samplesInSupport[T <: Tuple: Labels, V](d: Distribution[T, V], n: Int = 1000): Boolean =
    Random.Key(7).splitvmap(Axis[Samples] -> n)(k => d.support.contains(d.sample(k))).all.item

  /** contract 2: logProb is -inf outside the support */
  def logProbOutside[T <: Tuple: Labels, V](d: Distribution[T, V], x: Tensor[T, V]): Float =
    d.support.contains(x).item shouldBe false
    d.logProb(x).asFloat.item

  // one instance of each distribution, with a point outside its support where there is one
  val normal = Normal(vec(0f, 1f), vec(1f, 2f))
  val cauchy = Cauchy(vec(0f, 1f), vec(1f, 2f))
  val studentT = StudentT(Tensor0(3f), vec(0f, 1f), vec(1f, 2f))
  val mvNormal = MVNormal(vec(0f, 1f), Tensor2(Axis[K] -> 2).eye)
  val exponential = Exponential(vec(0.5f, 2f))
  val halfNormal = HalfNormal(vec(0f, 1f), vec(1f, 2f))
  val halfCauchy = HalfCauchy(vec(0f, 1f), vec(1f, 2f))
  val uniform = Uniform(vec(0f, 1f), vec(1f, 3f))
  val truncatedNormal = TruncatedNormal(vec(0f, 1f), vec(1f, 2f), vec(-1f, 0f), vec(1f, 2f))
  val beta = Beta(vec(2f, 5f), vec(5f, 2f))
  val dirichlet = Dirichlet(vec(1f, 2f, 3f))
  val bernoulli = Bernoulli(Prob(vec(0.2f, 0.7f)))
  val binomial = Binomial(Tensor0(10), Prob(vec(0.2f, 0.7f)))
  val discreteUniform = DiscreteUniform(ivec(0, 2), ivec(3, 5))
  val poisson = Poisson(vec(1f, 5f))
  val categorical = Categorical(Prob(vec(0.1f, 0.2f, 0.7f)))
  val multinomial = Multinomial(Tensor0(5), Prob(vec(0.1f, 0.2f, 0.7f)))

  describe("support types are narrow"):
    it("each distribution returns its own case"):
      val _: Support.Real[Tuple1[K], Float32] = normal.support
      val _: Support.Real[Tuple1[K], Float32] = mvNormal.support
      val _: Support.GreaterThan[Tuple1[K], Float32] = exponential.support
      val _: Support.Interval[Tuple1[K], Float32] = uniform.support
      val _: Support.Simplex[K, Float32] = dirichlet.support
      val _: Support.Binary[Tuple1[K]] = bernoulli.support
      val _: Support.IntegerInterval[Tuple1[K], Int32] = binomial.support
      val _: Support.IntegerGreaterThanOrEqual[Tuple1[K], Int32] = poisson.support
      val _: Support.IntegerInterval[EmptyTuple, Int32] = categorical.support
      val _: Support.MultinomialCounts[K, Int32] = multinomial.support

    it("isDiscrete separates continuous and discrete cases"):
      Seq(normal.support, uniform.support, exponential.support).map(_.isDiscrete) shouldBe Seq(false, false, false)
      dirichlet.support.isDiscrete shouldBe false
      Seq(poisson.support, binomial.support, discreteUniform.support).map(_.isDiscrete) shouldBe Seq(true, true, true)
      bernoulli.support.isDiscrete shouldBe true
      multinomial.support.isDiscrete shouldBe true

  describe("samples lie in the support"):
    it("Normal")(samplesInSupport(normal) shouldBe true)
    it("Cauchy")(samplesInSupport(cauchy) shouldBe true)
    it("StudentT")(samplesInSupport(studentT) shouldBe true)
    it("MVNormal")(samplesInSupport(mvNormal) shouldBe true)
    it("Exponential")(samplesInSupport(exponential) shouldBe true)
    it("HalfNormal")(samplesInSupport(halfNormal) shouldBe true)
    it("HalfCauchy")(samplesInSupport(halfCauchy) shouldBe true)
    it("Uniform")(samplesInSupport(uniform) shouldBe true)
    it("TruncatedNormal")(samplesInSupport(truncatedNormal) shouldBe true)
    it("Beta")(samplesInSupport(beta) shouldBe true)
    it("Dirichlet")(samplesInSupport(dirichlet) shouldBe true)
    it("Bernoulli")(samplesInSupport(bernoulli) shouldBe true)
    it("Binomial")(samplesInSupport(binomial) shouldBe true)
    it("DiscreteUniform")(samplesInSupport(discreteUniform) shouldBe true)
    it("Poisson")(samplesInSupport(poisson) shouldBe true)
    it("Categorical")(samplesInSupport(categorical) shouldBe true)
    it("Multinomial")(samplesInSupport(multinomial, n = 100) shouldBe true)

  describe("logProb is -inf outside the support"):
    it("Exponential")(logProbOutside(exponential, vec(-1f, 1f)) shouldBe Float.NegativeInfinity)
    it("HalfNormal")(logProbOutside(halfNormal, vec(1f, 0.5f)) shouldBe Float.NegativeInfinity)
    it("HalfCauchy")(logProbOutside(halfCauchy, vec(1f, 0.5f)) shouldBe Float.NegativeInfinity)
    it("Uniform")(logProbOutside(uniform, vec(0.5f, 4f)) shouldBe Float.NegativeInfinity)
    it("TruncatedNormal")(logProbOutside(truncatedNormal, vec(0f, 3f)) shouldBe Float.NegativeInfinity)
    it("Beta")(logProbOutside(beta, vec(1.5f, 0.5f)) shouldBe Float.NegativeInfinity)
    it("Dirichlet")(logProbOutside(dirichlet, vec(0.5f, 0.6f, 0.2f)) shouldBe Float.NegativeInfinity)
    it("Binomial")(logProbOutside(binomial, ivec(3, 11)) shouldBe Float.NegativeInfinity)
    it("DiscreteUniform: max is exclusive")(logProbOutside(discreteUniform, ivec(3, 3)) shouldBe Float.NegativeInfinity)
    it("Poisson")(logProbOutside(poisson, ivec(-1, 2)) shouldBe Float.NegativeInfinity)
    it("Categorical: index K")(logProbOutside(categorical, Tensor0(3)) shouldBe Float.NegativeInfinity)
    it("Categorical: negative index")(logProbOutside(categorical, Tensor0(-1)) shouldBe Float.NegativeInfinity)
    it("Multinomial")(logProbOutside(multinomial, ivec(1, 1, 1)) shouldBe Float.NegativeInfinity)

  describe("contains"):
    it("accepts the boundary of continuous supports"):
      uniform.support.contains(vec(0f, 3f)).item shouldBe true
      exponential.support.contains(vec(0f, 1f)).item shouldBe true

    it("Real rejects non-finite values"):
      normal.support.contains(vec(Float.PositiveInfinity, 0f)).item shouldBe false
      normal.support.contains(vec(Float.NaN, 0f)).item shouldBe false

  describe("Binomial"):
    it("broadcasts n to the event shape"):
      binomial.support.upper.asFloat32 should approxEqual(vec(10f, 10f))
      binomial.support.lower.asFloat32 should approxEqual(vec(0f, 0f))

    it("works for a scalar Binomial"):
      val scalar = Binomial(Tensor0(10), Prob(Tensor0(0.3f)))
      scalar.support.upper.item shouldBe 10
      samplesInSupport(scalar) shouldBe true
      logProbOutside(scalar, Tensor0(11)) shouldBe Float.NegativeInfinity

  describe("IndependentDistribution.fromUnivariate"):
    it("broadcasts the univariate support to the event shape"):
      val d = IndependentDistribution.fromUnivariate(Shape(Axis[K] -> 3), Exponential(Tensor0(1f)))
      d.support match
        case s: Support.GreaterThan[Tuple1[K], Float32] => s.lower.shape shouldBe Shape(Axis[K] -> 3)
        case other                                      => fail(s"expected GreaterThan, got $other")
      samplesInSupport(d) shouldBe true
