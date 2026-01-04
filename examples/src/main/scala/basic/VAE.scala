package examples.basic

import dimwit.*
import dimwit.stats.Normal
import dimwit.random.Random
import examples.basic.Encoder.EncoderParams
import examples.basic.Decoder.DecoderParams
import examples.basic.MLPClassifierMNist.MNISTLoader
import nn.LinearLayer
import nn.ActivationFunctions.relu
import examples.basic.MLPClassifierMNist.{Sample, TrainSample, Height, Width}
import nn.GradientDescent
import dimwit.jax.Jax
import nn.ActivationFunctions.sigmoid
import dimwit.random.Random.Key
import examples.basic.MLPClassifierMNist.TestSample

type SourceFeature = Height |*| Width
type ReconstructedFeature = Height |*| Width

trait EHidden1 derives Label
trait EHidden2 derives Label

trait Latent derives Label
trait MeanLatent extends Latent derives Label
trait LogVarLatent extends Latent derives Label

trait DHidden1 derives Label
trait DHidden2 derives Label

trait Batch derives Label

type FTensor1[T] = Tensor1[T, Float]

class Encoder(p: EncoderParams):
  def apply(v: FTensor1[Height |*| Width]): (FTensor1[MeanLatent], FTensor1[LogVarLatent]) =
    val layer1 = LinearLayer(p.layer1)
    val layer2 = LinearLayer(p.layer2)
    val meanLayer = LinearLayer(p.meanLayer)
    val logVarLayer = LinearLayer(p.logVarLayer)

    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    val mean = meanLayer(h2)
    val logVar = logVarLayer(h2).clip(Tensor0(-10f), Tensor0(10f))

    (mean, logVar)

object Encoder:
  case class EncoderParams(
      layer1: LinearLayer.Params[Height |*| Width, EHidden1],
      layer2: LinearLayer.Params[EHidden1, EHidden2],
      meanLayer: LinearLayer.Params[EHidden2, MeanLatent],
      logVarLayer: LinearLayer.Params[EHidden2, LogVarLatent]
  )

class Decoder(p: DecoderParams):
  def apply(v: FTensor1[Latent]): FTensor1[ReconstructedFeature] =
    val layer1 = LinearLayer(p.layer1)
    val layer2 = LinearLayer(p.layer2)
    val outputLayer = LinearLayer(p.outputLayer)

    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    val reconstructed = sigmoid(outputLayer(h2))

    reconstructed

object Decoder:
  case class DecoderParams(
      layer1: LinearLayer.Params[Latent, DHidden1],
      layer2: LinearLayer.Params[DHidden1, DHidden2],
      outputLayer: LinearLayer.Params[DHidden2, ReconstructedFeature]
  )

def reparametrize(mean: FTensor1[Latent], logVar: FTensor1[Latent], key: Random.Key): FTensor1[Latent] =
  val std = (logVar *! Tensor0(0.5f)).exp
  Normal(mean, std).sample(key)

case class VAE(params: VAE.Params):

  val encoder = Encoder(params.encoderParams)
  val decoder = Decoder(params.decoderParams)

  def apply(v: FTensor1[SourceFeature], key: Random.Key): (FTensor1[ReconstructedFeature], FTensor1[Latent], FTensor1[Latent]) =
    val (mean, logVar) = encoder(v)
    val latent = reparametrize(mean, logVar, key)
    val reconstructed = decoder(latent)
    (reconstructed, mean, logVar)

  def loss(original: FTensor1[Height |*| Width], key: Random.Key): Tensor0[Float] =
    val (reconstructed, mean, logVar) = apply(original, key)
    val eps = Tensor0(1e-5f)
    val reconLoss = -((original * (reconstructed +! eps).log) + ((Tensor0(1f) -! original) * (Tensor0(1f) -! reconstructed +! eps).log)).sum
    val kldLoss = Tensor0(-0.5f) * (Tensor0(1f) +! logVar -! mean.pow(Tensor0(2f)) -! logVar.exp).sum

    reconLoss + kldLoss

object VAE:
  case class Params(
      encoderParams: Encoder.EncoderParams,
      decoderParams: Decoder.DecoderParams
  )
  object Params:
    def apply(params: VAE.Params)(key: Random.Key): Params =
      Params(
        params.encoderParams,
        params.decoderParams
      )

object VAEExample:
  @main
  def main(): Unit =

    val learningRate = 5e-4f

    val numTestSamples = 128 // 9728
    val batchSize = 256
    val numSamples = batchSize * 50 // 59904
    val numEpochs = 800
    val latentDim = 20

    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()

    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(numTestSamples)).get

    /*
     * Initialize the model parameters
     * */
    val initKeys = initKey.split(7)
    val encoderParams = Encoder.EncoderParams(
      LinearLayer.Params[Height |*| Width, EHidden1](initKeys(0))(
        Axis[Height |*| Width] -> (28 * 28),
        Axis[EHidden1] -> 512
      ),
      LinearLayer.Params[EHidden1, EHidden2](initKeys(1))(
        Axis[EHidden1] -> 512,
        Axis[EHidden2] -> 256
      ),
      LinearLayer.Params[EHidden2, MeanLatent](initKeys(2))(
        Axis[EHidden2] -> 256,
        Axis[MeanLatent] -> latentDim
      ),
      LinearLayer.Params[EHidden2, LogVarLatent](initKeys(3))(
        Axis[EHidden2] -> 256,
        Axis[LogVarLatent] -> latentDim
      )
    )
    val decoderParams = Decoder.DecoderParams(
      LinearLayer.Params[Latent, DHidden1](initKeys(4))(
        Axis[Latent] -> 20,
        Axis[DHidden1] -> 256
      ),
      LinearLayer.Params[DHidden1, DHidden2](initKeys(5))(
        Axis[DHidden1] -> 256,
        Axis[DHidden2] -> 512
      ),
      LinearLayer.Params[DHidden2, ReconstructedFeature](initKeys(6))(
        Axis[DHidden2] -> 512,
        Axis[ReconstructedFeature] -> (28 * 28)
      )
    )

    // we need to scale down the initial parameters for
    // better training stability.
    // TODO linear layer et al. should support custom initializers
    // or xavier initialization
    val initialParams = VAE.Params(encoderParams, decoderParams)
    val scaledInitialParams = FloatTensorTree[VAE.Params].map(
      initialParams,
      [T <: Tuple] => (n: Labels[T]) ?=> (t: Tensor[T, Float]) => t *! Tensor0(0.1f)
    )

    /*
     * Training loop
     * */

    def batchLoss(key: Random.Key, trainData: Tensor3[Sample, Height, Width, Float], params: VAE.Params): Tensor0[Float] =
      val vae = VAE(params)
      val keys = key.splitToTensor(Axis[Sample], trainData.shape.dim(Axis[Sample])._2)
      zipvmap(Axis[Sample])(trainData, keys) { (sample, key) =>
        vae.loss(sample.ravel, Key(key.jaxValue))
      }.mean

    val jitBatchLoss = jit(batchLoss)

    val batches = trainX.chunk(Axis[TrainSample], numSamples / batchSize)
    def trainBatchWithShuffle(key: Random.Key, batch: Tensor3[Sample, Height, Width, Float], params: VAE.Params): VAE.Params =
      val (shuffleKey, trainKey) = key.split2()
      val shuffledBatch = Random.shuffle(batch, Axis[Sample], shuffleKey)
      val df = Autodiff.grad(params => batchLoss(trainKey, shuffledBatch, params))
      GradientDescent(df, Tensor0(learningRate)).step(params)

    val jittedTrainBatch = jitUpdate(trainBatchWithShuffle)

    def trainBatch(key: Random.Key, trainData: Tensor3[Sample, Height, Width, Float])(initialParams: VAE.Params): VAE.Params =
      val trainedParams = jittedTrainBatch(key, trainData, initialParams)
      trainedParams

    def trainEpoch(key: Random.Key, epoch: Int, params: VAE.Params): VAE.Params =
      val batchKeys = key.split(batches.size)
      batches.zip(batchKeys).foldLeft(params) { case (batchParams, (batch, key)) =>
        jittedTrainBatch(key, batch, batchParams)
      }

    // run the loop
    val keysForEpochs = dataKey.split(numEpochs)
    val trainedParams = (0 until numEpochs).foldLeft(scaledInitialParams) { (params, epoch) =>

      if epoch % 100 == 0 then
        val lossValue = jitBatchLoss(keysForEpochs(epoch), testX, params)
        println(s" Test loss in epoch $epoch: $lossValue")
        dimwit.gc()

      dimwit.withLocalCleanup {
        trainEpoch(keysForEpochs(epoch), epoch, params)
      }
    }

    /*
     * Evaluation
     * */
    val vae = VAE(trainedParams)

    val reconstructed = testX.vmap(Axis[TestSample]) { sample =>
      val (mean, logVar) = vae.encoder(sample.ravel)
      val latent = reparametrize(mean, logVar, dataKey) // TODo Key management
      vae.decoder(latent)
    }
    // TODO Unstacking would be great here instead of slice
    (0 until 10).map { i =>
      val img = reconstructed.slice(Axis[TestSample] -> i)
      val img2d = img.rearrange(
        (Axis[Height], Axis[Width]),
        (Axis[Height] -> 28, Axis[Width] -> 28)
      )
      ImageVis.plotImage(img2d, s"./plots/vae_reconstructed-$i.html", "VAE Reconstructed Image")
    }

    /*
     * Sampling from the latent space
     */
    val stdNormal = Normal.standardNormal(Shape(Axis[Latent] -> latentDim))
    val sampled = dataKey.splitvmap(Axis[Sample], 10) { key =>
      val z = stdNormal.sample(key)
      vae.decoder(z)
    }
    (0 until sampled.shape.dim(Axis[Sample])._2).map { i =>
      val img = sampled.slice(Axis[Sample] -> i)
      val img2d = img.rearrange(
        (Axis[Height], Axis[Width]),
        (Axis[Height] -> 28, Axis[Width] -> 28)
      )
      ImageVis.plotImage(img2d, s"./plots/vae_sampled-$i.html", "VAE Sampled Image")
    }

object ImageVis:

  def plotImage(img: Tensor2[Height, Width, Float], filename: String, title: String): Unit =

    // import seq converters from scalapy
    import me.shadaj.scalapy.py
    import me.shadaj.scalapy.py.SeqConverters

    import scaltair.*
    import scaltair.PlotTargetFile.given

    val data = img.jaxValue.as[Seq[Seq[Float]]]
    val rows =
      for
        i <- 0 until data.size
        j <- 0 until data.head.size
      yield Map(
        "y" -> i,
        "x" -> j,
        "value" -> data(i)(j).toDouble
      )
    Chart(Data.fromRows(rows))
      .encode(
        Channel.X("x", FieldType.Ordinal),
        Channel.Y("y", FieldType.Ordinal),
        Channel.Color("value", FieldType.Quantitative)
      )
      .mark(Mark.Rect())
      .save(filename)
