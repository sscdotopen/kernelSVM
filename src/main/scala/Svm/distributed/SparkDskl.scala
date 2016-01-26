package Svm.distributed

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import Svm.{Plots, Utils, Dskl}

import scala.io.Source
import scala.util.Random

case class Instance(id: Int, label: Int, features: DenseVector[Double])
case class InstanceAssignment(partitionId: Int, posInPartition: Int, instance: Instance)
case class Partition(id: Int, sampledIndices: IndexedSeq[Int], X: DenseMatrix[Double], Y: DenseVector[Double])

object SparkDskl extends App {

  learnXOR()
  //learnMNISToneVersusAll()

  def learnMNISToneVersusAll() = {

    val labelA = 3

    val data = Source.fromFile("/home/ssc/Entwicklung/datasets/infimnist/infimnist/mnist8m-libsvm-indexed-first10k.txt")
      .getLines()
      .map { line =>
        val instance = DistUtils.mnistInstanceFromString(line)
        val newLabel = if (instance.label == labelA) { 1 } else { -1 }
        Instance(instance.id, newLabel, instance.features)
      }.toSeq

    //TODO for some weird reason, don't remove this line...
    println(data.size)

    val numPartitions = 10

    val conf = new SparkConf()
    DsklKryo.addKryoSettings(conf)
    val sc = new SparkContext(s"local[${numPartitions}]", "DoublyStochasticKernelLearning", conf)

    try {
      val instances = sc.parallelize(data, numPartitions)

      val numInstances = data.size

      val α = learn(instances = instances, seed = Random.nextInt(), NBefore = numInstances, D = DistUtils.MNIST_NUM_FEATURES,
        C = 0.00001, numPartitions = numPartitions, partitionSize = 5000, numGradients = 100,
        empiricalKernelMapWidth = 1, iterations = 10, testEvery = 5, numHeldoutInstances = 0)
    } finally {
      sc.stop()
    }
  }


  def learnXOR() = {

    val N = 2000
    val numPartitions = 5
    
    val conf = new SparkConf()
    val sc = new SparkContext(s"local[${numPartitions}]", "DoublyStochasticKernelLearning", conf)

    try {
      val xorData = Utils.shuffleData(Utils.generateXORData(N, 0.2))
      val numHeldoutInstances = 200

      val instances = sc.parallelize(
        (0 until xorData.X.cols).map { i =>
          Instance(i, xorData.Y(0, i).toInt, xorData.X(::, i).toDenseVector)
        })

      val α = learn(instances = instances, seed = Random.nextInt(), NBefore = N, D = 2, C = 0.00001,
        numPartitions = numPartitions, partitionSize = 500, numGradients = 10, empiricalKernelMapWidth = 1,
        iterations = 100, testEvery = 25, numHeldoutInstances = numHeldoutInstances)

      Plots.plotModel(xorData.X(::, 0 until (N - numHeldoutInstances)), xorData.Y(::, 0 until (N - numHeldoutInstances)), α, sigma = 1.0)
      Plots.plotModel2(xorData.X(::, (N - numHeldoutInstances) until N), xorData.X(::, 0 until (N - numHeldoutInstances)), xorData.Y(::, (N - numHeldoutInstances) until N), α, sigma = 1.0)
    } finally {
      sc.stop()
    }
  }

  def learn(instances: RDD[Instance], seed: Int, NBefore: Int, D: Int, C: Double, numPartitions: Int, partitionSize: Int,
      numGradients: Int, empiricalKernelMapWidth: Int, iterations: Int, testEvery: Int, numHeldoutInstances: Int): DenseVector[Double] = {

    val N = NBefore - numHeldoutInstances

    /* initialize coefficient vector */
    var α = DenseVector.rand(N, Rand.gaussian)
    val G = DenseVector.ones[Double](N)

    val sc = instances.sparkContext

    val trainInstances = instances.filter { _.id < N }
    trainInstances.cache()

    val partitions = partitionsFromInstances(trainInstances, N, D, numPartitions, seed, partitionSize)
    partitions.cache()
    partitions.count()


    for (iteration <- 1 to iterations) {

      /* broadcast dual coefficients */
      val α_broadcast = sc.broadcast(α)

      val (g, deltaGk) = partitions
        .map { partition =>

          /* slice out relevant part of coefficient vector */
          val α_local: Vector[Double] = α_broadcast.value(partition.sampledIndices)

          val rnPred = DistUtils.sampleNoSeed(partition.Y.length, numGradients)
          val rnexpand = DistUtils.sampleNoSeed(partition.Y.length, empiricalKernelMapWidth)

          val gk_localAndGhat = gradientAndG(partition.Y(rnPred), partition.X(::, rnPred), partition.X(::, rnexpand),
            α_local(rnexpand).asInstanceOf[Vector[Double]], C)

          /* correctly translate sample partition indexes to global indexes */
          val translation = rnexpand.map { expandIndex => partition.sampledIndices(expandIndex) }

          val gk = SparseVector.zeros[Double](N)
          gk(translation) := gk_localAndGhat._1

          val deltaGk = SparseVector.zeros[Double](N)
          deltaGk(translation) := gk_localAndGhat._2

          (gk, deltaGk)
        }
        .treeReduce { case ((gk1, deltaGk1), (gk2, deltaGk2)) => (gk1 + gk2, deltaGk1 + deltaGk2) }

      α -= invSqrtG(G) :* g


      //α -= g * (1.0 / iteration)

      G :+= deltaGk
    }

    partitions.unpersist()


    α
  }



  def invSqrtG(G: DenseVector[Double]): Vector[Double] = {
    val invSqrtG = DenseVector.zeros[Double](G.length)
    var i = 0
    while (i < invSqrtG.length) {
      invSqrtG(i) = if (G(i) != 0) { 1.0 / sqrt(G(i)) } else { 0 }
      i += 1
    }
    invSqrtG
  }

  // def compute_gradient(y,Xpred,Xexpand,w,kernel,C):
  def gradientAndG(y: Vector[Double], Xpred: Matrix[Double], Xexpand: Matrix[Double], w: Vector[Double],
               C: Double): (Vector[Double], Vector[Double]) = {

    //K = kernel[0](Xpred,Xexpand,kernel[1])
    val K = Dskl.gaussianKernel(Xpred.toDenseMatrix, Xexpand.toDenseMatrix, sigma = 1.0)

    //yhat = K.dot(w)
    val yhat: Vector[Double] = K * w

    //# compute whether or not prediction is in margin
    //inmargin = (yhat * y) <= 1
    val yHatTimesY = yhat :* y
    val yTimesInMargin = DenseVector.zeros[Double](y.length)
    var i = 0
    while (i < yHatTimesY.length) {
      if (yHatTimesY(i) <= 1) {
        yTimesInMargin(i) = y(i)
      }
      i += 1
    }

    //G = C * w - (y * inmargin).dot(K)

    val CwPerPoint = ((w * C) :/ y.length.toDouble).t

    //(w * C) - yTimesInMarginDotK(0, ::).t

    // in-place updates of K!
    i = 0
    while (i < yTimesInMargin.length) {
      K(i, ::) *= yTimesInMargin(i)
      K(i, ::) := CwPerPoint - K(i, ::)
      i += 1
    }

    //val rowSumOfK = sum(K(::, *))
    //val gk = Cw - rowSumOfK(0, ::).t
    val gk = sum(K(::, *))

    val K2 = K :* K
    val Ghatk = sum(K2(::, *))

    (gk(0, ::).t, Ghatk(0, ::).t)
  }

  def partitionsFromInstances(instances: RDD[Instance], N: Int, D: Int, numPartitions: Int, seed: Int,
      partitionSize: Int) = {

    val sc = instances.sparkContext

    val assignments = (0 until numPartitions).par
      .map { partitionId =>
        val sampledIndices = DistUtils.sampleIndicesForPartition(seed, partitionId, N, partitionSize)
        partitionId -> sampledIndices.zipWithIndex.toMap
      }
      .seq

    println("Generating data matrices for partitions from assignments")
    val assignmentsBc = sc.broadcast(assignments)

    instances
      .flatMap { instance =>
        val assignments = assignmentsBc.value
        for ((partition, partitionAssignments) <- assignments;
             if (partitionAssignments.contains(instance.id))) yield {
          InstanceAssignment(partition, partitionAssignments(instance.id), instance)
        }
      }
      .groupBy { _.partitionId }
      .map { case (partitionId, instanceAssignments) =>
        val sampledIndices = DistUtils.sampleIndicesForPartition(seed, partitionId, N, partitionSize)
        DistUtils.partitionFromInstanceAssignments(partitionId, sampledIndices, D, instanceAssignments)
      }
  }
  
}
