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

  //learnXOR()
  learnMNIST()

  val numHeldoutInstances = 100

  def learnMNIST() = {

    var id = 0

    val labelA = 3
    val labelB = 6

    val data = Source.fromFile("/home/ssc/Entwicklung/datasets/infimnist/infimnist/mnist8m-libsvm-indexed-first10k.txt")
      .getLines()
      .flatMap { line =>
        val instance = DistUtils.mnistInstanceFromString(line)
        if (instance.label == labelA || instance.label == labelB) {
          val newLabel = if (instance.label == labelA) { 1 } else { -1 }
          val newInstance = Instance(id, newLabel, instance.features)
          id += 1 //map with side effect, dirty...
          Some(newInstance)
        } else {
          None
        }
      }.toSeq

    val numPartitions = 4

    val conf = new SparkConf()
    val sc = new SparkContext(s"local[${numPartitions}]", "DoublyStochasticKernelLearning", conf)

    val instances = sc.parallelize(data, numPartitions)

    val numInstances = id

    val α = learn(instances = instances, seed = Random.nextInt(), N = numInstances, D = DistUtils.MNIST_NUM_FEATURES,
      C = 0.5, numPartitions = numPartitions, partitionSize = 1000, numGradients = 100,
      empiricalKernelMapWidth = 100, iterations = 150, testEvery = 30)
  }


  def learnXOR() = {

    val N = 2000
    val numPartitions = 10
    
    val conf = new SparkConf()
    val sc = new SparkContext(s"local[${numPartitions}]", "DoublyStochasticKernelLearning", conf)
    
    
    val xorData = Utils.shuffleData(Utils.generateXORData(N, 0.2))

    val instances = sc.parallelize(
      (0 until xorData.X.cols).map { i =>
        Instance(i, xorData.Y(0, i).toInt, xorData.X(::, i).toDenseVector)
      })
    
    val α = learn(instances = instances, seed = Random.nextInt(), N = N, D = 2, C = 0.003,
        numPartitions = numPartitions, partitionSize = 500, numGradients = 10, empiricalKernelMapWidth = 10,
        iterations = 20, testEvery = 3)

    Plots.plotModel(xorData.X, xorData.Y, α, sigma = 1.0)
  }
  
  def learn(instances: RDD[Instance], seed: Int, N: Int, D: Int, C: Double, numPartitions: Int, partitionSize: Int, 
      numGradients: Int, empiricalKernelMapWidth: Int, iterations: Int, testEvery: Int): DenseVector[Double] = {

    /* initialize coefficient vector */
    var α = DenseVector.rand(N, Rand.gaussian)
    val G = DenseVector.ones[Double](N)

    val sc = instances.sparkContext
    
    try {
      val partitions = partitionsFromInstances(instances, N, D, numPartitions, seed, partitionSize)
      partitions.cache()
      partitions.count()

      val convergenceStats = sc.accumulator("")(DistUtils.StringAccumulator)

      for (iteration <- 1 to iterations) {

        /* broadcast dual coefficients */
        val α_broadcast = sc.broadcast(α)

        val (g, deltaGk) = partitions
          .map { partition =>
  
            /* slice out relevant part of coefficient vector */
            val α_local = α_broadcast.value(partition.sampledIndices).toDenseVector

            /* compute training error on current partition */
            if (iteration % testEvery == 0) {
              convergenceStats.add(trainError(iteration, partition, α_local))
            }

            val rnPred = DistUtils.sampleNoSeed(partition.Y.length, numGradients)
            val rnexpand = DistUtils.sampleNoSeed(partition.Y.length, empiricalKernelMapWidth)

            val gk_localAndGhat = gradientAndG(partition.Y(rnPred), partition.X(::, rnPred), partition.X(::, rnexpand),
              α_local(rnexpand).asInstanceOf[Vector[Double]], C)

            /* correctly translate sample partition indexes to global indexes */
            val translation = rnexpand.map { expandIndex => partition.sampledIndices(expandIndex) }

            val gk = DenseVector.zeros[Double](N)
            gk(translation) := gk_localAndGhat._1

            val deltaGk = DenseVector.zeros[Double](N)
            deltaGk(translation) := gk_localAndGhat._2

            (gk, deltaGk)
          }
          .treeReduce { case ((gk1, deltaGk1), (gk2, deltaGk2)) => (gk1 + gk2, deltaGk1 + deltaGk2) }

        α -= invSqrtG(G) :* g
        
        G :+= deltaGk
      }

      partitions.unpersist()

      println("########################")
      println(convergenceStats.value)
      println("########################")
    } finally {
      sc.stop()
    }

    α
  }

  def trainError(iteration: Int, partition: Partition, α_local: Vector[Double]): String = {
    /* compute training error on current partition */
    val testStart = System.currentTimeMillis()
    val rnTestPred = DistUtils.sampleNoSeed(partition.Y.length, 100)
    val rnTestExpand = DistUtils.sampleNoSeed(partition.Y.length, 1000)
    val predictions = predictSvmEmp(partition.X(::, rnTestPred).toDenseMatrix,
      partition.X(::, rnTestExpand).toDenseMatrix, α_local(rnTestExpand).asInstanceOf[Vector[Double]])

    var incorrectA = 0
    var incorrectB = 0
    val correctLabels = partition.Y(rnTestPred)
    for (n <- 0 until predictions.length) {
      if ((correctLabels(n) * predictions(n)) < 0) {
        if (correctLabels(n) == 1) {
          incorrectA += 1
        } else {
          incorrectB += 1
        }
      }
    }
    val testDurationInMs = System.currentTimeMillis() - testStart
    s"iteration ${iteration}, partition ${partition.id}, " +
      s"# incorrectly classified ${incorrectA + incorrectB} / ${predictions.length} [${incorrectA} / ${incorrectB}] " +
      s"(on training data, took ${testDurationInMs}} ms)"
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

  def predictSvmEmp(Xexpand: DenseMatrix[Double], Xtarget: DenseMatrix[Double], α: Vector[Double]): Vector[Double] = {
    Dskl.gaussianKernel(Xexpand, Xtarget, sigma = 1.0) * α
  }

  def partitionsFromInstances(instances: RDD[Instance], N: Int, D: Int, numPartitions: Int, seed: Int,
      partitionSize: Int) = {

    val sc = instances.sparkContext

    println("Generating partition assignments")
    val assignments = (0 until numPartitions).map { partitionId =>
      val sampledIndices = DistUtils.sampleIndicesForPartition(seed, partitionId, N, partitionSize)
      partitionId -> sampledIndices.zipWithIndex.toMap
    }

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
