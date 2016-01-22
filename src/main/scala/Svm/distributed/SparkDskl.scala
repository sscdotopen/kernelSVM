package Svm.distributed

import breeze.linalg._
import breeze.stats.distributions.Rand
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import Svm.{Plots, Utils, Dskl}

import scala.util.Random

case class Instance(id: Int, label: Int, features: DenseVector[Double])
case class InstanceAssignment(partitionId: Int, posInPartition: Int, instance: Instance)
case class Partition(id: Int, sampledIndices: IndexedSeq[Int], X: DenseMatrix[Double], Y: DenseVector[Double])

object SparkDskl extends App {

  learnXOR()

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
        iterations = 20)

    Plots.plotModel(xorData.X, xorData.Y, α, sigma = 1.0)
  }
  
  def learn(instances: RDD[Instance], seed: Int, N: Int, D: Int, C: Double, numPartitions: Int, partitionSize: Int, 
      numGradients: Int, empiricalKernelMapWidth: Int, iterations: Int): DenseVector[Double] = {

    /* initialize coefficient vector */
    var α = DenseVector.rand(N, Rand.gaussian)
    val sc = instances.sparkContext
    
    try {
      val partitions = partitionsFromInstances(instances, N, D, numPartitions, seed, partitionSize)
      partitions.cache()
      partitions.count()

      val convergenceStats = sc.accumulator("")(DistUtils.StringAccumulator)

      for (iteration <- 1 to iterations) {

        /* broadcast dual coefficients */
        val α_broadcast = sc.broadcast(α)

        val g = partitions
          .map { partition =>
  
            /* slice out relevant part of coefficient vector */
            val α_local = α_broadcast.value(partition.sampledIndices).toDenseVector
  
            val rnPred = DistUtils.sampleNoSeed(partition.Y.length, numGradients)
            val rnexpand = DistUtils.sampleNoSeed(partition.Y.length, empiricalKernelMapWidth)
            val gk_local = Dskl.gradient(partition.Y(rnPred), partition.X(::, rnPred), partition.X(::, rnexpand),
              α_local(rnexpand).asInstanceOf[Vector[Double]], C)

            /* compute training error on current partition */
            if (iteration % 3 == 0) {
              val predictions = predictSvmEmp(partition.X, partition.X, α_local)
              var incorrect = 0
              for (n <- 0 until predictions.length) {
                if ((partition.Y(n) * predictions(n)) < 0) {
                  incorrect += 1
                }
              }
              convergenceStats.add(s"iteration ${iteration}, partition ${partition.id}," +
                s"# incorrectly classified ${incorrect} / ${partitionSize} (on training data)")
            }
  
            /* correctly translate sample partition indexes to global indexes */
            val gk = DenseVector.zeros[Double](N)
            val translation = rnexpand.map { expandIndex => partition.sampledIndices(expandIndex) }
  
            gk(translation) := gk_local
            gk
          }
          .treeReduce { _ + _ }

        α -= (1.0 / iteration) * g
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
  
  def predictSvmEmp(Xexpand: DenseMatrix[Double], Xtarget: DenseMatrix[Double], α: DenseVector[Double]) = {
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
