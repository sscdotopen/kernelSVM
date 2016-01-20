package Svm.distributed

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions.Rand
import org.apache.spark.{SparkConf, SparkContext}
import Svm.DoublyStochasticKernelLearningEmpirical

case class Instance(id: Int, label: Int, pixels: DenseVector[Double])
case class InstanceAssignment(partitionId: Int, posInPartition: Int, instance: Instance)
case class Partition(id: Int, sampledIndices: IndexedSeq[Int], X: DenseMatrix[Double], Y: DenseVector[Double])

//TODO println statements should be changed to log statements
object DoublyStochasticKernelLearningOnSpark extends App {

  val inputFile = "/home/ssc/Entwicklung/datasets/infimnist/infimnist/mnist8m-libsvm-indexed-first10k.txt"

  val numPartitions = 4
  val seed = 0xbeef
  val N = 10000
  val partitionSize = 1000

  val conf = new SparkConf()
  val sc = new SparkContext(s"local[${numPartitions}]", "DoublyStochasticKernelLearning", conf)

  try {

    val instances =
      sc.textFile(inputFile, numPartitions)
        .map { Utils.mnistInstanceFromString }

    //TODO update labels to make this a binary classification problem
    //TODO normalize data?

    println("Generating partition assignments")
    val assignments = (0 until numPartitions).map { partitionId =>
      val sampledIndices = Utils.sampleIndicesForPartition(seed, partitionId, N, partitionSize)
      partitionId -> sampledIndices.zipWithIndex.toMap
    }

    println("Generating data matrices for partitions from assignments")
    val assignmentsBc = sc.broadcast(assignments)

    val partitions = instances
      .flatMap { instance =>
        val assignments = assignmentsBc.value
        for ((partition, partitionAssignments) <- assignments;
          if (partitionAssignments.contains(instance.id))) yield {
          InstanceAssignment(partition, partitionAssignments(instance.id), instance)
        }
      }
      .groupBy { _.partitionId }
      .map { case (partitionId, instanceAssignments) =>
        val sampledIndices = Utils.sampleIndicesForPartition(seed, partitionId, N, partitionSize)
        Utils.partitionFromInstanceAssignments(partitionId, sampledIndices, instanceAssignments)
      }

    partitions.cache()
    // force execution
    partitions.count()


    val C = 0.1
    val eta = 1.0
    val nPredSamples: Int = 10
    val nExpandSamples: Int = 100
    val outerIterations = 10
    val innerIterations = 10

    var W = DenseVector.rand(N, Rand.gaussian)

    //TODO measure time per iteration
    for (outerIteration <- 0 until outerIterations) {

      println(s"Executing outer iteration ${outerIteration}")

      val WBc = sc.broadcast(W)

      val nextW = partitions
        .map { partition =>

          val W = WBc.value

          var subW = W(partition.sampledIndices).toDenseVector

          var innerIteration = 0
          while (innerIteration < innerIterations) {

            println(s"\tExecuting inner iteration ${innerIteration} on partition ${partition.id}")

            //TODO is this correct?
            val currentEta = eta / (outerIteration * innerIteration)

            subW = DoublyStochasticKernelLearningEmpirical
              .stepDsklEmpirical(partition.X, partition.Y, subW, currentEta, C, nPredSamples, nExpandSamples)

            innerIteration += 1
          }

          //TODO we should do some testing to get a picture of the convergence on this partition

          //TODO the actual aggregation is currently not implemented, we need to change W
          //TODO and smartly aggregate it, maybe this requires us to keep track of our updates
          //TODO we return the unmodified W to just test whether the job executes correctly
          W
        }
        .treeReduce { _ + _ }

      W = nextW
    }

    partitions.unpersist()

  } finally {
    sc.stop()
  }
}
