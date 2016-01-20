package Svm.distributed

import breeze.linalg.{DenseVector, DenseMatrix}

import scala.util.Random
import scala.util.hashing.MurmurHash3

object Utils {

  val MNIST_NUM_FEATURES = 28 * 28

  def mnistInstanceFromString(str: String): Instance = {
    val parts = str.split(" ")
    val id = parts(0).toInt
    val label = parts(1).toInt
    val pixels = DenseVector.zeros[Double](28 * 28)

    var i = 2
    while (i < parts.length) {
      val tokens = parts(i).split(":")
      pixels(tokens(0).toInt) = tokens(1).toDouble
      i += 1
    }
    Instance(id, label, pixels)
  }
  
  def sampleIndicesForPartition(seed: Int, partitionId: Int, maxIndex: Int, howMany: Int) = {
    val prng = new Random(MurmurHash3.stringHash(partitionId.toString, seed))
    prng.shuffle(Array.range(0, maxIndex).toIndexedSeq).take(howMany)
  }

  def partitionFromInstanceAssignments(partitionId: Int, sampledIndices: IndexedSeq[Int],
      instanceAssignments: TraversableOnce[InstanceAssignment]): Partition = {

    val partitionSize = sampledIndices.length

    val X = DenseMatrix.zeros[Double](MNIST_NUM_FEATURES, partitionSize)
    val Y = DenseVector.zeros[Double](partitionSize)

    //TODO we could do a consistency check here by looking at the sampledIndices
    instanceAssignments.foreach { assignment =>
      X(::, assignment.posInPartition) := assignment.instance.pixels
      Y(assignment.posInPartition) = assignment.instance.label
    }

    Partition(partitionId, sampledIndices, X, Y)
  }
}
