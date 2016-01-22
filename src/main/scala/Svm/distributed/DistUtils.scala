package Svm.distributed

import breeze.linalg.{Vector, DenseVector, DenseMatrix}
import org.apache.spark.AccumulatorParam

import scala.util.Random
import scala.util.hashing.MurmurHash3

object DistUtils {

  object StringAccumulator extends AccumulatorParam[String] {

    override def addInPlace(r1: String, r2: String): String = {
      if (r1 != "" && r2 != "") { r1 + "\n" + r2 }
      else if (r1 == "") { r2 }
      else { r1 }
    }

    override def zero(initialValue: String): String = ""
  }

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

  def assignFromRangeVector(finalLength: Int, range: IndexedSeq[Int], values: Vector[Double]): Vector[Double] = {
    val vector = DenseVector.zeros[Double](finalLength)
    vector(range) := values
    vector
  }

  def sampleNoSeed(maxIndex: Int, howMany: Int) = {
    new Random().shuffle(Array.range(0, maxIndex).toIndexedSeq).take(howMany)
  }

  def sample(seed: Int, key: String, maxIndex: Int, howMany: Int) = {
    val prng = new Random(MurmurHash3.stringHash(key.toString, seed))
    prng.shuffle(Array.range(0, maxIndex).toIndexedSeq).take(howMany)
  }

  def sampleIndicesForPartition(seed: Int, partitionId: Int, maxIndex: Int, howMany: Int) = {
    sample(seed, partitionId.toString, maxIndex, howMany)
  }

  def partitionFromInstanceAssignments(partitionId: Int, sampledIndices: IndexedSeq[Int], numFeatures: Int,
      instanceAssignments: TraversableOnce[InstanceAssignment]): Partition = {

    val partitionSize = sampledIndices.length

    val X = DenseMatrix.zeros[Double](numFeatures, partitionSize)
    val Y = DenseVector.zeros[Double](partitionSize)

    //TODO we could do a consistency check here by looking at the sampledIndices
    instanceAssignments.foreach { assignment =>
      X(::, assignment.posInPartition) := assignment.instance.features
      Y(assignment.posInPartition) = assignment.instance.label
    }

    Partition(partitionId, sampledIndices, X, Y)
  }
}
