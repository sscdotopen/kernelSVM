package Svm.distributed

import breeze.linalg.DenseVector
import org.junit.Test
import org.scalatest.junit.JUnitSuite

class SimpleTests extends JUnitSuite {

  @Test def instanceFromString(): Unit = {

    val instance = DistUtils.mnistInstanceFromString("1 7 13:255 267:128 318:12")

    assert(instance.id == 1)
    assert(instance.label == 7)
    assert(instance.features.length == DistUtils.MNIST_NUM_FEATURES)

    assert(instance.features(12) == 0.0)
    assert(instance.features(13) == 255.0)
    assert(instance.features(267) == 128.0)
    assert(instance.features(268) == 0.0)
    assert(instance.features(318) == 12.0)
  }

  @Test def assignFromRangeVector(): Unit = {
    val range = Array(1, 4, 2).toIndexedSeq
    val small = DenseVector(1.0, 4.0, 2.0)
    val full = DistUtils.assignFromRangeVector(6, range, small)

    assert(full(0) == 0.0)
    assert(full(1) == 1.0)
    assert(full(2) == 2.0)
    assert(full(3) == 0.0)
    assert(full(4) == 4.0)
    assert(full(5) == 0.0)
  }

  @Test def sampleRepeatable(): Unit = {
    val sample1 = DistUtils.sample(0xbeef, "asdf", 1000, 300).toArray
    val sample2 = DistUtils.sample(0xbeef, "asdf", 1000, 300).toArray
    val sample3 = DistUtils.sample(0xbeef, "asdf", 1000, 300).toArray

    assert(sample1.deep == sample2.deep)
    assert(sample2.deep == sample3.deep)
  }

  @Test def sampleInstancesWithoutReplacement(): Unit = {

    val positionsAndIndexes = DistUtils.sampleIndicesForPartition(0xdead, 1, 100, 10)

    assert(positionsAndIndexes.toSeq.distinct.size == 10)
    assert(!positionsAndIndexes.exists { _ >= 100 })
  }

  @Test def partitionFromInstanceAssignments(): Unit = {

    val assignments = Array(
      InstanceAssignment(1, 0, DistUtils.mnistInstanceFromString("1 0 13:1 267:1 318:1")),
      InstanceAssignment(1, 2, DistUtils.mnistInstanceFromString("2 1 13:2 267:2 318:2")),
      InstanceAssignment(1, 1, DistUtils.mnistInstanceFromString("3 0 13:3 267:3 318:3"))
    )

    val partition = DistUtils.partitionFromInstanceAssignments(partitionId = 1, sampledIndices = Array(0, 2, 1),
      numFeatures = DistUtils.MNIST_NUM_FEATURES, instanceAssignments = assignments)

    assert(partition.id == 1)
    assert(partition.X.rows == DistUtils.MNIST_NUM_FEATURES)
    assert(partition.X.cols == assignments.length)
    assert(partition.Y.length == assignments.length)

    assert(partition.Y(0) == 0)
    assert(partition.Y(1) == 0)
    assert(partition.Y(2) == 1)


    assert(partition.X(13, 0) == 1)
    assert(partition.X(13, 1) == 3)
    assert(partition.X(13, 2) == 2)
  }

}
