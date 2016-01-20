package Svm.distributed

import org.junit.Test
import org.scalatest.junit.JUnitSuite

class SimpleTests extends JUnitSuite {

  @Test def instanceFromString(): Unit = {

    val instance = Utils.mnistInstanceFromString("1 7 13:255 267:128 318:12")

    assert(instance.id == 1)
    assert(instance.label == 7)
    assert(instance.pixels.length == Utils.MNIST_NUM_FEATURES)

    assert(instance.pixels(12) == 0.0)
    assert(instance.pixels(13) == 255.0)
    assert(instance.pixels(267) == 128.0)
    assert(instance.pixels(268) == 0.0)
    assert(instance.pixels(318) == 12.0)
  }

  @Test def sampleInstancesWithoutReplacement(): Unit = {

    val positionsAndIndexes = Utils.sampleIndicesForPartition(0xdead, 1, 100, 10)

    assert(positionsAndIndexes.toSeq.distinct.size == 10)
    assert(!positionsAndIndexes.exists { _ >= 100 })
  }

  @Test def partitionFromInstanceAssignments(): Unit = {

    val assignments = Array(
      InstanceAssignment(1, 0, Utils.mnistInstanceFromString("1 0 13:1 267:1 318:1")),
      InstanceAssignment(1, 2, Utils.mnistInstanceFromString("2 1 13:2 267:2 318:2")),
      InstanceAssignment(1, 1, Utils.mnistInstanceFromString("3 0 13:3 267:3 318:3"))
    )

    val partition = Utils.partitionFromInstanceAssignments(partitionId = 1, sampledIndices = Array(0, 2, 1),
      instanceAssignments = assignments)

    assert(partition.id == 1)
    assert(partition.X.rows == Utils.MNIST_NUM_FEATURES)
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
