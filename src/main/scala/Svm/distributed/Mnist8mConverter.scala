package Svm.distributed

import java.io.{FileWriter, BufferedWriter}
import scala.io.Source


object Mnist8mConverter extends App {

  //./infimnist svm 10000 8109999 > mnist8m-libsvm.txt

  val writer =
    new BufferedWriter(new FileWriter("/home/ssc/Entwicklung/datasets/infimnist/infimnist/mnist8m-libsvm-indexed.txt"))

  try {
    var index = 0
    Source.fromFile("/home/ssc/Entwicklung/datasets/infimnist/infimnist/mnist8m-libsvm.txt").getLines.foreach { line =>

      writer.write(index.toString)
      writer.write(" ")
      writer.write(line)
      writer.newLine()

      index += 1

      if (index % 250000 == 0) {
        println(s"${index} lines read")
      }
    }
  } finally {
    writer.close()
  }

}
