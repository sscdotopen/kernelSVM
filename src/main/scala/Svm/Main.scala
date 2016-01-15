package Svm

import breeze.numerics.{abs, sqrt}
import org.apache.commons.math3.linear.RealMatrix
import breeze.linalg._

import breeze.plot._

import scala.collection.mutable.ListBuffer

object Main extends App {

    trainXOR()
//    train_mnist()



  def trainXOR(): Unit = {

    val labeledData = Utils.shuffleData(Utils.generateXORData(1000, 0.25))

    val N = labeledData.X.cols
    val W = DenseVector.rand[Double](N)

    val testInterval = 100

    val result = KernelSVM.fitRandomPointCompleteKernel(W, labeledData.X, labeledData.Y, iterations = 2000, eta = 1.0,
      C = 10.0, testInterval = testInterval)

    assert(result.errors != null && result.weights != null, "check your privileges!")

    //Utils.plotData(labeledData.X)
    Utils.plotLine(result.errors, testInterval)
    Plots.plotModel(labeledData.X, labeledData.Y, result.weights, sigma = 1.0)
  }
/*
  def train_mnist(): Unit ={
    //    var filepathtrain = "/media/owner/extension/mnist_m8/infimnist/mnist_std/test-libsvm"
    var filepathtrain = "/media/owner/extension/mnist_m8/infimnist/mnist_std/mnist60k-libsvm"
//    var filepathtrain = "/media/owner/extension/mnist_m8/infimnist/mnist_std/mnist-classes01-libsvm"
//    var filepathtrain = "/media/owner/extension/mnist_m8/infimnist/mnist_std/mnist-classes01-every10th-libsvm"
    var res = UtilsDist.loadMnist(filepathtrain)
    var X = res._1
    var Y_all = res._2

    var filepathtest = "/media/owner/extension/mnist_m8/infimnist/mnist_std/test10k-libsvm"
    var res_test = UtilsDist.loadMnist(filepathtest)
    var X_test = res_test._1
    var Y_all_test = res_test._2


    // create -1,1 training sets for each class
    var Ys: ListBuffer[DenseMatrix[Double]] = UtilsDist.createOneVsAllTrainingsets(Y_all)

    var Y: DenseMatrix[Double] = Ys(0)
    var N = X.cols

    var num_classes = Ys.length
    println("training data prepared:")
    println("number of classes=" + num_classes)
    println("inputs:")
    println("X.cols=" + X.cols)
    println("X.rows=" + X.rows)

    var ws: ListBuffer[DenseVector[Double]] = ListBuffer[DenseVector[Double]]()

    // TODO: find the right parameters?
    // http://arxiv.org/pdf/1407.5599.pdf
    // use sigma = 9.03, regularizer 1/n (n = num training samples)
    // sklearn example: var sigma = 44.0
    var sigma = 9.03 // sklearn example: sqrt(2/0.001)
    var C = 1/X.cols.toDouble
    println("running with parameters:")
    println("sigma = " + sigma + " C = " + C)

    var test_interval = 0
    for (i <- 0 until num_classes){
      println("training oneVsAll SVM " + (i + 1) + " of " + num_classes)
      var W: DenseVector[Double] = DenseVector.rand(N)
      var iterations = 300
      var res3 = Utils.fitKernelSVM(W, X, Ys(i), iterations = iterations, eta = 1.0 , C = C, sigma = sigma ,testInterval = test_interval)
      var errors = res3._2
      var w = res3._1
      ws += w
      if(test_interval != 0){
        Utils.plotLine(errors)
      }
    }

    // select support vectors
    // TODO: how to do this best ?
    var X_supportvectors: ListBuffer[DenseMatrix[Double]] = ListBuffer[DenseMatrix[Double]]()
    var ws_supportvectors: ListBuffer[DenseVector[Double]] = ListBuffer[DenseVector[Double]]()
    var reduced = 0
    var full = 0
    for(c <- 0 until num_classes){
      // percentile to keep
      var percentile = 0.01
      var maxi: Double = max(ws(c).data)
      var mini: Double = min(ws(c).data)
      var diff: Double = abs(maxi - mini) * percentile
      var maxi_lower: Double = maxi - diff
      var mini_upper: Double = mini + diff
      var current_weights = ws(c)
      var aux_listbufferpoints: ListBuffer[DenseVector[Double]] = ListBuffer[DenseVector[Double]]()
      var aux_listbufferweights: ListBuffer[Double] = ListBuffer[Double]()
      full += current_weights.length
      for(p <- 0 until current_weights.length){
        if((current_weights(p) <= mini_upper) || (current_weights(p) >= maxi_lower)){
          aux_listbufferpoints += X(::,p)
          aux_listbufferweights += current_weights(p)
        }
      }
      // convert back to matrix and vector
      var new_weights = DenseVector.zeros[Double](aux_listbufferweights.size)
      var new_points = DenseMatrix.zeros[Double](X.rows,aux_listbufferweights.size)
      for(p <- 0 until aux_listbufferweights.size){
        new_weights(p) = aux_listbufferweights(p)
        new_points(::,p) := aux_listbufferpoints(p)
      }
      reduced += aux_listbufferweights.size
      X_supportvectors += new_points
      ws_supportvectors += new_weights
    }
    println("reduced size of models from " + full + " to " + reduced + " thats: " + (100.0 - reduced/full.toDouble * 100.0) + " %" )

    // run all the svms on the test data, write out confidence, vote
    println("testing on test set:")
    var error = Utils.test_svm_multiclass(X_test, Y_all_test, X_supportvectors, ws_supportvectors, sigma)
    println("final error:" + error)
  }
*/

  def plotasImage(x: DenseMatrix[Double]):Unit= {

    // mnist should be 28x28 pixel files
    // we'll try this as well.
    var img = DenseMatrix.zeros[Double](28,28)
    var idx = 0
    for(x_ <- 0 until 28){
      for(y_ <- 0 until 28){
        if(idx < x.rows){
          print(x(idx,0) + " ")
          img(x_,y_) = x(idx,0)
        }
        idx += 1
      }
      println()
    }

    val f = Figure()
    f.subplot(0) += image(img)
  }
}
