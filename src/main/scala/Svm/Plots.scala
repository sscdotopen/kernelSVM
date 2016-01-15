package Svm

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.plot._

object Plots {

  def plotModel(X: DenseMatrix[Double], Y: DenseMatrix[Double], W: DenseVector[Double], sigma: Double): Unit ={
    // classify all points:
    var class0: DenseMatrix[Double] = null
    var class1: DenseMatrix[Double] = null

    var class0_gt: DenseMatrix[Double] = null
    var class1_gt: DenseMatrix[Double] = null

    for (i <- 0 until X.cols) {
      // compute prediction
      val prediction = KernelSVM.predictSVMKernel(X(::, i), X, W, sigma)

      if (prediction < 0) {
        if (class0 != null) {
          class0 = DenseMatrix.horzcat(class0, X(::, i).toDenseMatrix.t)
        } else {
          class0 = X(::, i).toDenseMatrix.t
        }
      } else {
        if (class1 != null) {
          class1 = DenseMatrix.horzcat(class1, X(::, i).toDenseMatrix.t )
        } else {
          class1 = X(::, i).toDenseMatrix.t
        }
      }

      if (Y(0, i) <= 0) {
        if (class0_gt != null) {
          class0_gt = DenseMatrix.horzcat(class0_gt, X(::, i).toDenseMatrix.t)
        } else {
          class0_gt = X(::,i).toDenseMatrix.t
        }
      } else {
        if (class1_gt != null) {
          class1_gt = DenseMatrix.horzcat(class1_gt, X(::, i).toDenseMatrix.t)
        } else {
          class1_gt = X(::, i).toDenseMatrix.t
        }
      }
    }


    val f = Figure()

    val predictionPlot = f.subplot(0)
    predictionPlot += plot(class0(0, ::).t, class0(1, ::).t,'.')
    predictionPlot += plot(class1(0, ::).t, class1(1, ::).t,'.')
    predictionPlot.title = "Prediction"

    val groundTruthPlot = f.subplot(2,1,1)
    groundTruthPlot += plot(class0_gt(0,::).t, class0_gt(1,::).t, '.')
    groundTruthPlot += plot(class1_gt(0,::).t, class1_gt(1,::).t, '.')
    groundTruthPlot.title = "Groundtruth"
  }

}
