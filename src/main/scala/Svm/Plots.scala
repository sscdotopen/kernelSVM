package Svm

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.plot._

object Plots {

  def concat(X: DenseMatrix[Double], filterFunction: (Int => Boolean)) = {
    (0 until X.cols).filter { filterFunction }
      .map { i => X(::, i).toDenseMatrix }
      .reduce { DenseMatrix.vertcat(_, _) }
  }

  def plotModel(X: DenseMatrix[Double], Y: DenseMatrix[Double], W: DenseVector[Double], sigma: Double): Unit ={

    val class0Predictions = concat(X, { i => KernelSVM.predictSVMKernel(X(::, i), X, W, sigma) <= 0 })
    val class1Prediction = concat(X, { i => KernelSVM.predictSVMKernel(X(::, i), X, W, sigma) > 0 })
    val class0_gt = concat(X, { i => Y(0, i) <= 0 })
    val class1_gt = concat(X, { i => Y(0, i) > 0 })

    val f = Figure()

    val predictionPlot = f.subplot(0)
    predictionPlot += plot(class0Predictions(::, 0), class0Predictions(::, 1), '.')
    predictionPlot += plot(class1Prediction(::, 0), class1Prediction(::, 1), '.')
    predictionPlot.title = "Prediction"

    val groundTruthPlot = f.subplot(2, 1, 1)
    groundTruthPlot += plot(class0_gt(::, 0), class0_gt(::, 1), '.')
    groundTruthPlot += plot(class1_gt(::, 0), class1_gt(::, 1), '.')
    groundTruthPlot.title = "Groundtruth"
  }

}
