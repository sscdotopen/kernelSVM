package Svm

import breeze.linalg.{DenseMatrix, DenseVector}

case class LabeledData(X: DenseMatrix[Double], Y: DenseMatrix[Double])
case class WeightsAndErrors(weights: DenseVector[Double], errors: DenseVector[Double])
case class TestResult(error: Double, numMisclassifiedPoints: Int)