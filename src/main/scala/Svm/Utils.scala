package Svm

import breeze.linalg._
import breeze.numerics.{ceil, sqrt, exp}
import breeze.plot._
import breeze.stats.distributions.MultivariateGaussian

import scala.collection.mutable.ListBuffer
import scala.util.Random
import breeze.plot._

object Utils {


/*
  def test_svm_multiclass(X: DenseMatrix[Double], Y: DenseMatrix[Double], X_model: ListBuffer[DenseMatrix[Double]], Ws: ListBuffer[DenseVector[Double]],sigma: Double): Double = {
    var N: Int = X.cols
    var point_error: Int = 0

    var error: Double = 0.0d

    var eta_update: Int = ceil(X.cols/10.0).toInt
    var tstart = System.nanoTime()

    var results: ListBuffer[Double] = ListBuffer[Double]()
    for(point <- 0 until X.cols){
      // test all classifiers
      var confidences: ListBuffer[Double] = ListBuffer[Double]()
      for(c <- 0 until Ws.length){
        // get confidence
        var yhat: Double = predict_svm_kernel_all(X(::,point),X_model(c),Ws(c),sigma)
        // how should this classifier label this thing?
        confidences += yhat
      }
      // vote!
      val max = confidences.max
      val index = confidences.indexOf(max)

      if(index != Y(0,point)){
        error += 1
      }
      if(point%eta_update == 0){
        var tend = System.nanoTime()
        println("estimated time left:" + (((tend - tstart)/1000000000.0)/point.toDouble) * (X.cols - point) + "s")
      }
    }
    error/N.toDouble
  }
*/


  def plotLine(errors: DenseVector[Double], testInterval: Int): Unit = {
    val f = Figure()
    val p = f.subplot(0)

    val iterations = convert(DenseVector.range(0, errors.length * testInterval, testInterval), Double)
    p += plot(iterations, errors, '-')
    p.xlabel = "# iterations"
    p.ylabel = "error"
  }

/*  def plotData(X: DenseMatrix[Double]): Unit= {
    assert(X.rows == 2)
    val f = Figure()
    val p = f.subplot(0)
    p += plot(X(0,::).t, X(1,::).t, '.')
    p.xlabel = "x axis"
    p.ylabel = "y axis"
  }*/



  def generateXORData(N: Int = 80, noise: Double = 0.25) = {

    assert(N%4 == 0,"please use multiple of 4 for number of random samples")
    val mu = DenseMatrix((-1.0,1.0), (1.0,1.0)).t
    val C = DenseMatrix.eye[Double](2) * noise

    // sample and convert to matrix
    val samples1 = MultivariateGaussian(mu(::, 0), C).sample(N / 4)
    val samples2 = MultivariateGaussian(-mu(::, 0), C).sample(N / 4)
    val samples3 = MultivariateGaussian(mu(::, 1), C).sample(N / 4)
    val samples4 = MultivariateGaussian(-mu(::, 1), C).sample(N / 4)

    // concat matrices TODO: this probably can be done smarter
    //X = sp.hstack((mvn(mu[:,0],C,N/4).T,mvn(-mu[:,0],C,N/4).T, mvn(mu[:,1],C,N/4).T,mvn(-mu[:,1],C,N/4).T))

    //should be N
    val totalSamplesLength = samples1.length + samples2.length + samples3.length + samples4.length
    val X = DenseMatrix.zeros[Double](2, totalSamplesLength)

    var numSamplesProcessed = 0
    for (i <- 0  until samples1.length){
      X(::,i) := samples1(i)
    }
    numSamplesProcessed += samples1.length

    for (i <- 0  until samples2.length) {
      X(::, i + numSamplesProcessed) := samples2(i)
    }
    numSamplesProcessed += samples2.length

    for (i <- 0  until samples3.length) {
      X(::, i + numSamplesProcessed) := samples3(i)
    }

    numSamplesProcessed += samples3.length
    for (i <- 0  until samples4.length) {
      X(::, i + numSamplesProcessed) := samples4(i)
    }

    // Y = sp.hstack((sp.ones((1,N/2.)),-sp.ones((1,N/2.))))
    val ones = DenseMatrix.ones[Double](1, N / 2)
    val Y = DenseMatrix.horzcat(ones.copy, -ones.copy)

    LabeledData(X, Y)
  }


  /** Fisher-Yates shuffle */
  def shuffleData(labeledData: LabeledData): LabeledData = {
    assert(labeledData.X.cols == labeledData.Y.cols, "feature matrix and target vector have different sizes!!")

    val originalX = labeledData.X
    val originalY = labeledData.Y

    val colsEnumeration = DenseVector.range(0, labeledData.X.cols, 1)
    val permutation = shuffle(colsEnumeration)

    val Y = DenseMatrix.zeros[Double](1, originalX.cols)
    val X = DenseMatrix.zeros[Double](originalX.rows, originalY.cols)

    for (i <- 0 until originalX.cols) {
      // copy values from permutation
      Y(::,i) := originalY(::, permutation(i))
      X(::,i) := originalX(::, permutation(i))
    }
    
    LabeledData(X, Y)
  }
}

