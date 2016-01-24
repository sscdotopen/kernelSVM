package Svm

import Svm.distributed.DistUtils
import breeze.numerics._
import breeze.stats.distributions.Rand

import breeze.linalg.{Matrix, Vector, DenseVector, DenseMatrix}


object Dskl {

  // def fit_svm_dskl_emp(X,Y,Xtest,Ytest,its=100,eta=1.,C=.1,nPredSamples=10,nExpandSamples=10, kernel=(GaussianKernel,(1.))):
  def fitSvmDsklEmp(X: DenseMatrix[Double], Y: DenseVector[Double], Xtest: DenseMatrix[Double],
    Ytest: DenseVector[Double], its: Int = 200, eta: Double = 1.0, C: Double = 0.1, nPredSamples: Int = 10,
    nExpandSamples: Int = 10) = {

    // Wemp = sp.randn(len(Y))
    var Wemp = DenseVector.rand(Y.length, Rand.gaussian)

    var it = 1
    while (it <= its) {

      //Wemp = step_dskl_empirical(X,Y,Wemp,eta/it,C,kernel,nPredSamples,nExpandSamples)
      Wemp = stepDsklEmpirical(X, Y, Wemp, eta / it, C, nPredSamples, nExpandSamples)

      assert(Wemp.length == Y.length)

      if (it % 20 == 0) {
        //TODO Eemp.append(sp.mean(Ytest != sp.sign(predict_svm_emp(X,Xtest,Wemp,kernel))))
        val predictions = predictSvmEmp(X, Xtest, Wemp)
        var incorrect = 0
        for (n <- 0 until predictions.length) {
          if ((Ytest(n) * predictions(n)) < 0) {
            incorrect += 1
          }
        }
        println(Wemp)
        println(s"Iteration ${it}, # incorrectly classified ${incorrect} / ${Xtest.cols}")
      }
      it += 1
    }

    Wemp
  }

  //def predict_svm_emp(Xexpand,Xtarget,w,kernel):
  def predictSvmEmp(Xexpand: DenseMatrix[Double], Xtarget: DenseMatrix[Double], w: DenseVector[Double]) = {
    //return w.dot(kernel[0](Xexpand,Xtarget,kernel[1]))
    gaussianKernel(Xexpand, Xtarget, sigma = 1.0) * w
  }

  // def step_dskl_empirical(X,Y,W,eta=1.,C=.1,kernel=(GaussianKernel,(1.)),nPredSamples=10,nExpandSamples=10):
  def stepDsklEmpirical(X: DenseMatrix[Double], Y: DenseVector[Double], W: DenseVector[Double], eta: Double = 1.0,
    C: Double = 1.0, nPredSamples: Int = 10, nExpandSamples: Int = 10) = {

    //rnpred = sp.random.randint(low=0,high=X.shape[1],size=nPredSamples)
    val rnPred = DistUtils.sampleNoSeed(Y.length, nPredSamples)

    //rnexpand = sp.random.randint(low=0,high=X.shape[1],size=nExpandSamples)
    val rnexpand = DistUtils.sampleNoSeed(Y.length, nExpandSamples)

    //G = compute_gradient(Y[rnpred],X[:,rnpred],X[:,rnexpand],W[rnexpand],kernel,C)
    val G = gradient(Y(rnPred), X(::, rnPred), X(::, rnexpand), W(rnexpand).asInstanceOf[Vector[Double]], C)

    W(rnexpand) -= eta * G

    W
  }

  // def compute_gradient(y,Xpred,Xexpand,w,kernel,C):
  def gradient(y: Vector[Double], Xpred: Matrix[Double], Xexpand: Matrix[Double], w: Vector[Double],
      C: Double): Vector[Double] = {

    //K = kernel[0](Xpred,Xexpand,kernel[1])
    val K = gaussianKernel(Xpred.toDenseMatrix, Xexpand.toDenseMatrix, sigma = 1.0)

    //yhat = K.dot(w)
    val yhat: Vector[Double] = K * w

    //# compute whether or not prediction is in margin
    //inmargin = (yhat * y) <= 1
    val yHatTimesY = yhat :* y
    val yTimesInMargin = DenseVector.zeros[Double](y.length)
    var i = 0
    while (i < yHatTimesY.length) {
      if (yHatTimesY(i) <= 1) {
        yTimesInMargin(i) = y(i)
      }
      i += 1
    }

    //G = C * w - (y * inmargin).dot(K)
    val yTimesInMarginDotK: DenseMatrix[Double] = (yTimesInMargin.asDenseMatrix * K).asInstanceOf[DenseMatrix[Double]]
    (w * C) - yTimesInMarginDotK(0, ::).t
  }


  //def GaussianKernel(X1, X2, sigma):
  //def gaussianKernel(X1: DenseMatrix[Double], X2: DenseMatrix[Double], sigma: Double): DenseMatrix[Double] = {
  def gaussianKernel(X1: DenseMatrix[Double], X2: DenseMatrix[Double], sigma: Double): DenseMatrix[Double] = {

    var i = 0
    var j = 0

    // K = cdist(X1.T, X2.T, 'euclidean')
    //TODO use matrix mult here
    val distances = DenseMatrix.zeros[Double](X1.cols, X2.cols)
    while (i < X1.cols) {

      while (j < X2.cols) {
        val subtracted = X1(::, i) - X2(::, j)
        //distances(i, j) = exp(pow(sqrt(subtracted.dot(subtracted)),2.0) / (-2.0 * sigma * sigma))
        distances(i, j) = exp(subtracted.dot(subtracted) / (-2.0 * sigma * sigma))
        j += 1
      }
      i += 1
      j = 0
    }
    distances
  }

}
