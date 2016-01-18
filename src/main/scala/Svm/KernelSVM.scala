package Svm

import breeze.linalg.{SliceVector, DenseMatrix, DenseVector, Vector}
import breeze.numerics._

import scala.util.Random

object KernelSVM {

  /**
  def fit_svm_kernel(W,X,Y,its=100,eta=1.,C=.1,kernel=(GaussianKernel,(1.)),visualize=False):
    D,N = X.shape[0],X.shape[1]
    X = sp.vstack((sp.ones((1,N)),X))

    errors_std_loc = []
    for it in range(its):
      errors_std_loc.append(test_svm(X,Y,W,kernel)[0])
      if visualize:
        #print "discount:",discount
        plot(errors_std_loc)

      rn = sp.random.randint(N)
      yhat = predict_svm_kernel(X[:,rn],X,W,kernel)
      discount = eta/(it+1.)
      if yhat*Y[:,rn] > 1: G = C * W
      else: G = C * W - Y[:,rn] * kernel[0](sp.vstack((X[:,rn] )),X,kernel[1]).flatten()

      W -= discount * G
    return W,errors_std_loc

    */
  def fitRandomPointCompleteKernel(W: DenseVector[Double], originalX: DenseMatrix[Double], Y: DenseMatrix[Double], iterations: Int,
    eta: Double = 1.0, C: Double = 0.1, sigma:Double = 1.0, testInterval: Int = 10): WeightsAndErrors = {

    val N = originalX.cols
    val X = DenseMatrix.vertcat[Double](DenseMatrix.ones[Double](1, N), originalX)

    assert(testInterval < iterations,
      s"please set test_interval ${testInterval} smaller than number of iterations ${iterations}")

    var errors: DenseVector[Double] = null
    if (testInterval != 0) {
      errors = DenseVector.zeros[Double](iterations / testInterval)
    }

    var i = 0
    while (i < iterations) {

      if (testInterval != 0 && i % testInterval == 0) {
        val testResult = test(X, Y, W, sigma)
        errors(i / testInterval) = testResult.error
        println(s"iteration: ${i}, thats ${i / iterations.toDouble * 100.0}%, error: ${errors(i / testInterval)}, " +
          s"number missclassified data points: ${testResult.numMisclassifiedPoints} of ${N}")
      }

      val randomIndex = Random.nextInt(N)
      val randomInstance = X(::, randomIndex)
      
      val yhat =  W.t * gaussianKernel(randomInstance, X, sigma)
      val discount = eta / (i + 1.0)

      val r = yhat * Y(::, randomIndex)
      assert(r.length == 1)

      val G =
        if (r(0) > 1) {
          C * W
        } else {
          val Y_ = Y(::, randomIndex)
          C * W - Y_(0) * gaussianKernel(randomInstance, X, sigma)
        }

      W := W - discount * G

      i += 1
    }

    WeightsAndErrors(W, errors)
  }


  /**
   * def test_svm(X,Y,W,(k,(kparam))):
        kernel = (k,(kparam))
        error = np.zeros(1)
        point_error = 0
        for rn in range(X.shape[1]):
          yhat = predict_svm_kernel(X[:,rn],X,W,kernel)
          err = yhat*Y[:,rn]
          if not err >= 0:
            error -= yhat*Y[:,rn]
            point_error += 1
        return [error[0]/float(X.shape[1]),point_error/float(X.shape[1])]
   */
  def test(X: DenseMatrix[Double], Y: DenseMatrix[Double], W: DenseVector[Double], sigma: Double): TestResult = {
    var index = 0
    val N = X.cols

    var numMisclassifiedPoints = 0
    var error = 0.0

    while (index < N) {
      val yhat = W.t * gaussianKernel(X(::, index), X, sigma)
      val err = yhat * Y(::, index)

      assert(err.length == 1)

      val misclassified = (err(0) < 0)

      if (misclassified) {
        error -= err(0)
        numMisclassifiedPoints += 1
      }
      index += 1
    }
    TestResult(error / N, numMisclassifiedPoints)
  }

  /**
  def GaussianKernel(X1, X2, sigma):
    assert(X1.shape[0] == X2.shape[0])
    K = cdist(X1.T, X2.T, 'euclidean')
    K = sp.exp(-(K ** 2) / (2. * sigma ** 2))
    return K
    */
  def gaussianKernel(x1: DenseVector[Double], x2: DenseMatrix[Double], sigma: Double): DenseVector[Double] = {
    assert(x1.length == x2.rows)
    var i = 0
    val diff = DenseVector.zeros[Double](x2.cols)
    // TODO there should be some way to use matrix vector multiplication here
    while (i < x2.cols) {
      diff(i) = sqrt((x1 :- x2(::,i)).dot(x1 :- x2(::,i)))
      i += 1
    }
    exp(-(diff :* diff) / (2 * sigma * sigma))
  }

  def predictSVMKernel(x: DenseVector[Double], X: DenseMatrix[Double],
    W: DenseVector[Double], sigma: Double): Double = {
    W.t * gaussianKernel(x, X, sigma)
  }
}
