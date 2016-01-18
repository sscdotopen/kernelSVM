package Svm

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._

import scala.util.Random


object EmpKernelSVM {

//  /*
//def step_dskl_empirical(X,Y,W,eta=1.,C=.1,kernel=(GaussianKernel,(1.)),nPredSamples=10,nExpandSamples=10):
//    if nPredSamples==0: rnpred=range(len(Y))
//    else: rnpred = sp.random.randint(low=0,high=X.shape[1],size=nPredSamples)
//    if nExpandSamples==0:rnexpand=range(len(Y))
//    else: rnexpand = sp.random.randint(low=0,high=X.shape[1],size=nExpandSamples)
//    # compute gradient
//    G = compute_gradient(Y[rnpred],X[:,rnpred],X[:,rnexpand],W[rnexpand],kernel,C)
//    # update
//    W[rnexpand] -= eta * G
//    return W
//
//
//    def compute_gradient(y,Xpred,Xexpand,w,kernel,C):
//    # compute kernel for random sample
//    K = kernel[0](Xpred,Xexpand,kernel[1])
//    # compute prediction
//    yhat = K.dot(w)
//    # compute whether or not prediction is in margin
//    inmargin = (yhat * y) <= 1
//    # compute gradient for
//    G = C * w - (y * inmargin).dot(K)
//    return G
//  */
//  def fitEmp(W: DenseVector[Double], originalX: DenseMatrix[Double], Y: DenseMatrix[Double], iterations: Int,
//             eta: Double = 1.0, C: Double = 0.1, sigma:Double = 1.0, testInterval: Int = 10, nPredSamples: Int = 10,
//             nExpandSamples: Int = 10): WeightsAndErrors = {
//
//    val N = originalX.cols
//    val X = DenseMatrix.vertcat[Double](DenseMatrix.ones[Double](1, N), originalX)
//
//    assert(testInterval < iterations,
//      s"please set test_interval ${testInterval} smaller than number of iterations ${iterations}")
//
//    var errors: DenseVector[Double] = null
//    if (testInterval != 0) {
//      errors = DenseVector.zeros[Double](iterations / testInterval)
//    }
//
//    var i = 0
//    while (i < iterations) {
//
//      if (testInterval != 0 && i % testInterval == 0) {
//        val testResult = KernelSVM.test(X, Y, W, sigma)
//        errors(i / testInterval) = testResult.error
//        println(s"iteration: ${i}, thats ${i / iterations.toDouble * 100.0}%, error: ${errors(i / testInterval)}, " +
//          s"number missclassified data points: ${testResult.numMisclassifiedPoints} of ${N}")
//      }
//
//      val predictSampleIndices = (0 until nPredSamples).map { _ => Random.nextInt(N) }
//      val expandSampleIndices = (0 until nExpandSamples).map { _ => Random.nextInt(N) }
//
//      val kernel = gaussianKernelEmp(X, sigma, predictSampleIndices, expandSampleIndices)
//
//      //TODO slicing doesn't seem to work here....
//      //val We: SliceVector[Int, Double] = W(expandSampleIndices)
//      val We = DenseVector.zeros[Double](nExpandSamples)
//      for (i <- 0 until expandSampleIndices.length) {
//        We(i) = W(expandSampleIndices(i))
//      }
//
//      val yhat = kernel * We
//      val notInMargin = DenseVector.zeros[Double](predictSampleIndices.length)
//
//      val labels: DenseVector[Double] = Y(::, predictSampleIndices).flatten().toDenseVector
//      val r: DenseVector[Double] = labels :* yhat
//      for (i <- 0 until predictSampleIndices.length) {
//        if (r(i) <= 1) {
//          val Y_ = Y(::, i)
//          notInMargin(i) = Y_(0)
//        }
//      }
//
//      //val discount = eta / (i + 1.0)
//      val discountedGradient = eta * (C * We - kernel * notInMargin)
//      println(discountedGradient)
//
//      //      W := W - discount * G
//      for (i <- 0 until predictSampleIndices.length) {
//        W(expandSampleIndices(i)) -= discountedGradient(i)
//      }
//
//      i += 1
//    }
//
//    WeightsAndErrors(W, errors)
//  }
//
//  /*
//  def GaussianKernel(X1, X2, sigma):
//     assert(X1.shape[0] == X2.shape[0])
//     K = cdist(X1.T, X2.T, 'euclidean')
//     K = sp.exp(-(K ** 2) / (2. * sigma ** 2))
//     return K
//  */
//  def gaussianKernelEmp(X: DenseMatrix[Double], sigma: Double, predictSampleIndices: IndexedSeq[Int],
//                        expandSampleIndices: IndexedSeq[Int]): DenseMatrix[Double] = {
//
//    var i = 0
//    var j = 0
//
//    val distances = DenseMatrix.zeros[Double](predictSampleIndices.length, expandSampleIndices.length)
//    // TODO there should be some way to use matrix vector multiplication here
//    while (i < predictSampleIndices.length) {
//      val x1 = X(::, predictSampleIndices(i))
//      while (j < expandSampleIndices.length) {
//        val index = expandSampleIndices(j)
//        val subtracted = x1 :- X(::, index)
//        distances(i, j) = sqrt(subtracted.dot(subtracted))
//        j += 1
//      }
//      i += 1
//    }
//
//    exp(-(distances :* distances) / (2 * sigma * sigma))
//  }
//

}
