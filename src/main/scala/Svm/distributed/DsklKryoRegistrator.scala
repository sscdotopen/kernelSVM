package Svm.distributed

import breeze.linalg.{SparseVector, DenseMatrix, DenseVector}
import com.esotericsoftware.kryo.Kryo
import org.apache.spark.SparkConf
import org.apache.spark.serializer.KryoRegistrator

object DsklKryo {

  def addKryoSettings(conf: SparkConf) = {
    conf.set("spark.serializer", classOf[org.apache.spark.serializer.KryoSerializer].getCanonicalName)
    conf.set("spark.kryo.registrator", classOf[DsklKryoRegistrator].getCanonicalName)
    //conf.set("spark.kryo.registrationRequired", true.toString)
  }
}

class DsklKryoRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[DenseVector[Double]])
    kryo.register(classOf[DenseMatrix[Double]])
    kryo.register(classOf[SparseVector[Double]])
  }
}
