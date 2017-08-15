/**
  * Created by hanleyzhang on 2016/7/20.
  * word2vec 语料训练
  * argv0:语料所在位置
  * argv1:vocab文件所在位置
  * argv2:modelsave
  * argv3:googlevector
  */

import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.SQLContext
object word2vec {
  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      println("Usage:word2vec need input FileName savemodule path and vectors save path");
      System.exit(1);
    }


    val conf = new SparkConf()
    val sc = new SparkContext(conf)

	var corpus:String = args(0)
	//var vocabs:String = args(1)
	//var w2vModle:String = args(2)
	//var googleVec:String = args(3)

	var w2vModle:String = args(1)
	var googleVec:String = args(2)
    
	val input = sc.textFile(corpus).map(line => line.split(" ").toSeq)
    val word2vec = new Word2Vec()
    word2vec.setVectorSize(200)
    word2vec.setWindowSize(6)
    word2vec.setNumIterations(1)
    word2vec.setLearningRate(0.025)
    word2vec.setMinCount(30)
    word2vec.setNumPartitions(200)
	
    val word2vecmodel = word2vec.fit(input)
    //val word2vecmodel = Word2VecModel.load(sc,w2vModle)
    //word2vecmodel.save(sc,w2vModle)

    //获取语料去重后的数组
    var corpusArray = sc.textFile(corpus).flatMap(line => line.split(" ")).distinct().collect()
    //var corpusArray = sc.textFile(vocabs).flatMap(line => line.split(" ").toSeq).collect()
    val googleVector = wordsToGoogleVectors(corpusArray, word2vecmodel);


    //sc.parallelize(googleVector).map(t=>t_1+t_2.map(line=>line.replaceAll("[,()\\[\\]]"," ").stripLineEnd).saveAsTextFile(googleVec)
    sc.parallelize(googleVector).map{case(a, b) => a+b}.map(line => line.replaceAll("[,()\\[\\]]"," ").stripLineEnd).filter(line => line != "").saveAsTextFile(googleVec)

  }


  // Make some helper functions
  def sumArray (m: Array[Double], n: Array[Double]): Array[Double] = {
    for (i <- 0 until m.length) {m(i) += n(i)}
    return m
  }

  def divArray (m: Array[Double], divisor: Double) : Array[Double] = {
    for (i <- 0 until m.length) {m(i) /= divisor}
    return m
  }

  def wordToVector (w:String, m: Word2VecModel): Vector = {
    try {
      return m.transform(w)
    } catch {
      case e: Exception => return Vectors.zeros(0)
    }
  }

  def wordsToVector(words: Array[String], model: Word2VecModel): Vector = {

    val vec = Vectors.dense(
      divArray(
        words.map(word => wordToVector(word, model).toArray).reduceLeft(sumArray),
        words.length))
    vec
  }


  //通过输入 words，得到google vectors.bin形式的 训练后的数据
  def wordsToGoogleVectors(words: Array[String],model: Word2VecModel):Array[(String,String)]={
    //val s=Array("a","b")
    val vec = words.map(word => if(wordToVector(word, model).size > 5){(word, wordToVector(word, model).toString())} else{ ("","")} )
    //val vec = words.map(word=> if(wordToVector(word,model).size > 5){(word,wordToVector(word,model).toString())} else{ null } )
    //return s
    return vec
  }

  //通过输入 words，得到google vectors.bin形式的 训练后的数据
  def wordsToGoogleVectors_test(words: Array[String], model: Word2VecModel):Array[(String,Vector)]={
    val vec = words.map(word => (word, wordToVector(word, model)))
    return vec
  }

}