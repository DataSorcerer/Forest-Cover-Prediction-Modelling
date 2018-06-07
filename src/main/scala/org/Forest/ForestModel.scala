
package org.Forest

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import scala.util.Random
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{ PipelineModel, Pipeline }
import org.apache.spark.ml.tuning.{ ParamGridBuilder, TrainValidationSplit }

object ForestModel {
  def main(args: Array[String]): Unit = {

    //Create a new spark session
    val spark = SparkSession.builder.appName("Forest Prediction")
      .config("spark.master", "local[4]")
      .config("spark.executor.memory", "4g")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    //import methods to convert Scala objects to DataFrame
    import spark.implicits._

    //Import forest data file and infer column datatypes and schema upon import
    val rawForestData = spark.read
      .option("inferSchema", true)
      .option("header", false)
      .csv("covtype.csv")

    //Inspect imported data
    //forestData.show(5)

    //Define column headers before converting to dataframe
    val columnNames = Seq(
      "Elevation", "Aspect", "Slope",
      "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_Roadways",
      "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points") ++
      //Define columns names for 4 different types of wilderness areas
      ((0 to 3).map(x => s"Wilderness_Area_$x")) ++
      //Define columns names for 40 different soil types
      ((0 to 39).map(x => s"Soil_Type_$x")) ++
      Seq("Cover_Type")

    //Add column names to data and convert label (Class) column to double as required for Spark ML
    val forestData = rawForestData.toDF(columnNames: _*).
      withColumn("Cover_Type", $"Cover_Type".cast("double"))

    //Inspect data after adding column names
    //forestData.show(5)

    //Pre-process data before training the model
    //Keep 10% of data for evaluation and use rest 90% for training
    val Array(trainForest, testForest) = forestData.randomSplit(weights = Array(0.9, 0.1), seed = 99)
    trainForest.cache()
    testForest.cache()

    //Train the model
    //In order to train the model, we need to create feature Vectors of type Double
    val featureColumns = trainForest.columns.filter(p => p != "Cover_Type")
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("featureVector")

    //Apply vector assembler transformation to forest data frame
    val vectorisedForestData = assembler.transform(trainForest)

    //Take a look at Sparse Vector representation of the assembled column
    //vectorisedForestData.select("featureVector").show(truncate = false)

    //Define a decision tree classifier model
    val classifier = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setFeaturesCol("featureVector")
      .setLabelCol("Cover_Type")
      .setPredictionCol("Predicted_Cover_Type")
    //Train the model
    val model = classifier.fit(vectorisedForestData)

    //Show complete description of the trained model
    //println(model.toDebugString)

    //Check out which features dominate the decision tree rules in descending order
    // model.featureImportances.toArray.zip(featureColumns).
    // sorted.reverse.foreach(println)

    //Predicting forest cover types with training data to compare with known class
    val predicted_type = model.transform(vectorisedForestData)
    predicted_type.select("Cover_Type", "Predicted_Cover_Type", "probability").show(truncate = false)

    //Evaluate model performance with multiclass classification evaluator
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("Cover_Type")
      .setPredictionCol("Predicted_Cover_Type")

    //Check Accuracy as well as F1 score
    println("Accuracy = " + evaluator.setMetricName("accuracy").evaluate(predicted_type))
    println("F1 Score = " + evaluator.setMetricName("f1").evaluate(predicted_type))

    //Inspect model performance through a confusion matrix
    //first convert predictions data frame to a RDD
    val predictionsRDD = predicted_type
      .select("Predicted_Cover_Type", "Cover_Type")
      .as[(Double, Double)]
      .rdd
    val mcMetrics = new MulticlassMetrics(predictionsRDD)
    //print(mcMetrics.confusionMatrix)

    //Group the predictions by actual cover type and then pivot on the basis of predicted class type
    // to  obtain an elegant looking confusion matrix
    val confusionMatrix = predicted_type.
      groupBy("Cover_Type").
      pivot("Predicted_Cover_Type", (1 to 7)).
      count().
      na.fill(0.0).
      orderBy("Cover_Type")

    confusionMatrix.show()

    //Hyperparameter tuning
    //Construct a pipeline of transformations to iteratively check accuracy with different hyperparameters
    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    //We use a ParamGridBuilder to construct a grid of parameters to search over.
    //We try 2 values each of the for 4 hyper parameters, i.e. 2 x 2 x 2 x 2 = 16 combinations
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.impurity, Seq("gini", "entropy"))
      .addGrid(classifier.maxDepth, Seq(1, 20))
      .addGrid(classifier.maxBins, Seq(40, 300))
      .addGrid(classifier.minInfoGain, Seq(0.0, 0.05))
      .build()

    // TrainValidationSplit will try all combinations of values and determine best model using
    // the evaluator.
    val trainSplitValidator = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.9)

    val validationModel = trainSplitValidator.fit(trainForest)
    
    validationModel.save("models/RegularDecisionTree_Default_Parameters")
    
    //print metrics obtained on training each combination of hyperparameters
    val paramsAndMetrics = validationModel.validationMetrics.
      zip(validationModel.getEstimatorParamMaps).sortBy(-_._1)

    paramsAndMetrics.foreach {
      case (metric, params) =>
        println(metric)
        println(params)
        println("===================================")
    }

    //Extract the best model along with the corresponding parameters
    val bestModel = validationModel.bestModel

    //Show hyper paramters used in training the best model
    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    //Print the performance metrics for the best model
    println(validationModel.validationMetrics.max)

    //Evaluate Accuracy over unseen (held out) test dataset
    val testAccuracy = evaluator.evaluate(bestModel.transform(testForest))
    println("Test dataset accuracy: " + testAccuracy)

    //Accuracy over training dataset
    val trainAccuracy = evaluator.evaluate(bestModel.transform(testForest))
    println("Training dataset accuracy: " + trainAccuracy) 
  }

}