name := "Forest_Type_Predictor"

version := "1.0"

scalaVersion := "2.11.8"


libraryDependencies ++= Seq(
"org.apache.spark" %% "spark-core" % "2.3.0",
"org.apache.spark" %% "spark-sql" % "2.3.0",
"org.apache.spark" %% "spark-mllib" % "2.3.0",
"org.apache.spark" %% "spark-streaming" % "2.3.0")
	
