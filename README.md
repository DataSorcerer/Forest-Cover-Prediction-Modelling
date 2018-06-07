# Forest Cover Type prediction model using Spark ML

This project makes use of a well-known Machine Learning dataset, *"Forest Cover Type*, that contains information about land covered by forests in the state of Colorado, USA. The dataset has been donated by Jock A. Blackard and Colorado State University and can be accessed at [UCI Machine Learning Datasets Archive-ForestCovType](https://archive.ics.uci.edu/ml/datasets/covertype).   
      
### Why Spark and Scala for building the predictive model?   

The (moderately) **"large scale"** characteristics of the dataset are:- more than 580,000 records and 54 features which can be used to predict the class - type of forest cover. These characteristics make it an ideal candidate to try large scale distributed analytics framework like Spark with functional programming language like Scala.      
We train and tune a **scalable decision tree classifier** leveraging the following features of Spark architecture:    
1. Highly parallelizable memory intensive data preprocessing: Allows the spark job to be distributed over multiple nodes in cluster.    
2. Feature Engineering with Spark ML - VectorAssembler    
3. Use of **"Transformers"** for robust data wrangling and feature engineering tasks, also aids in transforming data autonomously and repeatedly in production   
4. **"Estimators"** to train the ML model  
5. Spark SQL for column transformations
6. **ParamGridBuilder()** for hyper-parameter tuning with different range of values  
7. Transformers for evaluating the ML model with cross-validation
5. **Pipelines** to chain all the stages of transformers and estimators which greatly **operationalizes** the scalable machine learning tasks.


### Instruction to build/ deploy the driver program:   
(suggested running with at least 8GB of RAM)

**Build:**   
host> git clone https://github.com/DataSorcerer/Forest-Cover-Prediction-Modelling.git    
host> cd Forest-Cover-Prediction-Modelling    
host> sbt assembly     

This creates a fat jar ready to submitted to a Spark job:    
target/ForestModel.jar   

**Deploy:**

#### 1. On a Kubernetes cluster in cluster deploy mode
./bin/org/Forest \
  --class org.Forest.ForestModel \
  --master k8s://xx.yy.zz.ww:443 \
  --deploy-mode cluster \
  --executor-memory 8G \
  --num-executors 5 \
  http://path/to/ForestModel.jar \
  1000

#### 2. On a local machine:   
./bin/org/Forest \
--class org.Forest.ForestModel \
--executor-memory 8G \
--master local[4]\
ForestModel.jar

#### 3. On a Spark standalone cluster in cluster deploy mode
./bin/org/Forest \
  --class org.Forest.ForestModel \
  --master <Master URL for cluster>\
  --deploy-mode cluster \
  --executor-memory 8G \
  --total-executor-cores 8 \
  ForestModel.jar \
  1000
  
