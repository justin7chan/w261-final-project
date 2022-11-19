# Databricks notebook source
# MAGIC %md
# MAGIC # *Anything in quote block markdown format should be deleted afterwards*
# MAGIC > Example is this
# MAGIC > - Anything like this should be deleted after final export

# COMMAND ----------

# MAGIC %md
# MAGIC > Video
# MAGIC > - Should be a 2-minute video (give or take 10 seconds)
# MAGIC > - All team members need to present a part of the video.
# MAGIC > - Your video should have a logical and scientific flow to it
# MAGIC >   - A title slide (with the project name, Team Number, team member names, and photos).
# MAGIC >   - Make sure your presentation has an outline slide with good descriptive section headings
# MAGIC >   - Project description slide (problem being solved, customer, key machine learning metrics, key performance indicators)
# MAGIC >   - Please provide phase updates and a sample of the deliverables. E.g.,
# MAGIC >     - Features that were engineered
# MAGIC >     - EDA of engineered features
# MAGIC >     - Overview of Modeling Pipelines explored
# MAGIC >     - Experimental results
# MAGIC >   - Discussion of key findings
# MAGIC >   - Conclusion and next steps
# MAGIC >   - Open issues or problems
# MAGIC > - NOTE: Did each section in your notebook from abstract to conclusions, and each slide in your presentation address the new features you engineered?

# COMMAND ----------

# MAGIC %md
# MAGIC > Notebook Structure \
# MAGIC > Your notebook should have the following major sections and be easy to follow:
# MAGIC > - Abstract
# MAGIC > - Introduction: problem, datasets, metrics
# MAGIC > - Key steps/pipelines/hyperparameters.
# MAGIC > - Results
# MAGIC > - Discussion
# MAGIC > - Conclusion

# COMMAND ----------

# MAGIC %md
# MAGIC #✈ Final Project: Phase 3 Report ✈
# MAGIC ##An Algorithmic Approach to Predicting Flight Delays
# MAGIC ###Section 3 Group 2 | Ramki Gummadi
# MAGIC ###Job Bangayan, Justin Chan, Matthew Rubino, Steven Sung
# MAGIC #####Phase Leader: Matthew Rubino

# COMMAND ----------

# DBTITLE 1,Get raw blob storage
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# DBTITLE 1,Read raw parquet files
raw_airlines_df = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
raw_weather_df = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data/")
raw_stations_df = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abstract
# MAGIC 
# MAGIC > Abstract Criteria
# MAGIC > - 150 words approximately describing this phase of the project
# MAGIC > - Details the problem you are tackling
# MAGIC > - Main goal of this phase
# MAGIC > - What you did (main experiments)
# MAGIC > - What were your results/findings (best pipeline and the corresponding scores)
# MAGIC > - Next steps
# MAGIC > - Any problems
# MAGIC 
# MAGIC The focus of this project is to build a predictive analysis for predicting delays from weather data
# MAGIC for US-bound and departing flights to avoid economic losses and passenger inconvenience. We
# MAGIC used a logistic model to predict if a flight will take off 15 minutes or later from the respective
# MAGIC planned time of departure. We have determined that using weather data has little to no impact
# MAGIC on how well a model performs. Next steps that we are planning are to include variables on
# MAGIC airplane status and delay time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introduction
# MAGIC 
# MAGIC > Project Description
# MAGIC > - Data description
# MAGIC > - Task to be tackled
# MAGIC > - Provide diagrams to aid understanding the workflow
# MAGIC 
# MAGIC According to the U.S. Bureau of Transportation Statistics, over the last 10 years, on average 18.74% of scheduled flights are delayed by 15 minutes or more per year, with 2022 representing the highest percentage (21.16%) of flights delayed since 2014. Delays represent a significant headache for both travelers and airlines alike. Researchers from the University of California Berkeley commissioned by the Federal Aviation Administration found that domestic flight delays cost the U.S. economy $32.9bn, with about half of the cost borne by airline passengers and the balance borne by airline operators. Today, flight delays remain a widespread issue that places a significant burden on airlines, travelers, and the broader economy at large.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Problem
# MAGIC 
# MAGIC The problem we aim to solve is how to best predict delays for US-bound and departing flights using weather data to avoid economic losses for airlines and to reduce passenger inconvenience. Data on severe weather such as tornados, heavy winds, and floods is captured in weather stations, where we can correlate this data with air flight delays.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Datasets
# MAGIC We will leverage three data sets including (i) flight data, (ii) weather data, and (iii) airport data. The data will be ingested from a Parquet file into Pandas data frames for ease of use and wrangling. The airlines data includes a subset of passenger flights' on-time performance data derived from the U.S Department of Transportation over the time window spanning 2015 to 2021 by quarter. Similarly, the weather data spans 2015 to 2021 and is derived from the National Oceanic and Atmospheric Administration repository. The stations dataset contains metadata about weather station location and airports nearest to it. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join

# COMMAND ----------

# MAGIC %md
# MAGIC ### Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Pipeline
# MAGIC 
# MAGIC ### Key Steps
# MAGIC 
# MAGIC ### Feature Engineering
# MAGIC 
# MAGIC > Feature Engineering
# MAGIC > - Describe newly engineering features that were added (in the form of feature families if possible)
# MAGIC > - Show the impact (if any) that these new features added to the model in the modeling pipelines section below
# MAGIC > - Explain why you chose the method and approach in presentation/results/discussion sections
# MAGIC 
# MAGIC ### Pipelines
# MAGIC > Modeling Pipelines
# MAGIC > - A visualization of the modeling pipeline (s) and sub pipelines if necessary
# MAGIC > - Families of input features and count per family
# MAGIC > - Number of input features
# MAGIC > - Hyperparameters and settings considered
# MAGIC > - Loss function used (data loss and regularization parts) in latex
# MAGIC > - Number of experiments conducted
# MAGIC > - Experiment table (using a pandas dataframe)with the following details per experiment:
# MAGIC >   - Baseline experiment
# MAGIC >   - Any additional experiments
# MAGIC >   - Final model tuned
# MAGIC >   - best results (1 to three) for all experiments you conducted with the following details
# MAGIC >   - The families of input features used
# MAGIC 
# MAGIC > Provide sample:
# MAGIC > - Plots of loss and other score curves per epoch/tree
# MAGIC 
# MAGIC ### Hyperparameters
# MAGIC 
# MAGIC > Hyperparameter Tuning
# MAGIC > - Describe the hyperparameters that your explored in your modeling pipelines
# MAGIC > - Explain the method and scoring methodology

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC > Results
# MAGIC > - Make sure your experiments are properly enumerated/tabulated and discussed (they are missing accurate descriptions, performance metrics). 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discussion
# MAGIC > Discussion
# MAGIC > - Discussion’s aim is result interpretation, which means explain, analyse, and compare them. Often, this part is the most important, simply because it lets the researcher take a step back and give a broader look at the experiment. 
# MAGIC > - Do not discuss any outcomes not presented in the results part. 
# MAGIC > - Write a gap analysis of your best pipeline against the Project Leaderboard
# MAGIC >   - A gap analysis is a process that compares actual performance or results with what was expected or desired (versus other teams in this case). The method provides a way to identify suboptimal or missing strategies, structures, capabilities, processes, practices, technologies, or skills, and then recommends steps that will help the company meet its goals.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC > Conclusion
# MAGIC > - Restate your project focus and explain why it’s important. Make sure that this part of the conclusion is concise and clear.
# MAGIC > - Restate your hypothesis (e.g., ML pipelines with custom features can accurately predict blah blah
# MAGIC > - Summarize main points of your project: Remind your readers of your key contributions.
# MAGIC > - Discuss the significance of your results
# MAGIC > - Discuss the future of your project.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Credit Assignment
# MAGIC > Updated Credit Assignment Plan
# MAGIC > - Credit assignment plan updates (who does/did what and when, amount of effort in terms of person hours, start and end dates, estimated improvement in terms of key metrics) in Table format
# MAGIC > - No Credit assignment plan means ZERO points
# MAGIC > - A credit assignment plan not in Table format means ZERO points
# MAGIC > - No start and end dates and (budgeted) hours of effort mean an incomplete plan. This may result in zero points.

# COMMAND ----------


