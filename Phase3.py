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
# MAGIC 
# MAGIC > Feedback from previous phases
# MAGIC > - Narrow down the project description to more of a business case. Are you taking the side of the airline or the customer. Not entirely clear from video. Don't use accuracy. Video just suddenly cuts out.

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
# MAGIC #####Phase Leader: Steven Sung

# COMMAND ----------

# MAGIC %md
# MAGIC **Run the cells below to obtain information about Azure blob storage and dataframes**

# COMMAND ----------

# DBTITLE 1,Imports
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.types as T
import pyspark.sql.functions as F

from pyspark.sql.window import Window

# COMMAND ----------

# DBTITLE 1,Get raw blob storage
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# DBTITLE 1,Read raw parquet files
raw_airlines_df = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
raw_stations_df = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
raw_weather_df = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data/")

# COMMAND ----------

# DBTITLE 1,Blob Storage Setup
blob_container = "latam" # The name of your container created in https://portal.azure.com
storage_account = "mrubino" # The name of your Storage account created in https://portal.azure.com
secret_scope = "sect3group2" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "sect3group2key" # saskey The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net", dbutils.secrets.get(scope = secret_scope, key = secret_key))

# COMMAND ----------

# DBTITLE 1,Read clean tables from storage if applicable (update filepath as needed)
# Make sure that the filepath version is the latest one
clean_airlines_df = spark.read.parquet(f"{blob_url}/clean_airlines_2")
clean_stations_df = spark.read.parquet(f"{blob_url}/clean_stations_1")
clean_weather_df = spark.read.parquet(f"{blob_url}/clean_weather_2")
final_df = spark.read.parquet(f"{blob_url}/final_df_9")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abstract
# MAGIC 
# MAGIC > Abstract Criteria
# MAGIC > - 150 words approximately describing this phase of the project
# MAGIC > - **Details the problem you are tackling**
# MAGIC > - Main goal of this phase
# MAGIC > - What you did (main experiments)
# MAGIC > - **What were your results/findings (best pipeline and the corresponding scores)**
# MAGIC > - **Next steps**
# MAGIC > - **Any problems**
# MAGIC 
# MAGIC > Feedback from previous phases
# MAGIC > - No metrics were given. No business was really given. What perspective is the team going to take within the given problem?
# MAGIC > - No mention of the data join. No mention of metrics.
# MAGIC 
# MAGIC We have been tasked by the FAA to set new flight regulations based on hazardous weather conditions. More specifically, what type of weather conditions will create flight delays longer than 15 minutes. Datasets provided by the US DoT and NOAA contain information about airlines, weather, and stations where we'll perform EDA, clean the records, and join them based on UNIX timestamp. **<Talk about feature engineering, hyperparameters>**. The baseline models include logistic regression and deceision tree models which had a 47.94% and 62.61% accuracy respectively. **< Change to reflect current models (recall as main metric) >** Our next step is to create additional models that utilize XGBoost, SMOTE, and ensemble methods to improve our recall score.
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
# MAGIC > - **Provide diagrams to aid understanding the workflow**
# MAGIC 
# MAGIC **Rewrite**
# MAGIC 
# MAGIC According to the U.S. Bureau of Transportation Statistics, over the last 10 years, on average 18.74% of scheduled flights are delayed by 15 minutes or more per year, with 2022 representing the highest percentage (21.16%) of flights delayed since 2014. Delays represent a significant headache for both travelers and airlines alike. Researchers from the University of California Berkeley commissioned by the Federal Aviation Administration (FAA) found that domestic flight delays cost the U.S. economy $32.9 billions, with about half of the cost borne by airline passengers and the balance borne by airline operators. Today, flight delays remain a widespread issue that places a significant burden on airlines, travelers, and the broader economy at large.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Problem
# MAGIC 
# MAGIC We have been tasked by the FAA to set new flight regulations based on hazardous weather conditions. Dangerous weather can risk the health of many passengers and employees. One of the flight regulations we aim to change how flight delays are defined. There are many influences that can delay a flight such as National Air Systems and late arrival, but we want to look at just weather data. More specifically, what type of weather conditions will create flight delays longer than 15 minutes. If flights are cancelled or delayed due to severe and hazardous weather like rainstorms, floods, or blizzards, it can decrease our customer satisfaction, revenue, and long term economic growth. By decreasing delays, airlines can send out more flights at approriate times to increase their revenue and decrease flight fatalities.
# MAGIC 
# MAGIC To solve this problem, we plan to understand the correlation between weather and flight delays using machine learning models. We define a flight delay as a any flight departing 15 minutes later than expected scheduled departure time. Data on severe weather such as tornados, heavy winds, and floods is captured by stations, where we can correlate this data with air flight delays.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Datasets
# MAGIC 
# MAGIC > Feedback from previous phase
# MAGIC > - Data description could be more detailed. Not many data entries in the weather data. Difference between flight and airport data? Adding a diagram can help with the workflow. You discuss ways of handling missing values but never actually come to a conclusion.
# MAGIC > - There is some inconsistency when you say the three different data frames are i) flight data, (ii) weather data, and (iii) airport data in the data description but then (i) airlines, (ii) stations, and (iii) weather in the EDA.
# MAGIC 
# MAGIC 
# MAGIC > - Give a brief overview of each data set
# MAGIC > - How will these datasets help us?
# MAGIC > - What is our steps in a high overview.
# MAGIC 
# MAGIC We will leverage three data sets which are airline, station, and weather data. The airline and station dataset were retrieved from the [US Department of Transportation](https://www.transtats.bts.gov/homepage.asp) and the weather data was obtained by the [National Oceanic and Atmospheric Administration Repository](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00679). We used these specific datasets since they were open source and contained a lot of recent information about flights and weather data. Since these datasets come from different data sources with no guarantees about data integrity, we must do our own exploratory data anlaysis on each table
# MAGIC 
# MAGIC The datasets were downloaded and converted into parquet files into Azure Blob Cloud Storage so that reading and writing the files over millions of records would be fast. When ready to analyze the data, we pulled the data from the cloud into PySpark dataframes through EDA like plots and histograms. Once we have a better understanding of each dataset, we will clean each dataframe based on null counts and create checkpoints for the celan dataframes so that they can be referenced later. A join on keys like date, time, and location will combine all three datasets into one master table which will be used to train our models as well as any additional EDA.
# MAGIC 
# MAGIC We will discuss more about the cleaning process of each dataset in detail in each of the subsections below. We also show a visual diagram below.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/sysung/w261-final-project/blob/master/full_workflow_pipeline.drawio%20(1).png?raw=true" width=80%>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airlines

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Description
# MAGIC 
# MAGIC The airline dataset containing flights from 2015 to 2021 inclusive has 74,177,433 total rows and 109 fields of data. Each row corresponds to an individual flight within the United States. Looking at their database dictionary from the [Bureau of Transportation Statistics](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ) under the US Department of Transportation, we can delve into each feature family and how it relates to our problem statement.
# MAGIC 
# MAGIC ###### Time Period
# MAGIC Time Period contains all of the information related to the flights time like Year, Quarter, Month, Day of Month, Day of Week, and Flight Date. This will be used as our join key with the weather dataset so that we can get weather information surrounding flight departure.
# MAGIC 
# MAGIC ###### Airline
# MAGIC Airline containes information about the airline like unique carrier codes for the airline company, the flights coming in and out of airports, and tail numbers for specific planes. There are many codes used for an individual airline company where some of them are assigned by IATA while others are assigned by US DOT. Therefore, we picked the field `OP_UNIQUE_CARRIER` which is a unique identifier for a specific carrier.
# MAGIC 
# MAGIC ###### Origin and Destination
# MAGIC The fields within Origin and Destination are very similar where they had information on the origin and destinatino airport like the ID, city, State, FIPS code, etc. Since we're only interested in departing flight delays, we'll mostly be looking at the Origin airport information
# MAGIC 
# MAGIC ###### Departure and Arrival Performance
# MAGIC Departure and Arrival Performance has infromation about the flight status which includes minute details from wheels off, wheels on, taxi in, taxi, out, and delays. We'll be focusing on departure performance as this is our main motvie. More importantly it has a field called `DEP_DEL15` which is a binary variable that states if a flight is delayed by more than 15 minutes or not. This variable will be our main predictor variable.
# MAGIC 
# MAGIC ###### Cancellations and Diversions
# MAGIC This section does not contain a lot of fields, but is somewhat relevant in flights. However, in our use case, cancellations and diversions occur during a flight of a specific tail number, not pre-flight information. Therefore, we can probably ignore this field.
# MAGIC 
# MAGIC ###### Cause of Delay
# MAGIC The data dictionary lists multiple causes of delays such as carrier delay, weather delay, national air system delay, and others. While these may seem intriguing, some of these delays can be handled through human intervention. We can see how weather delays in minutes correlates with departure delays.
# MAGIC 
# MAGIC ###### Gate Return Information at Origin Airport
# MAGIC The Gate Return Information at Origin Airport contains redundant information like departure time at origin airport and post-flight details which does not help solve our problem. We can safely ignore the fields under this feature family.
# MAGIC 
# MAGIC ###### Diverted Airport Information
# MAGIC Diverted airport information pertains to only flights that have diverted. Flight that have diverted do not necessarily mean that they departed early or late. In conclusion, we can remove the fields pertaining to diverted airports.

# COMMAND ----------

# DBTITLE 0,Display raw row and column count
print(f"Raw Airlines Row Count: {raw_airlines_df.count()}")
print(f"Raw Airlines Column Count: {len(raw_airlines_df.columns)}")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##### Null Counts
# MAGIC 
# MAGIC The first form of EDA that we did was look for null counts in the raw data set. Raw data usually contains nulls from either human error or by logic, for example, if the field is not applicable to the flight. The histogram is sorted by the indices that the column appears in the raw table. We've removed all columns which have no nulls so that we can see which field has the highest null counts. 

# COMMAND ----------

# DBTITLE 1,Count nulls of each column
raw_null_ct_df = raw_airlines_df.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in raw_airlines_df.columns]).cache()

# COMMAND ----------

# DBTITLE 1,Transpose null counts dataframe
null_ct_list = raw_null_ct_df.rdd.flatMap(lambda x: x).collect()
colnames = ["fields", "nonzero_nulls"]
null_rdd = spark.sparkContext.parallelize(list(zip(raw_null_ct_df.columns, null_ct_list)))
null_df = spark.createDataFrame(null_rdd).toDF(*colnames)

# COMMAND ----------

# MAGIC %md
# MAGIC Upon further inspection of the data, most of the columns that have high null values fall under the Cause of Delay and Diverted Airpot Information features. This is understandable as a high volume of null values means that there are very few flights that were not enroute.

# COMMAND ----------

# DBTITLE 0,Plot histogram of non-zero null counts
null_pd = null_df.filter(F.col("nonzero_nulls") > 0).pandas_api().to_pandas()

chart = sns.catplot(y="nonzero_nulls", x="fields", kind="bar", data=null_pd, color="blue", height=5, aspect=15/5)
chart.set_axis_labels("Field Names", "Null Count")
chart.set(title='Non-Zero Null Count of Fields')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Weather Delay
# MAGIC In the airlines data, there is a column called `WEATHER_DELAY`. Since we are exploring how weather impact flights, this is an interesting column to dive into. For this EDA, will will format all `WEATHER_DELAY` values where if it's null or 0, it will turn to 0. Otherwise, it will be 1.

# COMMAND ----------

weather_delay = raw_airlines_df.select(
        'WEATHER_DELAY',
        'DEP_DEL15'
    ).withColumn(
        'BIN_WEATHER_DELAY', F.when((F.col('WEATHER_DELAY').isNull()) | (F.col('WEATHER_DELAY')==0), 0).otherwise(1)
    ).drop('WEATHER_DELAY')

# COMMAND ----------

# MAGIC %md
# MAGIC At first glance in the table, we see that there are flights that delayed, but there is no weather delay. This means that the airline was delayed by something that wasn't weather.

# COMMAND ----------

display(weather_delay)

# COMMAND ----------

# MAGIC %md
# MAGIC We will count how many records were delayed by weather, delayed but not by weather, and not delayed at all. These will be the count of the variations of 0s and 1s between `WEATHER_DELAY` and `DEP_DEL15`. We will also count the scenario where `DEP_DEL15==0` and `BIN_WEATHER_DELAY==1`, but we expect this to be 0 because a flight cannot be on time while there is a weather delay.

# COMMAND ----------

no_delay = weather_delay.filter((F.col('DEP_DEL15') == 0) & (F.col('BIN_WEATHER_DELAY') == 0)).count()
delay_not_by_weather = weather_delay.filter((F.col('DEP_DEL15') == 1) & (F.col('BIN_WEATHER_DELAY') == 0)).count()
delay_by_weather = weather_delay.filter((F.col('DEP_DEL15') == 1) & (F.col('BIN_WEATHER_DELAY') == 1)).count() 
impossible_scenario = weather_delay.filter((F.col('DEP_DEL15') == 0) & (F.col('BIN_WEATHER_DELAY') == 1)).count()

print(f"No delayed flights: {no_delay}")
print(f"Delay, but not by weather: {delay_not_by_weather}")
print(f"Delay by weather: {delay_by_weather}")
print(f"No delay, but weather delay (impossible): {impossible_scenario}")

# COMMAND ----------

# MAGIC %md
# MAGIC We see that there are 59,955,343 flights that departed on time, 12,131,052 flights that were delayed but not by weather, and 681,827 flights that were delayed by weather. There were also 73,976 anomalous flights that fell under the impossible scenario of having no delay, but labeled as a flight delay.
# MAGIC 
# MAGIC Since we only want to include airlines that were either affected by weather delay or not delayed at all, we will keep rows where the condition `BIN_WEATHER_DELAY==1 & DEP_DEL15==1 ` and `BIN_WEATHER_DELAY==0 & DEP_DEL15==0`

# COMMAND ----------

weather_delay_airlines_df = raw_airlines_df.withColumn(
        'BIN_WEATHER_DELAY', 
        F.when((F.col('WEATHER_DELAY').isNull()) | (F.col('WEATHER_DELAY')==0), 0).otherwise(1)
    ).filter(
        ((F.col('BIN_WEATHER_DELAY')==1) & (F.col('DEP_DEL15')==1)) | \
        ((F.col('BIN_WEATHER_DELAY')==0) & (F.col('DEP_DEL15')==0))
    ).drop('BIN_WEATHER_DELAY')

# COMMAND ----------

# MAGIC %md
# MAGIC Our new airlines dataframe now contains 60,637,170 flights while keeping the same number of columns

# COMMAND ----------

print(f"New row count: {weather_delay_airlines_df.count()}")
print(f"New column count: {len(weather_delay_airlines_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Feature Selection
# MAGIC > - I like the example of how you filled in missing data but you need to include all instances, some columns are not as easy to fill in the missing data. 
# MAGIC 
# MAGIC To clean the data, we removed all of the columns that have an extremely high null value count. This will reduce our dimensions and remove the curse of dimensionality. There are a few exceptions in which we kept like airport information because it will be used to join on the different tables; in the final dataframe, it will be dropped.
# MAGIC 
# MAGIC From the subsetted features,  we can easily fill some of the columns using basic arithmetics for columns with null values. Since we are doing it on a smaller set of fields, we only need to calculate null values for a few fields and not all 109 fields. We used `CRS_DEP_TIME` and `DEP_TIME` to calculate the `DEP_DELAY`, which is then used to calculate `DEP_DEL15` based on a simple condition statement if `DEP_DELAY` is greater than or equal to 15 or not. If some rows still contained nulls, we removed the row altogether as it would only be a few rows lost.
# MAGIC 
# MAGIC We've also dropped columns that were collinear to the output variable. For example, `DEP_DELAY`, which is the amount of time delayed, and `WEATHER_DELAY`, which is the amount of time delayed by weather, are both dropped.

# COMMAND ----------

# DBTITLE 0,Drop columns
subset_cols = [
    "YEAR",
    "QUARTER",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER",
    "TAIL_NUM",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "ORIGIN_AIRPORT_ID",
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_NM",
    "ORIGIN_STATE_ABR",
    "DEST",
    "DEST_AIRPORT_ID",
    "DEST_CITY_NAME",
    "DEST_STATE_NM",
    "DEST_STATE_ABR",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DEL15"
]

na_drop_cols = [
    "CRS_DEP_TIME",
    "DEP_TIME"
]

ft_select_airlines_df = weather_delay_airlines_df \
    .select(subset_cols) \
    .na.drop(subset=na_drop_cols) \
    .withColumn("DEP_DEL15", F.when(F.col("DEP_DELAY") >= 15, 1).otherwise(0)) \
    .drop(*["DEP_TIME", "DEP_DELAY"])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Primary Keys
# MAGIC A huge flaw in raw datasets is that they can contain duplicates flights. Therefore, since we want to make each flight unique, we can create a set of primary keys that will uniquely identify each row. We can also remove rows which have null keys since it may also heavily skew the data.
# MAGIC 
# MAGIC The primary keys we have chosen are `YEAR`, `MONTH`, `DAY_OF_MONTH`, `OP_UNIQUE_CARRIER`, `TAIL_NUM`, `OP_CARRIER_FL_NUM`, `ORIGIN_AIRPORT_ID`, and `DEST_AIRPORT_ID` because on any given day, there should only be one airplane going from origin airport to destination airport with the flight number.

# COMMAND ----------

# DBTITLE 1,Primary Keys
primary_keys = [
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "OP_UNIQUE_CARRIER",
    "TAIL_NUM",
    "OP_CARRIER_FL_NUM",
    "ORIGIN_AIRPORT_ID",
    "DEST_AIRPORT_ID"
]

# COMMAND ----------

# DBTITLE 1,View duplicate flights
display(ft_select_airlines_df.select(primary_keys).groupBy(primary_keys).count().filter(F.col("count") > 0))

# COMMAND ----------

# DBTITLE 1,Find an example of a duplicate flight
example_dupe_flight = ft_select_airlines_df \
    .filter(
        (F.col("YEAR")==2019) &
        (F.col("MONTH")==5) &
        (F.col("DAY_OF_MONTH")==18) &
        (F.col("OP_UNIQUE_CARRIER")=="UA") &
        (F.col("TAIL_NUM")=="N36247") &
        (F.col("OP_CARRIER_FL_NUM")==2314) &
        (F.col("ORIGIN_AIRPORT_ID")==11042) &
        (F.col("DEST_AIRPORT_ID")==12264))

display(example_dupe_flight)

# COMMAND ----------

# MAGIC %md
# MAGIC We can see in this example that one of the flights has two of the same exact occurrences. To remedy this problem, we can retrieve only distinct records to remove duplicates. After the cleaning process, there are no more duplicate flights in the dataset.

# COMMAND ----------

# DBTITLE 0,Drop duplicates
clean_airlines_df = ft_select_airlines_df \
    .na.drop(subset=primary_keys) \
    .distinct()

# COMMAND ----------

dupe_flight_count = clean_airlines_df.select(primary_keys).groupBy(primary_keys).count().filter(F.col("count") > 1).count()
print(f"Number of duplicate records: {dupe_flight_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Dictionary
# MAGIC The data dictionary contains all of the fields in the clean airlines dataframe.
# MAGIC 
# MAGIC Field                 | Description                                                                                                | Data Type
# MAGIC --------------------- | ---------------------------------------------------------------------------------------------------------- | ---------
# MAGIC YEAR                  | Year                                                                                                       | Integer
# MAGIC QUARTER               | Quarter (1-4)                                                                                              | Integer
# MAGIC MONTH                 | Month                                                                                                      | Integer
# MAGIC DAY_OF_MONTH          | Day of month                                                                                               | Integer
# MAGIC DAY_OF_WEEK           | Day of week                                                                                                | Integer
# MAGIC OP_UNIQUE_CARRIER     | Unique Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years.                                                                 | String
# MAGIC TAIL_NUM              | Tail Number used to identify a specific airplane                                                           | String
# MAGIC OP_CARRIER_FL_NUM     | Flight Number                                                                                              | Integer
# MAGIC ORIGIN                | Origin Airport                                                                                             | String
# MAGIC ORIGIN_AIRPORT_ID     | Origin Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.                            | Integer
# MAGIC ORIGIN_CITY_NAME      | Origin Airport, City Name                                                                                  | String
# MAGIC ORIGIN_STATE_NM       | Origin Airport, State Name                                                                                 | String
# MAGIC ORIGIN_STATE_ABR      | Origin Airport, State Code                                                                                 | Integer
# MAGIC DEST                  | Destination Airport                                                                                        | String
# MAGIC DEST_AIRPORT_ID       | Destination Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.                   | Integer
# MAGIC DEST_CITY_NAME        | Destination Airport, City Name                                                                             | String
# MAGIC DEST_STATE_NM         | Destination Airport, State Name                                                                            | String
# MAGIC DEST_STATE_ABR        | Destination Airport, State Code                                                                            | String
# MAGIC CRS_DEP_TIME          | CRS Departure Time (local time: hhmm)                                                                      | Integer
# MAGIC DEP_DEL15             | Departure Delay Indicator, 15 Minutes or More (1=Yes)                                                      | Integer

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Summary Statistics
# MAGIC The new clean airlines dataframe contains 34,794,709 rows and 20 columns.

# COMMAND ----------

# DBTITLE 0,Display clean row and column count
print(f"Clean Airlines Row Count: {clean_airlines_df.count()}")
print(f"Clean Airlines Column Count: {len(clean_airlines_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC We created summary statistics of our cleaned table from Databricks Data Profile to understand the distribution of each column and possible similarities while supplementing additional graphs to better understand the data. Looking at the data profile, we can see that there are no null values in any of the fields, so the data cleaning was successful. Another interesting discovery is that there are 82.88% of flights that were not delayed, so we hav an imbalanced dataset. Therefore, we would have to use precision or recall as a metric or balance the dataset to have an equal number of delayed and non-delayed flights so that we can use accuracy as a metric.

# COMMAND ----------

# DBTITLE 0,Run Data Profile over clean airlines dataframe
dbutils.data.summarize(clean_airlines_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Flights by Year and Month
# MAGIC For a better understanding of the columns, we have zoomed in on the histograms `YEAR` and `MONTH` and created visualizations. We can see in "Histogram of Flights Per Year" that there were many flights in 2018 and 2019, but dramatically fell in 2020. This trend could be a result of COVID-19 and the travel restrictions being imposed. In "Histogram of Flights Per Year and Month", we see a similar trend in early 2020, there are extremely few flights.

# COMMAND ----------

# DBTITLE 0,Histogram of Flights Per Year and Month
# See Visualization
display(clean_airlines_df \
    .groupBy("YEAR", "MONTH").count().orderBy("YEAR", "MONTH") \
    .withColumn("YEAR_MONTH", F.concat(F.col("YEAR"), F.lit("-"), F.col("MONTH"))))

# COMMAND ----------

# MAGIC %md
# MAGIC Additionally, we also took a look at the histogram of delayed flights versus non-delayed flights as shown in "Delay vs No Delay Flight Histogram". We can see that across all months that there were mostly non-delayed flights in comparison to delayed flights. If we look at "Ratio of Delay to No Delay Histogram", we can see a similar dip in 2020 that was present in the histograms of flight count by year and month, but the pattern seems to last longer throughout 2020.

# COMMAND ----------

# See Visualization
display(clean_airlines_df \
    .select("YEAR", "MONTH", "DEP_DEL15") \
    .withColumn("DELAY", F.when(F.col("DEP_DEL15")==1, 1).otherwise(0)) \
    .withColumn("NO_DELAY", F.when(F.col("DEP_DEL15")==0, 1).otherwise(0)) \
    .drop("DEP_DEL15") \
    .groupBy("YEAR", "MONTH") \
    .agg(F.sum(F.col("DELAY")).alias("SUM_DELAY"), F.sum(F.col("NO_DELAY")).alias("SUM_NO_DELAY")) \
    .withColumn("RATIO", F.col("SUM_DELAY") / F.col("SUM_NO_DELAY")) \
    .orderBy("YEAR", "MONTH") \
    .withColumn("YEAR_MONTH", F.concat(F.col("YEAR"), F.lit("-"), F.col("MONTH"))))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Departure Times
# MAGIC The departure times of Departure and Departure times shows a distribution where flights peak around 8:00 am, 12:00 pm, 3:00 pm, and 6:00 pm. We expect these times to be the busiest for airlines, which means more delays in flights.

# COMMAND ----------

display(clean_airlines_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Correlation Matrix
# MAGIC 
# MAGIC In order to better understand our data, we created a correlation matrix between each variable. The string data type columns were transformed into integers in order to calculate Pearson's correlation matrix. We can see a dark blue diagonal line which indicates that each column is perfectly correlated with itself. Origin and destination airports are highly correlated with its respective state and airport code. The correlated fields such as `ORIGIN_AIRPORT_ID` and `DEST_AIRPORT_ID` are used to join on other datasets and will be dropped later.

# COMMAND ----------

def plot_correlation_matrix(df, method='pearson', title="Correlation Matrix"):
    '''
    Plots the heatmap of the correlation matrix
    '''
    from pyspark.mllib.stat import Statistics
    import pandas as pd
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer
    
    # Creates a new column with factorized string columns ending in "_INDEX"
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_INDEX").fit(df) for c in df.columns if (df.schema[c].dataType == T.StringType())]
    
    # Drops string columns
    string_cols = [c for c in df.columns if (df.schema[c].dataType == T.StringType())]
    pipeline = Pipeline(stages=indexers)
    factorized_df = pipeline.fit(df).transform(df).drop(*string_cols)

    # Calculates correlation matrix
    factorized_rdd = factorized_df.rdd.map(lambda row: row[0:])
    corr_mat = Statistics.corr(factorized_rdd, method=method)
    corr_mat_df = pd.DataFrame(corr_mat,
                    columns=factorized_df.columns, 
                    index=factorized_df.columns)
    
    # Plot correlation matrix in seaborn
    plt.subplots(figsize=(10,10))  
    plt.title(title)
    ax = sns.heatmap(
        corr_mat_df, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        horizontalalignment='right'
    );
    return corr_mat_df

# COMMAND ----------

_ = plot_correlation_matrix(clean_airlines_df, title="Clean Airlines Correlation Matrix")

# COMMAND ----------

# DBTITLE 1,Write into blob storage
clean_airlines_df.write.mode("overwrite").parquet(f"{blob_url}/clean_airlines_2")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stations

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Description
# MAGIC 
# MAGIC  Each record of the stations dataset is an edge between two stations. The unique identifier of weather station is `station_id` whereas airport is `neighbor_id`. We can also see that the relationship is many to many. For a given `station_id` and `neighbor_id` we've displayed below, we see multiple stations connected to it, and even to itself. However, there are no guarantees that a station-to-station relationship goes both ways. For example, in the picture below, there are many bidirectional arrows, but it's possible where one station (Station B) can see the other (Station F). Stations also don't necessarily also need to point to each other, like Station A and Station D.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/sysung/w261-final-project/blob/master/full_workflow_pipeline-Page-2.drawio.png?raw=true" width=50%>

# COMMAND ----------

# MAGIC %md
# MAGIC Other than the ids, each record also contains the latitude, longitude, and other geographic information about both stations. There is also a distance between two stations which can be useful in finding the nearest station

# COMMAND ----------

display(raw_stations_df.filter(F.col('station_id')==69002093218))

# COMMAND ----------

display(raw_stations_df.filter(F.col('neighbor_id')==6900209321869002093218))

# COMMAND ----------

# MAGIC %md
# MAGIC The raw dataset contains 5,004,169 rows and 12 columns, so we can expect multiple duplicate stations.

# COMMAND ----------

print(f"Raw Stations Row Count: {raw_stations_df.count()}")
print(f"Raw Stations Column Count: {len(raw_stations_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Station Distance to Neighbor
# MAGIC Within this dataframe, logically, we found that the station’s `distance_to_neighbor` would be the most important feature to clean our dataset, given that the closer a station is to an airport in proximity, the higher likelihood that the weather forecast/data is accurate. Since incremental weather chagnes can impact flight delays, we decided to run a groupBy on states and measure each station’s average distance to neighbor, expecting more remote states to be farther away from stations. The chart below shows the output. 

# COMMAND ----------

dist_by_state = raw_stations_df.select('distance_to_neighbor', 'neighbor_state')
df_by_state = dist_by_state.groupBy(dist_by_state['neighbor_state']).mean('distance_to_neighbor')

df_by_state.toPandas().sort_values('avg(distance_to_neighbor)', ascending=False).plot.bar(x='neighbor_state', y='avg(distance_to_neighbor)', figsize=(30,10), ylabel = 'Distance', xlabel = 'Neighbor State', title='Average Distance to Neighbor by State', fontsize=25)
plt.xlabel('Neighbor State', size=30)
plt.ylabel('Average Distance', size=30)
plt.title('Average Distance to Neighbor by State', size = 40)
plt.legend(prop={'size':20})
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The chart above confirmed our expectations around more remote stations in states such as Hawaii, Alaska, and the Virgin Islands having further average distances to neighbors. We also explored using the Haversine formula to calculate an alternative distance to neighbor to quantify the shortest distance between a station’s latitude and longitude and its neighbor’s latitude and longitude, but the results did not have a significant impact on our findings; therefore, we decided to stick with the stock distance to neighbor initially provided in the dataframe.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Duplicate Stations
# MAGIC 
# MAGIC With almost 5 million rows, we can see that there is a high possibility that there are duplicate stations. Since each row represents an edge between stations, there is a possibility that a station can point to numerous airports and vice versa. To ensure that each row is distinct such that each station only contains a unique airport, we decided to use the closest airport distance to station using a window function. 

# COMMAND ----------

w3 = Window.partitionBy("neighbor_call").orderBy(F.col("distance_to_neighbor").asc())
clean_stations_df = raw_stations_df.withColumn("row", F.row_number().over(w3)) \
                              .filter(F.col("row") == 1).drop("row")

# COMMAND ----------

display(clean_stations_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Summary Statistic
# MAGIC After the cleaning of the dataframe, we are left with 2,229 rows which greatly reduces the complexity of our joins

# COMMAND ----------

print(f"Clean Stations Row Count: {clean_stations_df.count()}")
print(f"Clean Stations Column Count: {len(clean_stations_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Null Count
# MAGIC 
# MAGIC Next, we looked for null values within the dataframe; fortunately, there were no null values contained within the dataframe. We then looked at the dataframe at a high level, using the summarize function to better understand summary statistics of each of the features as is shown below.
# MAGIC 
# MAGIC We see that the distributions of `lat` and `neighbor_lat` are extremely similar as well as `lon` and `neighbor_lon`. This is expected since stations are located in the United States.
# MAGIC 
# MAGIC After the removal of duplicate stations, the distance to neighbor is all zero, which means that most of the stations, if not all, are actually pointing to itself.

# COMMAND ----------

dbutils.data.summarize(clean_stations_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Dictionary
# MAGIC When performing exploratory data analysis on the raw features of the stations, we first looked at the underlying data including its definition by referencing a data dictionary that we compiled from various sources, which we show below.
# MAGIC 
# MAGIC Field                | Description                                                                                                                                      | Data Type
# MAGIC -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------
# MAGIC usaf                 | A character string identifying the fixed weather station from the USAF(United States Air Force) Master Station Catalog                           | STRING
# MAGIC wban                 | WBAN(Weather Bureau, Air Force and Navy) is a five-digit station identifier for digital data storage and general station identification purposes | STRING
# MAGIC station_id           | Identifier for the weather station                                                                                                               | STRING
# MAGIC lat                  | Latitude coordinates for the weather station                                                                                                     | DOUBLE
# MAGIC lon                  | Longitude coordinates for the weather station                                                                                                    | DOUBLE
# MAGIC neighbor_id          | Identifier for the airport close to the weather station                                                                                          | STRING
# MAGIC neighbor_name        | Name of the airport close to the weather station                                                                                                 | STRING 
# MAGIC neighbor_state       | State of the airport close to the weather station                                                                                                | STRING 
# MAGIC neighbor_call        | A four-letter alphanumeric code representing the ICAO airport code or location indicator                                                         | STRING 
# MAGIC neighbor_lat         | Latitude coordinates of the airport close to the weather station                                                                                 | DOUBLE
# MAGIC neighbor_lon         | Latitude coordinates of the airport close to the weather station                                                                                 | DOUBLE
# MAGIC distance_to_neighbor |  Distance of the airport to the weather station.                                                                                                 | STRING

# COMMAND ----------

# DBTITLE 1,Write into blob storage
clean_stations_df.write.mode("overwrite").parquet(f"{blob_url}/clean_stations_2")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather
# MAGIC 
# MAGIC Within the weather dataframe, there were 126 columns and 898,983,399 rows to explore, with features ranging from `STATION` to `HourlyVisibility` and `HourlyWindSpeed`.  The weather data is important for predicting flight delays as pilots cannot always fly safely in inclement weather. 

# COMMAND ----------

print(f"Raw Weather Row Count: {raw_weather_df.count()}")
print(f"Raw Weather Column Count: {len(raw_weather_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Feature Selection
# MAGIC After ingesting and profiling the data, we learned that the majority of the columns have missing or null values.  We performed a variety of pre-processing steps including dropping data that had null values and casting appropriate data types such as integer for columns containing temperature, speed, or direction and double for `latitude`, `longitude`, `elevation`, `HourlyAltimeterSetting`, `HourlySeaLevelPressure`, `HourlyStationPressue`, and `HourlyVisibility`.  The columns with at least 70% missing or null values were excluded from the dataset because we still had plenty of weather statistics post-exclusion.
# MAGIC 
# MAGIC Additional fields we decided to drop were monthly, daily, and backup features because these features did not give us the amount of granularity that we needed in our analysis. This allowed us to drop numerous features while maintaining a the same information.

# COMMAND ----------

weather_cols = [
    "STATION",
    "DATE",
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "NAME",
    "REPORT_TYPE",
    "SOURCE",
    "HourlyAltimeterSetting",
    "HourlyDewPointTemperature",
    "HourlyDryBulbTemperature",
    "HourlyRelativeHumidity",
    "HourlySkyConditions",
    "HourlyStationPressure",
    "HourlyVisibility",
    "HourlyWetBulbTemperature",
    "HourlyWindDirection",
    "HourlyWindSpeed"
]

clean_weather_df = raw_weather_df \
    .select(weather_cols) \
    .withColumn("HourlyDewPointTemperature", F.col("HourlyDewPointTemperature").cast(T.DoubleType())) \
    .withColumn("HourlyDryBulbTemperature", F.col("HourlyDryBulbTemperature").cast(T.DoubleType())) \
    .withColumn("HourlyRelativeHumidity", F.col("HourlyRelativeHumidity").cast(T.DoubleType())) \
    .withColumn("HourlyWetBulbTemperature", F.col("HourlyWetBulbTemperature").cast(T.DoubleType())) \
    .withColumn("HourlyWindDirection", F.col("HourlyWindDirection").cast(T.DoubleType())) \
    .withColumn("HourlyWindSpeed", F.col("HourlyWindSpeed").cast(T.DoubleType())) \
    .withColumn("LATITUDE", F.col("LATITUDE").cast(T.DoubleType())) \
    .withColumn("LONGITUDE", F.col("LONGITUDE").cast(T.DoubleType())) \
    .withColumn("ELEVATION", F.col("ELEVATION").cast(T.DoubleType())) \
    .withColumn("HourlyAltimeterSetting", F.col("HourlyAltimeterSetting").cast(T.DoubleType())) \
    .withColumn("HourlyStationPressure", F.col("HourlyStationPressure").cast(T.DoubleType())) \
    .withColumn("HourlyVisibility", F.col("HourlyVisibility").cast(T.DoubleType())) \
    .na.drop("any") \
    .distinct()

# COMMAND ----------

display(clean_weather_df)

# COMMAND ----------

# MAGIC %md
# MAGIC The pre-processing and cleaning of the dataframe brought our total rough count down to 215,666,248 with 18 columns. 

# COMMAND ----------

print(f"Clean Weather Row Count: {clean_weather_df.count()}")
print(f"Clean Weather Column Count: {len(clean_weather_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Dictionary
# MAGIC We have created a data dictionary from our subsetted features that we deemed useful in our analysis.
# MAGIC 
# MAGIC Field                     | Description                                                        | Data Type 
# MAGIC ------------------------- | ------------------------------------------------------------------ | ---------
# MAGIC STATION                   | Identifier for the weather station                                 | STRING
# MAGIC DATE                      | Date of the when the data is pulled                                | STRING
# MAGIC LATITUDE                  | Latitude coordinates of the weather station                        | DOUBLE 
# MAGIC LONGITUDE                 |  Longitude coordinates of the weather station                      | DOUBLE
# MAGIC ELEVATION                 | Elevation of the weather station                                   | DOUBLE
# MAGIC NAME                      | Name of the weather station                                        | STRING
# MAGIC REPORT_TYPE               | The code that denotes the type of geophysical surface observation  | STRING
# MAGIC SOURCE                    | Code for weather station                                           | STRING
# MAGIC HourlyAltimeterSetting    | Altimeter Setting Value                                            | DOUBLE
# MAGIC HourlyDewPointTemperature | Dew Point Temperature of the weather station at a particular date. | DOUBLE
# MAGIC HourlyDryBulbTemperature  | Dry Bulb Temperature of the weather station at a particular date.  | DOUBLE
# MAGIC HourlyRelativeHumidity    | Relative Humidity of the weather station at a particular date.     | DOUBLE
# MAGIC HourlySkyConditions       | Description of sky conditions at a particular date                 | DOUBLE
# MAGIC HourlyStationPressure     | Station Pressure of the weather station at a particular date.      | DOUBLE
# MAGIC HourlyVisibility          | Visibility of the weather station at a particular date.            | DOUBLE
# MAGIC HourlyWetBulbTemperature  | Wet Bulb Temperature of the weather station at a particular date.  | DOUBLE
# MAGIC HourlyWindDirection       | Wind Direction of the weather station at a particular date.        | DOUBLE
# MAGIC HourlyWindSpeed           | Wind Speed of the weather station at a particular date.            | DOUBLE

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Summary Statistic
# MAGIC 
# MAGIC We then took a look at the summary statistics of the clean dataframe. We show a sample of the summary statistics below, but the full statistics can be found in the Weather EDA databricks notebook.

# COMMAND ----------

dbutils.data.summarize(clean_weather_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We then found the number of statistics for the weather by year to ensure that each year had a roughly similar level of data, using the groupBy, count, and orderBy functions. The below table allowed us to visually find that each year had around 33 million to 35 million weather points. 

# COMMAND ----------

display(clean_weather_df \
        .withColumn("YEAR", F.split(F.col("DATE"), '-').getItem(0)) \
        .groupBy("YEAR").count().orderBy("YEAR"))

# COMMAND ----------

# DBTITLE 1,Write into blob storage
clean_weather_df.write.mode("overwrite").parquet(f"{blob_url}/clean_weather_2")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join
# MAGIC 
# MAGIC To train our model with the data, we require to join all of the tables which are airlines, stations, and weather. Since doing a full join on all of the data would create an extremely large dataset, we decided to look at a timeframe of a flight and see if we could model after it. Our hypothesis is that by forecasting weather patterns that we can justify whether or not a flight should be cancelled. We use a timefrime of 4 hours prior and 1 hour after departure as arbitrary values. If required, we will extend the window to allow for more values.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/sysung/w261-final-project/blob/master/full_workflow_pipeline-Page-3.drawio.png?raw=true" width=50%/>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Preprocessing
# MAGIC 
# MAGIC However, before we join all of the tables together, we have to apply a number of preprocessing steps on each table such as formatting timezones and airport ids so that we can easily join on their corresponding fields. We also plan on applying some additional fields that pertain to COVID to see how it has impacted flight delays

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Station to Airport ID
# MAGIC To join the stations data with airlines, it needs a common key between the datasets. We have identified that stations contained a field called `neighbor_call` which contains the airport id within the last three characters. We used this information to reformat `neighbor_call` with just the necessary letters so that it can join on the airlines dataframe

# COMMAND ----------

# DBTITLE 1,Get Airport ID
join_stations_df = clean_stations_df.withColumn("neighbor_call", F.col("neighbor_call").substr(-3,3))
display(join_stations_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Timezone Format
# MAGIC In order to join on the weather data, we needed to create a uniform standard for time since airlines data was local timezone whereas weather data was UTC timezone. Therefore, we required the timezone of the airport which we grabbed from an open source file. The conversion was applied to `CRS_DEP_TIME` and `DEP_TIME`. Then, we created a unix timestamp from the UTC datetime timestamp to make date ranges easier to calculate. Our data only looks at weather data 4 hours prior and 1 hour after departures which we've labeled as `EARLY_LIMIT` and `LATE_LIMIT`.
# MAGIC 
# MAGIC It is also important to note that we are keeping the local timezone in the final join since time relative to timezone can be correlated with peak traffic.
# MAGIC 
# MAGIC Here is the data dictionary of the additional fields for the airlines dataframe.
# MAGIC Field                 | Description                                                                                                                     | Data Type
# MAGIC --------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------
# MAGIC ORIGIN_TIMEZONE       | Timezone of Origin Airport                                                                                                      | STRING
# MAGIC UNIX_CRS_DEP_TIME_UTC | Unix timestamp of CRS Departure Time in UTC                                                                                     | INTEGER
# MAGIC EARLY_LIMIT           | Unix timestamp 4 hours prior of CRS Departure Time                                                                              | INTEGER
# MAGIC LATE_LIMIT            | Unix timestamp 1 hour after CRS Departure Time                                                                                  | INTEGER
# MAGIC WEATHER_DEP_DIFF_TIME | Represents a custom field we created, which is the difference between CRS_DEP_TIME and UNIX_WEATHER_TIME from the weather table | INTEGER
# MAGIC FL_DATE               | Date in yyyy-mm-dd format                                                                                                       | STRING
# MAGIC 
# MAGIC For the weather data, we added an additional field to join with the airlines data
# MAGIC Field             | Description                           | Data Type
# MAGIC ----------------- | ------------------------------------- | ---------
# MAGIC UNIX_WEATHER_TIME | Unix timestamp of weather time in UTC | INTEGER 
# MAGIC FL_DATE           | Date in yyyy-mm-dd format             | STRING

# COMMAND ----------

# DBTITLE 1,Airport to timezone conversion dictionary
import requests

# Get airport timezone from url
link = "https://raw.githubusercontent.com/hroptatyr/dateutils/tzmaps/iata.tzmap"
f = requests.get(link)
iata_data = f.text.split("\n")

# Create dictionary of airports to timezone
missing_iata = ["XWA\tAmerica/Chicago"]
iata_rdd = sc.parallelize(iata_data).union(sc.parallelize(missing_iata))
iata_rdd = iata_rdd.map(lambda x: tuple(x.split('\t'))).filter(lambda x: len(x) == 2)
iata_dict_bc = sc.broadcast(dict(iata_rdd.collect()))

# COMMAND ----------

# DBTITLE 1,Functions to format date and time
# Formats the time with a colon {123 -> 01:23}
def format_time_column(colname):
    return F.regexp_replace(F.format_string("%04d", F.col(colname) % 2400), "(\\d{2})(\\d{2})", "$1:$2")

# Formats the date like FL_DATE (2022-11-11)
def format_date_column():
    return F.concat(F.col("YEAR"), F.lit("-"), F.lpad(F.col("MONTH"), 2, "0"), F.lit("-"), F.lpad(F.col("DAY_OF_MONTH"), 2, "0"))

# Converts the date [yyyy-MM-dd HH:mm:ss] into a unix timestamp
def format_unix_column(colname, timezone):
    return F.unix_timestamp(F.to_utc_timestamp(F.concat(format_date_column(), F.lit(" "), format_time_column(colname)), F.col(timezone)))

# Get timezone
@udf(returnType=T.StringType()) 
def get_timezone(AIRPORT):
    return iata_dict_bc.value[AIRPORT]

# COMMAND ----------

# DBTITLE 1,Call functions for datetime conversions on tables
early_threshold = 4
late_threshold = 1

join_airlines_df = clean_airlines_df \
    .withColumn("ORIGIN_TIMEZONE", get_timezone(F.col("ORIGIN"))) \
    .withColumn("UNIX_CRS_DEP_TIME_UTC", format_unix_column("CRS_DEP_TIME", "ORIGIN_TIMEZONE")) \
    .withColumn('EARLY_LIMIT', F.col("UNIX_CRS_DEP_TIME_UTC") - (early_threshold*3600)) \
    .withColumn('LATE_LIMIT', F.col("UNIX_CRS_DEP_TIME_UTC") + (late_threshold*3600)) \
    .withColumn('FL_DATE', format_date_column())

join_weather_df = clean_weather_df.withColumn('DATE', F.regexp_replace('DATE', 'T', ' ')) \
    .withColumn("UNIX_WEATHER_TIME", F.unix_timestamp(F.col("DATE"),"yyyy-MM-dd HH:mm:ss")) \
    .withColumn('FL_DATE', F.split(F.col('DATE'), ' ').getItem(0))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Join Process
# MAGIC We took a two-pronged approach to joining the dataframes. We started by joining the ORIGIN feature from the Airlines dataframe with the airport_id column from the Stations dataframe as well as the ORIGIN_STATE_ABR feature from the Airlines dataframe with neighbor_state from the Stations dataframe.

# COMMAND ----------

join_df1 = join_airlines_df.join(
    join_stations_df,
    (join_airlines_df.ORIGIN == join_stations_df.neighbor_call) & \
    (join_airlines_df.ORIGIN_STATE_ABR == join_stations_df.neighbor_state)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we joined the stations and date feature from the Airlines and Stations dataframes. We also wanted to focus on weather data within the timeframe of 4 hours before the flight and 1 hour after the flight.

# COMMAND ----------

join_df2 = join_df1.join(
    join_weather_df,
    (join_df1.station_id == join_weather_df.STATION) & \
    (join_df1.FL_DATE == join_weather_df.FL_DATE),
    "left"
).filter(
    (join_weather_df.UNIX_WEATHER_TIME <= join_df1.LATE_LIMIT) &
    (join_weather_df.UNIX_WEATHER_TIME >= join_df1.EARLY_LIMIT)
)

# COMMAND ----------

# MAGIC %md
# MAGIC A diagram of our two-tiered join strategy is shown below.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/sysung/w261-final-project/master/data_joins.png" width=50%>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Dimensionality Reduction
# MAGIC 
# MAGIC After joining the tables, we can drop some of the keys that are no longer relevant such as join keys and highly correlated fields. We also added an additional field called `WEATHER_DEP_DIFF_TIME` which is the amount of time between the flight departure time and the weather reading time. If the time difference is negative, then the weather reading time occurred after the flight departure

# COMMAND ----------

final_df = join_df2.select(
    F.col('DEP_DEL15'),
    F.col('UNIX_WEATHER_TIME'),
    F.col('UNIX_CRS_DEP_TIME_UTC'),
    F.col('LATITUDE'),
    F.col('LONGITUDE'),
    F.col('ELEVATION'),
    F.col('HourlyAltimeterSetting').alias("HOURLY_ALTIMETER_SETTING"),
    F.col('HourlyDewPointTemperature').alias("HOURLY_DEW_POINT_TEMP"),
    F.col('HourlyDryBulbTemperature').alias('HOURLY_DRY_BULB_TEMP'),
    F.col('HourlyRelativeHumidity').alias('HOURLY_RELATIVE_HUMIDITY'),
    F.col('HourlyStationPressure').alias('HOURLY_STATION_PRESSURE'),
    F.col('HourlyVisibility').alias('HOURLY_VISIBILITY'),
    F.col('HourlyWetBulbTemperature').alias('HOURLY_WET_BULB_TEMP'),
    F.col('HourlyWindDirection').alias('HOURLY_WIND_DIRECTION'),
    F.col('HourlyWindSpeed').alias('HOURLY_WIND_SPEED'),
    F.col('REPORT_TYPE'),
    F.col('SOURCE')
)

# COMMAND ----------

display(final_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Altogether, the join took around 5.84 minutes to display and 7.42 minutes to write to blob storage on four 4-core 16-gb worker nodes. 

# COMMAND ----------

# DBTITLE 1,Write to blob storage
final_df.write.mode("overwrite").parquet(f"{blob_url}/final_df_9")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Metrics
# MAGIC 
# MAGIC > - What other kinds of metrics can we gather even if they are simple?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary Statistic
# MAGIC We analyzed the final dataframe and saw that there were 208,066,549 rows and 17 columns with no missing data, so our dataset is clean. We also saw that there was a huge imbalance in data where there were more non-delayed flights than there are delayed flights. Therefore, we want to calculate the recall since we want to ensure that the we maximize the probability that we identified a delayed flight correctly.

# COMMAND ----------

print(f"Final Dataframe Row Count: {final_df.count()}")
print(f"Final Datafrmae Column Count: {len(final_df.columns)}")

# COMMAND ----------

print(f'Number of Non-delayed Flights: {final_df.filter(F.col("DEP_DEL15")==0).count()}')
print(f'Number of Delayed Flights {final_df.filter(F.col("DEP_DEL15")==1).count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlation Matrix
# MAGIC 
# MAGIC We also ran a correlation matrix on our final dataframe to better understand the underlying relationships between the various features. The correlation matrix was run on a subset of data since the full dataset was too large and often caused the cluster to crash. We hope that by sampling the dataset to a certain degree, we can keep the integrity of the distribution and not affect the pearson's correlation coefficient by too much while increasing the process speed.
# MAGIC 
# MAGIC The output made sense logically, given that the `HOURLY_STATION_PRESSURE` was negatively correlated to the `ELEVATION`. One would expect the hourly station pressure to drop when elevation rises. Conversely, `HOURLY_WET_BULB_TEMP`, `HOURLY_DRY_BULB_TEMP` and `HOURLY_DEW_POINT_TEMP` were all positively correlated. The dry bulb temperature is the ambient air temperature that is measured by regular thermometers, while the wet bulb temperature is measured by thermometers that are wrapped in wetted wicks. The dry and wet bulb temperatures are empirically similar under relative humidity conditions.
# MAGIC 
# MAGIC We also see that `UNIX_WEATHER_TIME` and `UNIX_CRS_DEP_TIME_UTC` have high correlation. Since these numbers are large as UNIX timestamps, we expect the differences between the two fields to be minimal. When training our model we should do feature engineering to normalize these values or convert them back to date timestamps.

# COMMAND ----------

sample_final_df = final_df.sample(False, 0.01)
print(f"Sample Size: {sample_final_df.count()}")
_ = plot_correlation_matrix(sample_final_df, title="Final Dataframe Correlation Matrix")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model
# MAGIC 
# MAGIC In this section, we will be discussing our modeling process including the different featured that were added and model pipelines. Our dataset contains classified data, so we can use supervised learning models to train on our data.
# MAGIC 
# MAGIC As our main metric, we noticed that there was a major imbalance of data. Of the 208,066,549 weather readings from flights, 172,093,071 were listed as no delay where as 35,973,478 were delays. Note that each record is a unique flight, rather contain a timeframe within a specific flight.

# COMMAND ----------

display(final_df.groupBy('DEP_DEL15').count())

# COMMAND ----------

# MAGIC %md
# MAGIC Due to this imbalanced nature, we could not use accuracy since it relies on a balanced data set. Precision measures how many flights were correctly predicted as delayed over the true number of delayed flights, whereas recall measures the number of correctly predicted delayed flights over all positive cases for both delay and non-delayed flights. Specificity measures how many non delayed predictions were correct and F1-Score combines both precisino and recall.
# MAGIC 
# MAGIC We aim to minimize false negatives, so flights that predicted to not be delayed but are actually delayed. This is because if we misinterpret a flight as being on time but is actually late, there is a lot of economic loss for us. On the other hand, if we predict a flight to be late but is actually on time, it may actully increase customer satisfaction knowing that their flight has arrived earlier than expected. 
# MAGIC 
# MAGIC Therefore, we will use recall as our main metric, but will also list precision and F1 score for curiosity sake.
# MAGIC 
# MAGIC <script
# MAGIC   src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
# MAGIC   type="text/javascript">
# MAGIC </script>
# MAGIC 
# MAGIC $$Recall = \frac{TP}{TP+FN} $$

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Steps

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering
# MAGIC 
# MAGIC > Feature Engineering
# MAGIC > - Describe newly engineering features that were added (in the form of feature families if possible)
# MAGIC > - Show the impact (if any) that these new features added to the model in the modeling pipelines section below
# MAGIC > - Explain why you chose the method and approach in presentation/results/discussion sections
# MAGIC > - Create new features that are highly predictive:
# MAGIC >     - at least one time-based feature, e.g., recency, frequency, monetary, (RFM)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### COVID Impact
# MAGIC In our EDA, we discovered that COVID had a huge impact on flights and the volume of it. To compensate for this anomaly, we decided to add a dummy variable called `COVID` which marked the beginning of travel bans for international flights during the Trump Administration. 
# MAGIC - TODO: What's start (March 2020) and end date

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Time Difference from Flight Departure Time and Weather Reading
# MAGIC     ```'WEATHER_DEP_DIFF_TIME' = 
# MAGIC     F.col('UNIX_CRS_DEP_TIME_UTC') - F.col('UNIX_WEATHER_TIME')```

# COMMAND ----------

# MAGIC %md
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
# MAGIC > Feedback from previous phase
# MAGIC > - Don't use accuracy, your dataset is not balanced. There are way more flights on time then delayed. Please describe why you are using cross validation
# MAGIC > - Need more details within this section. Why are you using the success metrics that you are using? Make a business case and why that specific metric of success fits the case. Do not use accuracy as the data is unbalanced.

# COMMAND ----------

# MAGIC %md
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
# MAGIC 
# MAGIC > Feedback from previous phase
# MAGIC > - Results section was lacking any analysis. Why did your last fold do not as well? What about the threshold in logistic regression?

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
# MAGIC 
# MAGIC > Feedback from previous phase
# MAGIC > - In the abstract you mentioned that weather had little to no impact but that same sentiment was not carried out in the conclusion. It actually seemed like certain weather features did have an impact. There was no restatement of the hypothesis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Credit Assignment
# MAGIC > Updated Credit Assignment Plan
# MAGIC > - Credit assignment plan updates (who does/did what and when, amount of effort in terms of person hours, start and end dates, estimated improvement in terms of key metrics) in Table format
# MAGIC > - No Credit assignment plan means ZERO points
# MAGIC > - A credit assignment plan not in Table format means ZERO points
# MAGIC > - No start and end dates and (budgeted) hours of effort mean an incomplete plan. This may result in zero points.
# MAGIC 
# MAGIC > Feedback from previous phase
# MAGIC > - Please follow the instructions regarding the submission of files, mainly that you need to submit an HTML file of the main report (NOT a ZIP file), same with the ipynb. The credit assignment plan needs to have the amount of time it took to complete each task and the tasks need to be more detailed.
# MAGIC 
# MAGIC | Task | Estimated Improvement | Hours Spent | Start Date | End Date | Team Member
# MAGIC | ---- | --------------------- | ----------- | ---------- | -------- | ----------- |
# MAGIC | Add list of tasks to do for Phase 3                                | 0 | 0.5 | 11/18 | 11/18 | Steven Sung |
# MAGIC | Migrate Introduction from Google Doc to Jupyter Notebook           | 0 | 2  | 11/19 | 11/19 | Steven Sung |
# MAGIC | Combine all 5 Jupyter Notebooks into Phase 3 Master Notebook       | 0 | 24 | 11/19 | 11/21 | Steven Sung |
# MAGIC | Remove data balancing. Use Precision as main metric                |   | 0.25| 11/21 | 11/21 | Steven Sung |
# MAGIC | Rewrote "Abstract" section based on feedback                       | 0 | 1  | 11/22 | 11/22 | Steven Sung |
# MAGIC | Rewrote "Problem" section under "Introduction" based on feedback   | 0 | 1  | 11/22 | 11/22 | Steven Sung |
# MAGIC | Rewrote "Datasets" section under "Introduction" based on feedback  | 0 | 1  | 11/22 | 11/22 | Steven Sung |
# MAGIC | Add dataset workflow diagram from raw to clean                     | 0 | 0.5 | 11/22 | 11/22 | Steven Sung |
# MAGIC | Rewrote "Airlines" section under "Dataset" based on feedback       | 0 | 3  | 11/22 | 11/22 | Steven Sung |
# MAGIC | Rewrote "Stations" section under "Dataset" based on feedback       | 0 | 2  | 11/22 | 11/22 | Steven Sung |
# MAGIC | Migrate and rewrote "Model" section based on feedback              | 0 | 7  | 11/23 | 11/23 | Steven Sung |
# MAGIC | Changed join diagram                                               | 0 | 1  | 11/22 | 11/22 | Justin Chan |

# COMMAND ----------

|
