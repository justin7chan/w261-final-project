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
raw_weather_df = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data/")
raw_stations_df = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")

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
clean_airlines_df = spark.read.parquet(f"{blob_url}/clean_airlines_1")
clean_weather_df = spark.read.parquet(f"{blob_url}/clean_weather_1")
clean_stations_df = spark.read.parquet(f"{blob_url}/clean_stations_2")
final_df = spark.read.parquet(f"{blob_url}/final_df_5")

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
# MAGIC > - **Task to be tackled**
# MAGIC > - **Provide diagrams to aid understanding the workflow**
# MAGIC 
# MAGIC According to the U.S. Bureau of Transportation Statistics, over the last 10 years, on average 18.74% of scheduled flights are delayed by 15 minutes or more per year, with 2022 representing the highest percentage (21.16%) of flights delayed since 2014. Delays represent a significant headache for both travelers and airlines alike. Researchers from the University of California Berkeley commissioned by the Federal Aviation Administration found that domestic flight delays cost the U.S. economy $32.9bn, with about half of the cost borne by airline passengers and the balance borne by airline operators. Today, flight delays remain a widespread issue that places a significant burden on airlines, travelers, and the broader economy at large.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Problem
# MAGIC 
# MAGIC The problem is to predict flight delays for US-bound and departing flights using weather data to avoid economic losses for airlines and to reduce passenger inconvenience. We define a flight delay as a any flight departing 15 minutes later than expected scheduled departure time. Data on severe weather such as tornados, heavy winds, and floods is captured in weather stations, where we can correlate this data with air flight delays.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Datasets
# MAGIC 
# MAGIC We will leverage three data sets including (i) flight data, (ii) weather data, and (iii) airport data. The data will be ingested from a Parquet file into Pandas data frames for ease of use and wrangling. The airlines data includes a subset of passenger flights' on-time performance data derived from the U.S Department of Transportation over the time window spanning 2015 to 2021 by quarter. Similarly, the weather data spans 2015 to 2021 and is derived from the National Oceanic and Atmospheric Administration repository. The stations dataset contains metadata about weather station location and airports nearest to it. 
# MAGIC 
# MAGIC To gain a better understanding of the data, we will perform exploratory data analysis on the three main dataframes including the (i) airlines, (ii) stations, and (iii) weather dataframes. Additionally, we'll analyze our final joined dataframe, which combines these three dataframes into one master table.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airlines
# MAGIC The airline dataset containing flights from 2015 to 2021 inclusive has 74,177,433 total rows and 109 fields of data. Each row corresponds to an individual flight within the United States. Our response variable is `DEP_DEL15` which is a binary variable which states if a flight is delayed by more than 15 minutes or not. We can use this information to do some intial EDA to cut down the number of records by cleaning the dataframe.

# COMMAND ----------

# DBTITLE 1,Display raw row and column count
print(f"Raw Airlines Row Count: {raw_airlines_df.count()}")
print(f"Raw Airlines Column Count: {len(raw_airlines_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC We also took a look at a summarization of the raw dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Null Counts
# MAGIC 
# MAGIC The first form of EDA that we did was look for null counts in the raw data set. The histogram is sorted by the indices that the column appears in the raw table. We've removed all columns which have no nulls so that we can see which field has the highest null counts. Upon further inspection of the data, most of the columns that have high null values are canceled or diverted flights.The high null values means that there are very few flights that were not enroute. This may be a good indicator of delayed flights, but with so many nulls, it would be safer to remove the columns.

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

# DBTITLE 1,Plot histogram of non-zero null counts
null_pd = null_df.filter(F.col("nonzero_nulls") > 0).pandas_api().to_pandas()

chart = sns.catplot(y="nonzero_nulls", x="fields", kind="bar", data=null_pd, color="blue", height=5, aspect=15/5)
chart.set_axis_labels("Field Names", "Null Count")
chart.set(title='Non-Zero Null Count of Fields')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Feature Selection
# MAGIC To clean the data, we removed all of the columns that have an extremely high null value count. This will reduce our dimensions and remove the curse of dimensionality. There are also other features that were highly collinear to each other. There are a few exceptions in which we kept like airport information because it will be used to join on the different tables. In the final dataframe, it will be dropped.
# MAGIC 
# MAGIC For the columns that have some null values, we can easily fill some of the columns using basic arithmetics. We used `CRS_DEP_TIME` and `DEP_TIME` to calculate the `DEP_DELAY`, which is then used to calculate `DEP_DEL15` based on a simple condition statement if `DEP_DELAY` is greater than or equal to 15 or not.
# MAGIC 
# MAGIC If some rows still contained nulls, we removed the row altogether as it would only be a few rows lost.

# COMMAND ----------

# DBTITLE 1,Drop columns
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

drop_cols_airlines_df = raw_airlines_df \
    .select(subset_cols) \
    .na.drop(subset=na_drop_cols) \
    .withColumn("DEP_DEL15", F.when(F.col("DEP_DELAY") >= 15, 1).otherwise(0)) \
    .withColumn("DEP_DELAY", F.col("DEP_TIME") - F.col("CRS_DEP_TIME")) \
    .drop(F.col("DEP_TIME"))

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
display(drop_cols_airlines_df.select(primary_keys).groupBy(primary_keys).count().filter(F.col("count") > 0))

# COMMAND ----------

# DBTITLE 1,Find an example of a duplicate flight
example_dupe_flight = drop_cols_airlines_df \
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
# MAGIC We can see in this example that one of the flights has two of the same exact occurrences. To remedy this problem, we can retrieve only distinct records to remove duplicates.

# COMMAND ----------

# DBTITLE 1,Drop duplicates
prim_keys_airlines_df = drop_cols_airlines_df \
    .na.drop(subset=primary_keys) \
    .distinct()

dupe_flights = prim_keys_airlines_df.select(primary_keys).groupBy(primary_keys).count().filter(F.col("count") > 1)

# COMMAND ----------

display(dupe_flights)

# COMMAND ----------

# MAGIC %md
# MAGIC After the deduping of the dataframe, we were still left with three anomalous records. Since it was a small subset of records, we decided to remove it altogether

# COMMAND ----------

# DBTITLE 1,Remove last few duplicate rows
window_agg = Window.partitionBy(primary_keys)

clean_airlines_df = prim_keys_airlines_df \
    .withColumn("lit1", F.lit(1)) \
    .withColumn("count", F.sum(F.col("lit1")).over(window_agg)) \
    .filter(F.col("count")==1) \
    .drop(*["lit1", "count"]) \
    .cache()

# COMMAND ----------

dupe_count = clean_airlines_df.select(primary_keys).groupBy(primary_keys).count().filter(F.col("count") > 1).count()
print(f"Number of duplicate records: {dupe_count}")

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
# MAGIC DEP_DELAY             | Difference in minutes between scheduled and actual departure time. Early departures show negative numbers. | Integer
# MAGIC DEP_DEL15             | Departure Delay Indicator, 15 Minutes or More (1=Yes)                                                      | Integer

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Summary Statistics
# MAGIC The new clean airlines dataframe contains 41,577,767 rows and 21 columns.

# COMMAND ----------

# DBTITLE 1,Display clean row and column count
print(f"Clean Airlines Row Count: {clean_airlines_df.count()}")
print(f"Clean Airlines Column Count: {len(clean_airlines_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC We created summary statistics of our cleaned table from Databricks Data Profile to understand the distribution of each column and possible similarities while supplementing additional graphs to better understand the data. Looking at the data profile, we can see that there are no null values in any of the fields, so the data cleaning was successful. Another interesting discovery is that there are 82.88% of flights that were not delayed, so we hav an imbalanced dataset. Therefore, we would have to use precision or recall as a metric or balance the dataset to have an equal number of delayed and non-delayed flights so that we can use accuracy as a metric.

# COMMAND ----------

# DBTITLE 1,Run Data Profile over clean airlines dataframe
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
# MAGIC ##### Geoplot
# MAGIC 
# MAGIC As an additional EDA, we have plotted on the United States map the number of flights coming in and out of each state. In both maps, the shades of blue are very similar where the highest flight traffic occurs around California, Texas, and Florida. When we train our model, it can probably assume that if a flight comes in or out of those states that there is a higher probability for flight delays.

# COMMAND ----------

# DBTITLE 1,Origin Flights Map
# See Visualization
display(clean_airlines_df.groupBy("ORIGIN_STATE_ABR").count())

# COMMAND ----------

# DBTITLE 1,Destination Flights Map
# See Visualization
display(clean_airlines_df.groupBy("DEST_STATE_ABR").count())

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
    factorized_df = pipeline.fit(clean_airlines_df).transform(clean_airlines_df).drop(*string_cols)

    factorized_rdd = factorized_df.rdd.map(lambda row: row[0:])
    corr_mat = Statistics.corr(factorized_rdd, method=method)
    corr_mat_df = pd.DataFrame(corr_mat,
                    columns=factorized_df.columns, 
                    index=factorized_df.columns)
    
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

airlines_cor_matrix =plot_correlation_matrix(clean_airlines_df, title="Clean Airlines Correlation Matrix")

# COMMAND ----------

# DBTITLE 1,Write into blob storage
clean_airlines_df.write.mode("overwrite").parquet(f"{blob_url}/clean_airlines_1")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stations
# MAGIC 
# MAGIC Looking at the row and column count, the raw stations datafrmae contains 5,004,169 rows and 12 columns.

# COMMAND ----------

print(f"Raw Stations Row Count: {raw_stations_df.count()}")
print(f"Raw Stations Column Count: {len(raw_stations_df.columns)}")

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

# MAGIC %md
# MAGIC ##### Summary Statistic
# MAGIC Next, we looked for null values within the dataframe; fortunately, this dataframe required little pre-processing, given there were no null values contained within the dataframe. We then looked at the dataframe at a high level, using the summarize function to better understand summary statistics of each of the features as is shown below.

# COMMAND ----------

dbutils.data.summarize(raw_stations_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Station Distance to Neighbor
# MAGIC Within this dataframe, logically, we found that the station’s “distance_to_neighbor” would be the most important feature to apply to our model, given that the closer a station is to an airport in proximity, the higher likelihood that the weather forecast/data is accurate. Since inclement weather can often impact flight delays, we decided to run a groupBy on states and measure each station’s average distance to neighbor, expecting more remote states to be farther away from stations. The chart below shows the output. 

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
# MAGIC With almost 5 million rows, we can see that there is a high possibility that there are duplicate stations. Since each row represents an edge between stations and airport, there is a possibility that a station can point to numerous airports and vice versa. To ensure that each row is distinct such that each station only contains a unique airport, we decided to use the closest airport distance to station using a window function. 

# COMMAND ----------

w3 = Window.partitionBy("neighbor_call").orderBy(F.col("distance_to_neighbor").asc())
clean_stations_df = raw_stations_df.withColumn("row", F.row_number().over(w3)) \
                              .filter(F.col("row") == 1).drop("row")

# COMMAND ----------

display(clean_stations_df)

# COMMAND ----------

# MAGIC %md
# MAGIC After the cleaning of the dataframe, we are left with 2,229 rows which greatly reduces the complexity of our joins

# COMMAND ----------

print(f"Clean Stations Row Count: {clean_stations_df.count()}")
print(f"Clean Stations Column Count: {len(clean_stations_df.columns)}")

# COMMAND ----------

dbutils.data.summarize(clean_stations_df)

# COMMAND ----------

# DBTITLE 1,Write into blob storage
clean_stations_df.write.mode("overwrite").parquet(f"{blob_url}/clean_stations_1")

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
# MAGIC To train our model with the data, we require to join all of the tables which are airlines, stations, and weather.

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
# MAGIC DEST_TIMEZONE         | Timezone of Destination Airport                                                                                                 | STRING
# MAGIC UNIX_CRS_DEP_TIME_UTC | Unix timestamp of CRS Departure Time in UTC                                                                                     | INTEGER
# MAGIC EARLY_LIMIT           | Unix timestamp 4 hours prior of CRS Departure Time                                                                              | INTEGER
# MAGIC LATE_LIMIT            | Unix timestamp 1 hour after CRS Departure Time                                                                                  | INTEGER
# MAGIC WEATHER_DEP_DIFF_TIME | Represents a custom field we created, which is the difference between CRS_DEP_TIME and UNIX_WEATHER_TIME from the weather table | INTEGER

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

tz_airlines_df = clean_airlines_df \
    .withColumn("ORIGIN_TIMEZONE", get_timezone(F.col("ORIGIN"))) \
    .withColumn("UNIX_CRS_DEP_TIME_UTC", format_unix_column("CRS_DEP_TIME", "ORIGIN_TIMEZONE")) \
    .withColumn('EARLY_LIMIT', F.col("UNIX_CRS_DEP_TIME_UTC") - (early_threshold*3600)) \
    .withColumn('LATE_LIMIT', F.col("UNIX_CRS_DEP_TIME_UTC") + (late_threshold*3600))

# Format weather date to [yyyy-MM-dd HH:mm:ss]
tz_weather_df = clean_weather_df.withColumn('DATE', F.regexp_replace('DATE', 'T', ' ')) \
    .withColumn("unix_weather_time", F.unix_timestamp(F.col("DATE"),"yyyy-MM-dd HH:mm:ss"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### COVID Impact
# MAGIC In our EDA, we discovered that COVID had a huge impact on flights and the volume of it. To compensate for this anomaly, we decided to add a dummy variable called `COVID` which marked the beginning of travel bans for international flights during the Trump Administration. 
# MAGIC - TODO: What's start (March 2020) and end date

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Imbalance
# MAGIC Due to the volume of data, a sampling technique has to be implemented to reduce bias. We sampled an equal number of airlines with departure delay times over 15 minutes and under 15 minutes. This will minimize data set imbalance and allow an equal number of false positives, false negatives, true positives, and true negatives. Although we will lose a lot of valuable data, it will give us more consistent results and speed up our training process. We can use accuracy as an evaluation metric instead of precision or recall.

# COMMAND ----------

# DBTITLE 1,See number of delayed and non-delayed flights
dep_del15_0_ct = tz_airlines_df.filter(F.col("DEP_DEL15")==0).count()
dep_del15_1_ct = tz_airlines_df.filter(F.col("DEP_DEL15")==1).count()
fraction = dep_del15_1_ct / dep_del15_0_ct

print(f"Number of Non-Delayed Flights: {dep_del15_0_ct}")
print(f"Number of Delayed Flights: {dep_del15_1_ct}")
print(f"Ratio of Delayed to Non-Delayed Flights: {fraction}")

# COMMAND ----------

# DBTITLE 1,Balance Data
new_dep_del15_0 = tz_airlines_df.filter(F.col("DEP_DEL15")==0).sample(False, fraction)
old_dep_del15_1 = tz_airlines_df.filter(F.col("DEP_DEL15")==1)
balanced_airlines_df = new_dep_del15_0.union(old_dep_del15_1)

print(f"New number of Non-Delayed Flights: {new_dep_del15_0.count()}")
print(f"New number of Delayed Flights: {old_dep_del15_1.count()}")
print(f"Total number of flights: {balanced_airlines_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Join Process
# MAGIC - TODO: Edit paragraph
# MAGIC - TODO: Edit join diagram
# MAGIC - TODO: Add time it took to join
# MAGIC 
# MAGIC We took a two-pronged approach to joining the dataframes. We started by joining the ORIGIN feature from the Airlines dataframe with the airport_id column from the Stations dataframe as well as the ORIGIN_STATE_ABR feature from the Airlines dataframe with neighbor_state from the Stations dataframe. Next, we joined the FL_DATE feature from the Airlines and Stations dataframes with the _date column from the weather dataframe. Finally, we joined the station_id column from the Airlines and Stations dataframe with the STATION column from the weather dataframe. We show a diagram of our two-tiered join strategy below.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Drop select columns
# MAGIC 
# MAGIC After joining the tables, we can drop some of the keys that are no longer relevant such as join keys and highly correlated fields.

# COMMAND ----------

# DBTITLE 1,Write to blob storage


# COMMAND ----------

# MAGIC %md
# MAGIC ### Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary Statistic
# MAGIC - TODO: Row and column count

# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlation Matrix

# COMMAND ----------

# MAGIC %md
# MAGIC #### Input vs Output Variable

# COMMAND ----------

# pair_plot_lat_dep_del = sample_final_df.pandas_api().pivot(columns='DEP_DEL15', values='HOURLY_WIND_SPEED')
# pair_plot_lat_dep_del.plot.hist(bins=50, figsize=(6,6), alpha=0.7)  
# pair_plot_lat_dep_del.title("Hourly Wind Speed and Departure Delay Past 15 Minutes")

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
# MAGIC 
# MAGIC | Task | Estimated Improvement | Hours Spent | Start Date | End Date | Team Member
# MAGIC | ---- | --------------------- | ----------- | ---------- | -------- | ----------- |
# MAGIC | Migrate Google Doc to Jupyter Notebook           | 0 | 24 | 11/19 | 11/22 | Steven Sung |
# MAGIC | Clean weather data by removing extra nulls       | 0 | 3  | 11/20 | 11/20 | Steven Sung |
# MAGIC | Added COVID flight restriction binary variable   |   |    |       |       | Steven Sung |

# COMMAND ----------

|
