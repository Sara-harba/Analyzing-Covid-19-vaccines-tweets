# Analyzing-Covid-19-vaccines-tweets
This project is done as a part of a university project that aimed to extract tweets from twitter, cleaning them, store them and analyzing them

I used tweepy library to extract tweets, used MySQL and Cassandra to store the cleaned data, then used python and traditional learning alogrithmes to visualize and analyze data

data.csv file represents the columns and tweets that were picked from the data.txt file 

data_ar.csv presents all the arabic tweets that were extracted for labeling

data_en.csv presents the sample of 5000 english tweets that were extracted for labeling

dara_ar_l.csv presents the arabic tweets after labeling.

dara_en_l.csv presents the english tweets after labeling.


Analyzing COVID-19 Vaccines.pynb represents the main code that includes streaming, storing, cleaning, sampling, connection to MySQL & Cassandra,
reading, visualization, and analysis

replace.py file represent some functions and mappings that were used inside the main code for grouping the location column. 

preprocessing.py represent the file for the preprocessing functions that were tried and used within the analysis on the arabic dataset

english.py represent the file for the preprocessing functions that were tried and used within the analysis on the english dataset

features extraction.py represent the feature extraction teqchniques used for the analysis.

model.py represent the models used for the analysis.
