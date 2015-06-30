# LHCDataAnalysis

##Problem Statement

A classification problem which aims at finding if a dataset will become popular or not and when it will become popular. It includes time series analysis of the data, to calculate the approximate time period when the dataset will be popular, as well as classification, for calculating binary values of popular (1 / TRUE) or unpopular (0 / FALSE) of the 2013, 2014 and the currently producing 2015 data. 

This data is produced by the LHC weekly. Analysing this data leads to useful information about the physical processes. Reproducibility is necessary so that any process can be simulated just like the original in a software at different times. Besides this, some process may be researched more by users and hence it needs to be made easily accessible to all the users. Accessibility is possible using replicas of data at some specified places. The user can then obtain the data from the nearest replica. Creating numerous replicas of every dataset is not feasible because the all datasets are vast. Maintaining and mirroring such vast datasets is an expensive job hence, the need of predicting which dataset might become popular is necessary.  

The process of predicting popularity of datasets, hence which datasets replicas should be created, is a machine learning problem. Hence, the aim of the problem is to use machine learning to predict which dataset will become popular and when.  

##Goals

Define the popularity index. 
The target goal is to find the popular datasets, but what is a popular dataset? Which parameters define popularity of the dataset?
Apply machine learning algorithms and analytics for prediction. 
Find the hardware configuration that gives the relatively best run of a machine learning algorithm. 
Time Series analysis of the existing processes to see if any process might become popular in any given week of any year. 
Generalize machine learning algorithms so that they can streamline CMS data without much data formatting. May lead to creation of an API. 
Evaluate Apache Spark as an alternate framework for the complete analysis procedure. 

##Dataset

A dataset describes a process completely. A process is any interaction taking place in the LHC. An example would be proton-proton collision in the LHC, taking place at a single vertex. A single process may be composed of many collisions taking place at the same vertex. The weekly collection of data is uniquely represented by the name of the dataset. It describes which weeks data it contains, 20140101 - 20140107 will describe the first week of year 2014. 

A datasets format is defined by three distinct parts, 
a
b
c
 
where:

a is a process type, examples include Higgs Process, TopQuark, ttbq. 
b is the software used and its version/release number. This variable is important for reproducing identical results in the future.
c, defines the tier  {RAW, RECO, AOP, DIGI, DIGI-RECO, SIM}

These are combined as /a/b/c.

Dataframe is the input file in comma separated values format. This is the file on which the machine learning algorithms are run upon. It contains many datasets, and each dataset is uniquely represented by one row in the dataframe. 

Currently used popularity metrics: 

naccess - number of accesses to a dataset
totcpu - number of CPU hours utilized to access a dataset
nusers - number of users times days when the dataset was accessed

These are reported by the PopularityDB

