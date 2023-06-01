# Mining and Analysing Twitter Data on Insomnia 

This is my third-year Computer Science project about Analysing Twitter Data on Insomnia at the University of Manchester. 

----------------------------------------
# Code
There are five main code files:
1. DataCollector.py
2. TweetCollector.ipynb
3. TweetTransformersTraining.ipynb
4. TweetTopicModelling.ipynb
5. TweetExperiments.ipynb

**DataCollector.py** and **TweetCollector.ipynb** are very similar to each other because they basically do the same function - collect data from Twitter using its API. However, **DataCollector.py** was created to automate the data collection on the University Linux Data Science server.
**TweetTransformersTraining.ipynb** is used to fine-tune a chosen transformer model using k-Cross-Validation or Bagging Ensembles. Also, it does tweet sentiment prediction (inference) which results can be stored in the given location. 
**TweetTopicModelling.ipynb** trains a topic modelling model and visualises results on the tweets dataset using the BERTopic model. 
**TweetExperiments.ipynb** performs various experiments with sentiment and topic labelled tweets. 

**Collected data cannot be published due to approved ethics application requirements**.

----------------------------------------
# Running DataCollector.py
To be able to run this, firstly Twitter API Bearer key should be added to the **.env** file. Then a couple of libraries should be downloaded using `pip install -r requirements.txt`. Also, **Numpy** and **Pandas** are used there as well. What is more, the path variable `BASE_PATH` can be edited to accommodate your specific needs regarding data storage. 

The automation of data collection was achieved with Crontab jobs. It is needed as Twitter only allows to access tweets no older than one week with its API. This command was utilised to collect tweets each day at 9 am: `0 09 * * * /usr/bin/python3 /DataCollector.py`.

Also, logging was set up for this script to observe the statistics about the fetched and stored tweets each day. 

Tweets are stored in both `.json` and `.csv` (it may be used as backup) formats. The data is stored in `.json` in this way:

```json
[
    {
        "Publish Date":1679561886000,
        "Location":"SOME LOCATION",
        "Tweet":"I can't sleep"
    },
    {
        "Publish Date":1679561886000,
        "Location":"SOME KINGDOM",
        "Tweet":"I can't sleep :("
    },
    ...
]
```

When tweets are annotated, `"Sentiment"` property is added as well where integers 0, 1 and 2 corresponds to negative, neutral and positive sentiment.

----------------------------------------
# Running Notebooks

More details on how to run each notebook can be found at the beginning of each notebook with detailed instructions what is needed to run it and what it does. Notebooks are prepared to be run on the Google Colab platform, however, it is not be difficult to adapt them to run on other platforms like Kaggle.  


© 2023 Lukas Rimkus 