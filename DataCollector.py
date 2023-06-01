import os
import re
import json
from dotenv import load_dotenv, find_dotenv
import tweepy
import pandas as pd
import numpy as np
import logging
import datetime
from sys import platform


NUMBER_OF_MAX_REQUESTS = 1000  # in every 15 minutes there can be a maximum of 450 requests (but wait on rate limit is turned on)
NUMBER_OF_TWEETS_IN_REQUEST = 100  # between 10 - 100 (default is 10)

keywords = '(insomnia OR "sleep deprivation" OR "sleep problem" OR "sleeping problem" OR cantsleep OR "sleep pill" OR "sleeping pill" OR "sleep issue" OR "can’t sleep" OR melatonin OR ambien OR zolpidem OR trazadone OR teamnosleep OR sleepless OR sleepdeprived)'

# Depending on the OS, choose a different path as on Linux the path should be given to my local repository there
if platform == "linux":
    BASE_PATH = "/home/a81678lr/csimage/Third-Year-Project/third-year-project-/"
else:
    BASE_PATH = ""

csv_path = f"{BASE_PATH}data.csv"  # consists of all collected tweets in csv format
json_last_id_path = f"{BASE_PATH}start_id.json"  # path to the id of the last saved tweet to know from which tweet to collect subsequent tweets
json_path = f"{BASE_PATH}data.json"  # consists of all collected tweets in json format

load_dotenv(find_dotenv())


class DataCollector:
    """
    This class introduces methods which are used in collecting data from Twitter using 
    its API. 
    """

    def __init__(self, keywords: str, json_last_id_path: str, csv_path: str, json_file_path: str) -> None:
        """
        Constructor to set required objects and variables like paths for data. 
        """

        self.query = self.contruct_query(keywords)
        self.json_last_id_path = json_last_id_path
        self.csv_path = csv_path
        self.json_path = json_path
        self.json_last_id_file_exists = os.path.exists(self.json_last_id_path)
        self.csv_file_exists = os.path.exists(self.csv_path)
        self.json_file_exists = os.path.exists(self.json_path)
        self.client = tweepy.Client(os.environ.get("BEARER_TOKEN"), wait_on_rate_limit=True)
        self.most_recent_tweet_id = 0

    def contruct_query(self, keywords: str) -> str:
        """
        Define a query to be made to the API.
        Fetch only English tweets and ignore retweets
        """
        query = f"{keywords} lang:en -is:retweet"
        return query
    
    def update_keywords(self, keywords: str) -> None:
        """
        Set new keywords and reconstruct a query. 
        """
        self.query = self.contruct_query(keywords)

    def collect_data_without_saving(self, limit: int, max_results: int) -> pd.DataFrame:
        """
        Collect data with given parameters and return the dataframe containing fetched data without storing it anywhere locally.  
        """
        tweets, includes = self.fetch_data_from_twitter(limit=limit, max_results=max_results)
        success, tweets_df = self.construct_tweets_dataframe(tweets, includes)
        return success, tweets_df

    def collect_data(self, limit: int=400, max_results: int=100) -> pd.DataFrame:
        """
        Collect data with given parameters and return the dataframe containing fetched data.

        Firstly, the last tweet id is obtained to know from which tweet we should continue asking for data.
        Then the data is fetched from the API.
        After that, the data is cleaned and combined to make a dataframe. 
        Finally, the data is stored locally at a chosen location.   
        """
        start_id = self.get_start_id()
        tweets, includes = self.fetch_data_from_twitter(limit=limit, max_results=max_results, start_id=start_id)
        success, tweets_df = self.construct_tweets_dataframe(tweets, includes)
        self.save_data(success, tweets_df)

        return success, tweets_df

    def get_start_id(self) -> int:
        """
        Return the latest collected tweet id. 
        """
        # If a CSV does not exist either then it means we should start collecting
        # data from the oldest tweet as possible which is one week from now due to Twitter restrictions. 
        if not self.json_file_exists:
            logging.warning(f'There is no starting id file')
            return None
        
        with open(self.json_last_id_path, "r") as file:
            data = json.load(file)
        
        start_id = data["start_id"]
        logging.info(f'Tweets will be started to collect from the ID={start_id}.')

        return start_id

    def construct_paginator(self, limit: int=400, max_results: int=100, start_id: int=None) -> tweepy.Paginator:
        """
        This method constructs Tweepy object Paginator which is responsible for making requests
        to the API to fetch data according to the give parameters. 
        It is iterated through with pages to make requests as a max of 100 tweets can be fetched per one request.  
        """
        # These are required to fetch data e.g. author id, text, time, publish time and location
        tweet_fields = ["author_id", "geo", "id", "created_at", "text"]
        place_fields = ["full_name", "geo", "id", "country", "country_code"]
        user_fields = ["name", "username", "id", "location"]
        expansions = ["geo.place_id", "author_id"]
    
        paginator = tweepy.Paginator(
            self.client.search_recent_tweets,
            self.query,
            expansions=expansions,
            place_fields=place_fields,
            tweet_fields=tweet_fields,
            user_fields=user_fields,
            max_results=max_results,
            since_id=start_id,
            limit=limit)
        
        return paginator

    def fetch_data_from_twitter(self, limit: int=400, max_results: int=100, start_id: int=None) -> tuple:
        """
        This method iterated through a Paginator object and collects fetched data which is returned. 
        Includes dataframe consists of extra information like user locations. 
        """
        tweets = list()
        includes = list()

        try:
            paginator = self.construct_paginator(limit=limit, max_results=max_results, start_id=start_id)
            logging.info(f'Initiated a connection with the Twitter API.')

            for response in paginator:
                tweets.extend(response.data)
                includes.extend(response.includes["users"])
                errors = response.errors

                if errors:
                    print("BAD... DO SOMETHING!")
                    logging.warning(f'There are some errors from the API. Error: {errors}.')

        except Exception as e:
            logging.error(f'There was an exception while fetching tweets from the API. Error: {e}.')
            print(f"ERROR! Message: {e}")
        
        return tweets, includes

    def simple_preprocessing(self, text: str) -> str:
        """
        This methods does slight preprocessing of tweets before storing them. 

        Hyperlinks can be regarded as noise, thus they are removed. However, their 
        position within a tweet can carry some semantic information which is necessary for 
        transformer models, so they are replaced by "url".
        Also, mentions of other users are removed due to ethics concerns. 
        What is more, carriages '\r' are removed as they produced some problems when data 
        was saved in a .csv files. 
        """
        # replace urls with a token "url"
        text = re.sub(r'http[s]?://\S+', 'url', text)
        # remove mentions
        text = re.sub(r'@\S+', '', text)
        # remove carriages
        text = text.replace('\r', ' ')

        return text

    def construct_tweets_dataframe(self, tweets: list, includes: pd.DataFrame) -> pd.DataFrame:
        """
        This method constructs a dataframe of collected cleaned tweets from tweets and includes dataframes. 
        """
        if not tweets:
            logging.warning(f'None tweets have been found!')
            print("None tweets were found! Try again later!")
            return False, None

        tweets_df = pd.DataFrame(data=tweets, columns=['author_id', 'id', 'created_at', 'text'])
        includes_df = pd.DataFrame(data=includes)

        if 'location' in includes_df.columns:
            includes_df = includes_df[['id', 'location']]
        else:
            includes_df = includes_df[['id']]
            includes_df["location"] = np.nan

        includes_df.rename(columns={"id": "author_id"}, inplace=True)

        tweets_df = pd.merge(tweets_df, includes_df, on="author_id")
        tweets_df.drop(['author_id'], axis=1, inplace=True)

        # Sort the tweets dataframe as they start from the most recent one, as I need to start from the oldest one,
        # this is why the first tweet is skipped, as it already exists in the file. 
        tweets_df.sort_values(by=["id"], inplace = True)
        
        # numpy int values are not serialisable. 
        self.most_recent_tweet_id = int(tweets_df["id"].iloc[-1])

        tweets_df.drop(['id'], axis=1, inplace=True)

        tweets_df.rename(columns={"created_at": "Publish Date", "text": "Tweet", "location": "Location"}, inplace=True)
        tweets_df['Location'] = tweets_df['Location'].fillna("")
        tweets_df.insert(1, 'Location', tweets_df.pop("Location"))

        tweets_df['Tweet'] = tweets_df['Tweet'].apply(lambda tweet: self.simple_preprocessing(tweet))
        
        number_of_tweets = len(tweets_df)

        # Remove tweets which have the same Tweet.
        tweets_df.drop_duplicates(subset=['Tweet'], inplace=True)
        number_of_duplicates = number_of_tweets - len(tweets_df)

        logging.info(f'{number_of_duplicates} tweets have been removed as duplicates.')

        return True, tweets_df

    def save_data(self, success: bool, tweets_df: pd.DataFrame) -> None:
        """
        This method stores collected and preprocessed tweets in provided locations. 
        """
        # If the file does not yet exist, then it is just created.
        # Otherwise, an existing json file is read, then concatenated with collected ones 
        # and finally saved to the same location.
        if not success:
            return 

        logging.info(f'{len(tweets_df)} tweets have been fetched.')

        if not self.json_file_exists:
            tweets_df.to_json(self.json_path, orient="records", indent=4)
            logging.warning(f'Json file does not exist at {self.json_path}. New one will be created there.')
        else:
            json_tweets_df = pd.read_json(self.json_path, orient="records")
            
            merged_json_tweets_df = pd.concat([json_tweets_df, tweets_df])
            merged_json_tweets_df.to_json(self.json_path, orient="records", indent=4)
            logging.info(f'Tweets have been appended to the Json file at {self.json_path}.')

        if not self.csv_file_exists:
            # create a new file if it does not exist
            logging.warning(f'CSV file does not exist at {self.csv_path}. New one will be created there.')
            tweets_df.to_csv(self.csv_path, index = False, encoding='utf-8')
        
        else:
            # Append collected tweets to a created csv file
            tweets_df.to_csv(self.csv_path, mode='a', index=False, header=False, encoding='utf-8')
            logging.info(f'Tweets have been appended to the CSV file at {self.csv_path}.')

        start_id_dict = {"start_id": self.most_recent_tweet_id}

        # Save the last Tweet ID to a separate file. 
        with open(self.json_last_id_path, "w") as outfile:
            json.dump(start_id_dict, outfile)
            logging.info(f'New start ID={self.most_recent_tweet_id} which is stored at {self.json_last_id_path}.')


def read_csv_dataset(csv_path: str) -> tuple:
    """
    This method reads a collected tweets csv file. 
    """
    file_exists = os.path.exists(csv_path)
    if not file_exists:
        print(f"There is no file at: {csv_path}")
        logging.warning(f'Pre-processing: There is no file at: {csv_path}.')
        return False, None

    # Read the dataset
    parse_dates = ["Publish Date"]
    # tweets_df = pd.read_csv(csv_path, encoding='utf-8', parse_dates=parse_dates, on_bad_lines="skip") 
    tweets_df = pd.read_csv(csv_path, encoding='utf-8', parse_dates=parse_dates) 

    return True, tweets_df


def read_json_dataset(json_path: str) -> tuple:
    """
    This method reads a collected tweets json file. 
    """
    file_exists = os.path.exists(json_path)
    if not file_exists:
        print(f"There is no file at: {json_path}")
        logging.warning(f'Pre-processing: There is no file at: {json_path}.')
        return False, None

    # Read the dataset
    tweets_df = pd.read_json(json_path, orient="records")

    return True, tweets_df


def preprocess_datasets_manually(csv_path: str, json_path: str) -> None:
    """
    Clean tweets in both .csv and .json files as given in parameters. 
    """
    preprocess_csv_dataset_manually(csv_path)
    preprocess_json_dataset_manually(json_path)


def preprocess_csv_dataset_manually(csv_path: str) -> None:
    """
    This method does some preprocessing of the tweets .csv file form the given path. 
    Tweets are stored in the same location at the end.
    """
    success, tweets_df = read_csv_dataset(csv_path)
    if not success:
        return 

    number_of_original_tweets = len(tweets_df)
    print(f"Pre-processing: Current Tweet number in the CSV dataset: {number_of_original_tweets}")
    logging.info(f'Pre-processing: Current Tweet number in the CSV dataset: {number_of_original_tweets}.')

    # Remove duplicate tweets
    tweets_df.drop_duplicates(subset=['Tweet'], inplace=True)
    
    # Save the results
    tweets_df.to_csv(csv_path, index = False, encoding='utf-8')
    number_of_preprocessed_tweets = len(tweets_df)
    print(f"Pre-processing: Tweets number after some CSV pre-processing: {number_of_preprocessed_tweets}")
    logging.info(f'Pre-processing: Tweets number after some CSV pre-processing {number_of_preprocessed_tweets}.')

def preprocess_json_dataset_manually(json_path: str) -> None:
    """
    This method does some preprocessing of the tweets .json file form the given path. 
    Tweets are stored in the same location at the end.
    """
    success, tweets_df = read_json_dataset(json_path)
    if not success:
        return 

    number_of_original_tweets = len(tweets_df)
    print(f"Pre-processing: Current Tweet number in the Json dataset: {number_of_original_tweets}")
    logging.info(f'Pre-processing: Current Tweet number in the Json dataset: {number_of_original_tweets}.')

    # Remove duplicate tweets
    tweets_df.drop_duplicates(subset=['Tweet'], inplace=True)
    
    # Save the results
    tweets_df.to_json(json_path, orient="records", indent=4)

    number_of_preprocessed_tweets = len(tweets_df)
    print(f"Pre-processing: Tweets number after some Json pre-processing: {number_of_preprocessed_tweets}")
    logging.info(f'Pre-processing: Tweets number after some Json pre-processing {number_of_preprocessed_tweets}.')


def convert_to_json(csv_path: str) -> None:
    """
    Convert a .csv file to a .json one.
    This was needed as firstly I was storing the data in .csv format, but at the end
    I decided to switch to .json. 
    """
    success, tweets_df = read_csv_dataset(csv_path)
    if not success:
        return 

    tweets_df.to_json(json_path, orient="records", indent=4)
    logging.info(f'Convert to Json: CSV file at {csv_path} has been converted to a json file at {json_path}. {len(tweets_df)} number of tweets are stored.')


def setup_logging() -> None:
    """
    This method sets up logging directory, sets basic configuration on the logging process
    and creates the first file for that.
    """
    
    logs_directory_name = f"{BASE_PATH}DataCollectionLogs"
    
    # Windows does not allow ':' in the names
    current_time_string = str(datetime.datetime.now()).replace(":", "-")

    # Set up the format of logging
    logging.basicConfig(level=logging.INFO, filename=f'{logs_directory_name}/{current_time_string}.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a directory if it does not exist yet
    if not os.path.exists(logs_directory_name):
        os.mkdir(logs_directory_name)
    
    logging.info(f'Logging starts. The base path is {BASE_PATH}. The folder of logs is {logs_directory_name}')


if __name__ == "__main__":
    setup_logging()

    data_collector = DataCollector(keywords, json_last_id_path, csv_path, json_path)
    success, tweets_df = data_collector.collect_data(limit=NUMBER_OF_MAX_REQUESTS, max_results=NUMBER_OF_TWEETS_IN_REQUEST)

    if success:
        print(f"Number of tweets fetched: {len(tweets_df)}")

    preprocess_datasets_manually(csv_path, json_path)
