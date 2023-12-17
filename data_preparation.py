import sqlite3
import pandas as pd
import submission_parser


def load_data():
    connection = sqlite3.connect('../bjj.db')
    data_frame = pd.read_sql("SELECT winner, submission, category, fight, competition, id FROM matches WHERE submission NOT IN ('N/A') AND winner NOT IN ('') GROUP BY winner, competition, category, id", connection)
    return data_frame

def unique_submissions():
    data_frame = load_data()
    unique_submissions = dict(enumerate(data_frame.submission.unique()))
    return unique_submissions

def parse_data(data_frame):
    unique_submissions = dict(enumerate(data_frame.submission.unique()))
    submissions_by_competition = data_frame[data_frame.duplicated(['winner', 'category', 'competition'], keep=False)].groupby(['winner', 'competition', 'category'])['submission'].apply(list).reset_index()
    training_data = []
    for index, row in submissions_by_competition.iterrows():
        training_data.append(submission_parser.Parser(row['submission'],unique_submissions).parse())
    
    return training_data

def main():
    data_frame = load_data()
    data = parse_data(data_frame)
    return data