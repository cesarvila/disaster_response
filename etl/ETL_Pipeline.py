
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_path, categories_path):
    """
    This function read 2 csvs files 'messages' and 'categories', merge them.
    inputs: 2 paths
    output: dataframe
    """

    # 1. read the files
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)

    # 2 merge datasets
    df = messages.merge(categories, how='left', on='id')
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    #change the category names columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str.strip().str[-1]
        categories[column] = categories[column].astype('int32')

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
        Save processed dataframe into sqlite database
    Inputs:
        df: the dataframe ready
        database_filename: name of the database
    """
    # save data into a sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('CategorizedMessages', engine, index=False,  if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_path, categories_path, database_filename = sys.argv[1:]

        print('Loading and Cleaning data...\n    MESSAGES: {}\n'\
        'CATEGORIES: {}'.format(messages_path, categories_path))
        df = load_data(messages_path, categories_path)

        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        save_data(df, database_filename)

        print('\nCleaned data saved to database!')

    else:
        print('Please provide 3 arguments in order:'\
              '1. file path for messsages.csv'\
              '2. file path for categories.csv'\
              '3. file path for the database to store cleaned output dataframe'\
              'Example: python ETL_pipeline.py '\
              'data/messages.csv data/categories.csv DisasterResponse.db'
              )


if __name__ == '__main__':
    main()
