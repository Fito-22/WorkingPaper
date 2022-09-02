import pandas as pd
import numpy as np
import os

'''
USE the last function
'''

math_lst_topics = ["Algebraic Geometry",
                        "Algebraic Topology",
                        "Analysis of PDEs",
                        "Category Theory",
                        "Classical Analysis and ODEs",
                        "Combinatorics",
                        "Commutative Algebra",
                        "Complex Variables",
                        "Differential Geometry",
                        "Dynamical Systems",
                        "Functional Analysis",
                        "General Mathematics",
                        "General Topology",
                        "Geometric Topology",
                        "Group Theory",
                        "History and Overview",
                        "Information Theory",
                        "K-Theory and Homology",
                        "Logic; Mathematical Physics",
                        "Metric Geometry",
                        "Number Theory",
                        "Numerical Analysis",
                        "Operator Algebras",
                        "Optimization and Control",
                        "Probability; Quantum Algebra",
                        "Representation Theory",
                        "Rings and Algebras",
                        "Spectral Theory",
                        "Statistics Theory",
                        "Symplectic Geometry"]

def load_csv(path = '../../raw_data/small_dataset.csv'):

    '''creating an absolute path to be able
    to run the code on every machine '''

    abs_path = os.path.abspath(path)

    df = pd.read_csv(abs_path)
    #df = pd.read_csv(path)

    return df

def preparing_dataframe(df):
    '''
    Changing the names of the columns of the df
    '''

    df.rename(
        columns={"Unnamed: 0":"Index",
            "text":"paper_text"}
              ,inplace=True)
    df.set_index("Index",inplace=True)


    return df

def add_topic(df,lst_math_topics):
    '''
    Add a columns topics that sort
    the papers by their sub topics
    '''
    df['topic'] = df['subtopic'].apply(lambda x: 'mathematic' if x in lst_math_topics else 'physics')
    return df
    
def balancing_df(df):
    top_topics = df['topic'].value_counts().index.array[0:5]
    return df#[(df['topic'] == top_topics[0]) | (df['topic'] == top_topics[1])]

def load(path):
    '''
    Final load function that return a usable df
    '''
    ## this return was for the small dataset
    #return add_topic(preparing_dataframe(load_csv(path)),math_lst_topics)

    ## for the final data set we don't need to add topic column
    return balancing_df(preparing_dataframe(load_csv(path)))
