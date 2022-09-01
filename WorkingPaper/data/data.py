"""
This file is make to get data from https://arxiv.org and OpenAlex API
"""
import re
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
import time
import pandas as pd
import os

def get_data_arxiv():
    """
    From arxiv return a pd.DataFrame with text and the subtopic of that pdf.
    """













def get_data_alex(file='/home/adolfo/code/Fito-22/WorkingPaper/raw_data/joined_link_data_0-71.csv', start=0):
    """
    From OpenAlex API return some chunks.csv with text, year, topic and subtopic.
    """
    alex_links_df = pd.read_csv(file)
    alex_links = list(alex_links_df['pdf_link'])

    X=[]
    errors=[]
    chunk_num= 1
    chunk_size= 100

    for index, url in enumerate(alex_links[start:1000]):
        index = index + start
        name = f'/home/adolfo/code/Fito-22/WorkingPaper/raw_data/file_{index}.pdf'
        hdr = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)\
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

        print(f'Saving file_{index}')
        try:
            response=requests.get(url, headers=hdr)
            with open(name,'wb') as f:                                  #Create a copy of the PDF as a file in the local
                f.write(response.content)
        except:
            print('No response given')
            errors.append(index)
            print(f'ERROR {len(errors)} from {index-start} pdfs')       #Notice this skips but not without saving an empty string to X variable
            text=''
            X.append(text)
            continue
        try:
            text = extract_text(name)                                   #Extract the text from the pdf with the pdfminer.six module
        except:
            errors.append(index)
            print(f'ERROR {len(errors)} from {index-start} pdfs')       #Notice this skips but not without saving an empty string to X variable
            text=' '

        os.system(f'rm -r {name}')
        a = text.replace('\n', ' ')             #Fix Text with enter
        b = a.replace('´', '')                  #Fix Text with áéíóú
        X.append(b)

        if index % chunk_size == 0 and index != 0 :                 #Storage in Chunks
            print(f'Saving in alex_chunk_{chunk_num}')
            alex_dataset = pd.DataFrame(X, columns=['text'])
            alex_dataset['subtopic'] = list(alex_links_df[index-chunk_size:index+1]['subconcepts'])
            alex_dataset['topic'] = list(alex_links_df[index-chunk_size:index+1]['concepts'])
            alex_dataset['year'] = list(alex_links_df[index-chunk_size:index+1]['year'])
            alex_dataset.to_csv(f'/home/adolfo/code/Fito-22/WorkingPaper/raw_data/alex_chunk_{chunk_num}.csv')
            X=[X[-1]]
            chunk_num+=1

    alex_dataset = pd.DataFrame(X)
    alex_dataset['subtopic'] = alex_links_df.loc[chunk_size*(chunk_num-1)+1:index, ['subconcepts']]
    alex_dataset['topic'] = alex_links_df.loc[chunk_size*(chunk_num-1)+1:index, ['concepts']]
    alex_dataset['year'] = alex_links_df.loc[chunk_size*(chunk_num-1)+1:index, ['year']]
    alex_dataset.to_csv(f'/home/adolfo/code/Fito-22/WorkingPaper/raw_data/alex_chunk_{chunk_num}.csv')
    print('All done')

    return None

if __name__ == '__main__':
    print(get_data_alex())
