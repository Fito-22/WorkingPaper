"""
This file is make to get data from https://arxiv.org and OpenAlex API
"""
import re
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
import pandas as pd
import os

def get_data_arxiv():
    """
    From arxiv return a pd.DataFrame with text and the subtopic of that pdf.
    """
    response = requests.get('https://arxiv.org/')
    soup = BeautifulSoup(response.content, 'html.parser')
    #Math Subtopics
    a =  soup.find_all('a', id= re.compile('math.'))
    b=[]
    c=[]
    for i in a:
        b.append(i['id'])
        c.append(i.text)
    math_subtopics_names = [c[0]] + c[4:]
    math_subtopics_index = [b[0]] + b[4:]
    Math_subtopics ={}
    for i in range(len(math_subtopics_names)):
        Math_subtopics[math_subtopics_names[i]]=math_subtopics_index[i]

    #Physics Subtopics
    a =  soup.find_all('strong')
    a_2 = soup.find_all('a', id=re.compile('main'))
    b=[]
    c=[]
    for i in a:
        c.append('recent-' + i.text)
    for i in a_2:
        b.append(i.text)
    ph_subtopics_index = c[0:11] + [c[12]]
    ph_subtopics_names = b[:11] + [b[12]]

    Physics_subtopics ={}
    for i in range(len(ph_subtopics_index)):
        Physics_subtopics[ph_subtopics_names[i]]=ph_subtopics_index[i]

    #Computer Science Subtopics
    a_cs = soup.find_all('a', id=re.compile('cs.[A-Z]'))
    b=[]
    c=[]
    for i in a_cs:
        c.append(i['id'])
        b.append(i.text)
    cs_subtopics_index = c
    cs_subtopics_names = b

    ComputerScience_subtopics ={}
    for i in range(len(cs_subtopics_index)):
        ComputerScience_subtopics[cs_subtopics_names[i]]=cs_subtopics_index[i]

    #Biology
    a =  soup.find_all('a', id= re.compile('q-bio.'))
    b=[]
    c=[]
    for i in a:
        b.append(i['id'])
        c.append(i.text)
    biology_subtopics_names = c
    biology_subtopics_index = b
    Biology_subtopics ={}
    for i in range(len(biology_subtopics_names)):
        Biology_subtopics[biology_subtopics_names[i]]=biology_subtopics_index[i]

    #Quantitative Finance
    a =  soup.find_all('a', id= re.compile('q-fin.'))
    b=[]
    c=[]
    for i in a:
        b.append(i['id'])
        c.append(i.text)
    qfinance_subtopics_names = c
    qfinance_subtopics_index = b
    QFinance_subtopics ={}
    for i in range(len(qfinance_subtopics_names)):
        QFinance_subtopics[qfinance_subtopics_names[i]]=qfinance_subtopics_index[i]

    # Statistics
    a =  soup.find_all('a', id= re.compile('stat.'))
    b=[]
    c=[]
    for i in a:
        b.append(i['id'])
        c.append(i.text)
    statistics_subtopics_names = c
    statistics_subtopics_index = b
    Statistics_subtopics ={}
    for i in range(len(statistics_subtopics_names)):
        Statistics_subtopics[statistics_subtopics_names[i]]=statistics_subtopics_index[i]

    # Electrical Engineering and Systems Science
    a =  soup.find_all('a', id= re.compile('eess.'))
    b=[]
    c=[]
    for i in a:
        b.append(i['id'])
        c.append(i.text)
    eess_subtopics_names = c
    eess_subtopics_index = b
    EESS_subtopics ={}
    for i in range(len(eess_subtopics_names)):
        EESS_subtopics[eess_subtopics_names[i]]=eess_subtopics_index[i]

    # Economics
    a =  soup.find_all('a', id= re.compile('econ.'))
    b=[]
    c=[]
    for i in a:
        b.append(i['id'])
        c.append(i.text)
    economics_subtopics_names = c
    economics_subtopics_index = b
    Economics_subtopics ={}
    for i in range(len(economics_subtopics_names)):
        Economics_subtopics[economics_subtopics_names[i]]=economics_subtopics_index[i]

    #PDF LIST
    main_topics =[Physics_subtopics, Math_subtopics, ComputerScience_subtopics, Biology_subtopics, Statistics_subtopics, EESS_subtopics, Economics_subtopics ]
    PDF_list=[]
    counts={}
    a=[]
    for topic in main_topics:
        for sub_topic_name, sub_topic_index in topic.items():
            sub_topic = soup.find_all('a', id=f'{sub_topic_index}')[0]['href']
            # print(f'Taking PDFs from: {sub_topic_name}')
            st_response = requests.get(f'https://arxiv.org{sub_topic}?show=100')
            st_soup = BeautifulSoup(st_response.content, 'html.parser')
            articles = st_soup.find_all('span', class_='list-identifier') #All the articles here
            for i in articles:
                try:
                    a.append([i.find("a", title="Download PDF")["href"], sub_topic_name])
                    PDF_list.append(f'https://arxiv.org{i.find("a", title="Download PDF")["href"]}.pdf')
                except: continue

    X=[]
    errors=[]
    for index, url in enumerate(PDF_list):
        print(f'Downloading: file_{index}.pdf')
        name = f"file_{index}.pdf"
        response=requests.get(url)
        with open(name,'wb') as f:                                  #Create a copy of the PDF as a file in the local
            f.write(response.content)
        print('Saving it')
        try:
            text = extract_text(name)
        except:
            text=' '
            print('error')
            errors.append(index)
            os.system(f'rm -r {name}')
            continue

        a = text.replace('\n', ' ')             #Text with enter
        b = a.replace('´', '')                  #Text with áéíóú
        X.append(b)
        print(f'Removing: file_{index}.pdf')
        os.system(f'rm -r {name}')











def get_data_alex(file='/home/adolfo/code/Fito-22/WorkingPaper/raw_data/joined_link_data_0-71.csv', start=0):
    """
    From OpenAlex API return some chunks.csv with text, year, topic and subtopic.
    """
    alex_links_df = pd.read_csv(file)
    alex_links = list(alex_links_df['pdf_link'])

    X=[]
    errors=[]
    chunk_num= 1+start/100
    chunk_size= 100

    for index, url in enumerate(alex_links):
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

        if index % chunk_size == 0 and index-start != 0 :                 #Storage in Chunks
            print(f'Saving in alex_chunk_{chunk_num}')
            alex_dataset = pd.DataFrame(X, columns=['text'])
            alex_dataset['subtopic'] = list(alex_links_df[index-chunk_size:index+1]['subconcepts'])
            alex_dataset['topic'] = list(alex_links_df[index-chunk_size:index+1]['concepts'])
            alex_dataset['year'] = list(alex_links_df[index-chunk_size:index+1]['year'])
            alex_dataset['title'] = list(alex_links_df[index-chunk_size:index+1]['title'])
            alex_dataset['id'] = list(alex_links_df[index-chunk_size:index+1]['id'])
            alex_dataset.to_csv(f'/home/adolfo/code/Fito-22/WorkingPaper/raw_data/alex_chunk_{chunk_num}.csv')
            X=[X[-1]]
            chunk_num+=1
    return None

if __name__ == '__main__':
    get_data_alex()
