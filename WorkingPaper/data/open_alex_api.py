# This file calls the OpenAlex API and extracts information on open access scientific papers

import os
import requests
import pandas as pd
import glob

def get_data_from_openalex():

    '''The function requests the OpenAlex API and retrieves a paper's ID, title, link to an open access PDF,
    publication year, the name of the academic field ('concept') and the name of the academic subfield ('subconcept'),
    which will be our targets for the model training'''

    filter_condition_oa = 'has_oa_accepted_or_published_version:true' # filtering for open access works
    res_per_page = 100
    cursor = '*' # cursor pagination
    url_openalex = f'https://api.openalex.org/works?filter={filter_condition_oa}&per-page={res_per_page}&cursor={cursor}'

    # for chunking or if run on different machines/instances: making the cursor start at a certain page
    #for i in range(1, 100):
    #    cursor = requests.get(url_openalex).json()['meta']['next_cursor']
    #    url_openalex = f'https://api.openalex.org/works?filter={filter_condition_oa}&per-page={res_per_page}&cursor={cursor}'
    #    print(f'cursor num {i}: {cursor}')

    for page in range(1, 1000):
        ids = []
        titles = []
        open_source_article_pdfs = []
        publication_years = []
        concept_names = []
        subconcept_names = []
        for i in range(res_per_page): # check whether the type of the publication is 'journal article',
            #whether it has a open access url pointing to a pdf and if it has been assigned to level 0 concepts and level 1 subconcepts
            try:
                if (requests.get(url_openalex).json()['results'][i]['type'] == 'journal-article') \
                & (requests.get(url_openalex).json()['results'][i]['open_access']['oa_url'].endswith('.pdf')):
                    for concept in requests.get(url_openalex).json()['results'][i]['concepts']:
                        if concept['level'] == 0:
                            topic = concept['display_name']
                        elif concept['level'] == 1:
                            subtopic = concept['display_name']
                    if concept and subtopic: # only append information to lists if concepts and subconcepts have been found
                        ids.append(requests.get(url_openalex).json()['results'][0]['id'])
                        titles.append(requests.get(url_openalex).json()['results'][0]['title'])
                        publication_years.append(requests.get(url_openalex).json()['results'][i]['publication_year'])
                        open_source_article_pdfs.append(requests.get(url_openalex).json()['results'][i]['open_access']['oa_url'])
                        subconcept_names.append(subtopic)
                        concept_names.append(topic)
            except:
                continue

        cursor = requests.get(url_openalex).json()['meta']['next_cursor'] # making cursor move to the next page
        url_openalex = f'https://api.openalex.org/works?filter={filter_condition_oa}&per-page={res_per_page}&cursor={cursor}'
        print(f'cursor num {page}: {cursor}') # for the impatient

        # creating a new csv for every page in the json, otherwiese for bulking could use "if page%10 == 0:""
        link_data = pd.DataFrame({'id': ids, 'pdf_link':open_source_article_pdfs, 'title':titles, 'year': publication_years, 'concepts': concept_names, 'subconcepts': subconcept_names})
        link_data.to_csv(f'../../raw_data/link_data_id/page{page}_link_data.csv', index=False)

    return None

def concatenating_openalex_data_into_single_file():

    ''' Make list of DataFrames from raw_data folder and then concatenate them'''

    joined_files = os.path.join('/home/stefanie/code/WorkingPaper/raw_data/link_data', '*.csv')
    joined_list = glob.glob(joined_files)
    joined_link_data = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)

    return joined_link_data

if __name__ == '__main__':
    get_data_from_openalex()
    concatenating_openalex_data_into_single_file()
