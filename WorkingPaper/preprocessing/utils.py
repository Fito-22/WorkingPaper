'''This file is helping to map the branching subdisciplines from our OpenAlex API and ArXiv data to more broader subdiciplines which are more friendly with our models'''

import pandas as pd

def make_broader_subdisciplines_physics(data: pd.DataFrame) -> pd.DataFrame:
    '''The function joins all the subdisciplines from the discipline "physics" as a new column in a given DataFrame
    and returns the modified DataFrame'''

    # Make a mapping of physics subtopics from our api data to more generalised physics subtopics
    physics_mapping_dict = {'astrophysics': ['astronomy', 'astrophysics', 'astrobiology'],\
        'condensed matter': ['condensed matter', 'statistical physics', 'neuroscience', 'biological system', 'nanotechnology', 'molecular physics', 'computational physics'],\
        'general relativy and cosmology': ['classical mechanics', 'mechanics', 'optics'],\
        'high energy physics': ['atomic physics', 'particle physics'],\
        'mathematical physics': ['geometry', 'mathematical physics', 'applied mathematics', 'theoretical physics', 'pure mathematics'],\
        'nuclear': ['nuclear physics', 'nuclear medicine', 'physical medicine and rehabilitation'],\
        'quantum physics': ['quantum mechanics', 'quantum electrodinamics']}

    def turn_dict(physic_dict):
        new_dict = {}
        for key, values in physic_dict.items():
            for value in values:
                new_dict[value]=key
        return new_dict

    new_mapping_dict = turn_dict(physics_mapping_dict)

    # Mapping the subtopics from the mapping dict to the broader subtopics
    data['broader_subtopic'] = data.where(data['topic'] == 'physics')['subtopic'].map(new_mapping_dict)

    return data

def make_broader_subdisciplines_biology(data: pd.DataFrame) -> pd.DataFrame:
    '''The function joins all the subdisciplines from the discipline "biology" as a new column in a given DataFrame
    and returns the modified DataFrame'''

    # Make a mapping of biology subtopics from our api data to more generalised biology subtopics
    biology_mapping_dict = {'zoology': 'zoology', 'fishery':'zoology', 'animal science':'zoology', 'oceanography': 'zoology', 'cancer research':'genetics and molecular biology', \
        'biochemical engineering': 'genetics and molecular biology', 'genetics': 'genetics and molecular biology', 'molecular biology': 'genetics and molecular biology','biochemistry': 'genetics and molecular biology', 'chromatography': 'genetics and molecular biology', \
        'biotechnology':'genetics and molecular biology', 'biophysics': 'genetics and molecular biology', 'horticulture': 'botany','forestry': 'botany', 'agroforestry': 'botany', \
        'endocrinology': 'physiology', 'immunology': 'physiology', 'neuroscience': 'physiology', 'biological system': 'physiology', \
        'cell biology': 'microbiology', 'virology': 'microbiology', 'microbiology': 'microbiology', \
        'soil science': 'ecology', 'environmental health': 'ecology', 'environmental chemistry': 'ecology', 'ecology': 'ecology', \
        'computational biology': 'computational biology', 'bioinformatics': 'computational biology', 'statistics': 'computational biology',
        'database': 'computational biology', 'data mining': 'computational biology', 'world wide web': 'computational biology', 'data science': 'computational biology', 'information retrieval': 'computational biology', 'algorithm': 'computational biology', 'programming language': 'computational biology', 'artificial intelligence': 'computational biology',\
        'evolutionary biology': 'evolutionary biology', 'botany': 'botany', 'physiology': 'physiology', 'internal medicine': 'physiology', 'anatomy': 'physiology', 'pathology':'physiology'}

    # Mapping the subtopics from the mapping dict to the broader subtopics
    data['broader_subtopic'] = data.where(data['topic'] == 'biology')['subtopic'].map(biology_mapping_dict)

    return data
