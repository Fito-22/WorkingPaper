'''This file is helping to map the branching subdisciplines from our OpenAlex API and ArXiv data to more broader subdiciplines which are more friendly with our models'''

import pandas as pd

def make_broader_subdisciplines_physics(data: pd.DataFrame) -> pd.DataFrame:
    '''The function joins all the subdisciplines from the discipline "physics" as a new column in a given DataFrame
    and returns the modified DataFrame'''

    # Make a mapping of physics subtopics from our api data to more generalised physics subtopics
    physics_mapping_dict = {'astrophysics': ['astronomy', 'astrophysics', 'astrobiology'],\
        'condensed matter': ['condensed matter', 'statistical physics', 'neuroscience', 'biological system', 'nanotechnology', 'molecular physics', 'computational physics'],\
        'general relativy and cosmology': ['classical mechanics', 'mechanics', 'optics'],\
        'high enery physics': ['atomic physics', 'particle physics'],\
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

'''For the other disciplines the mapping dict should be written the right way around to begin with'''
