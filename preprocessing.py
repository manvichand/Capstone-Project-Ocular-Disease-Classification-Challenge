#preprocessing
#Library used
import pandas as pd
import os
import numpy as np

class_labels= ['N','D','C','A','F','M','O']

#These labels represent diagnostic categories:
keyword_label_mapping  = {
    'normal':'N',
    'retinopathy':'D',
    'glaucoma':'G',
    'cataract':'C',
    'macular degeneration':'A',
    'hypertensive':'H',
    'myopia':'M',
    'lens dust':'O', 'optic disk photographically invisible':'O', 'low image quality':'O', 'image offset':'O'
}
non_decisive_labels = ["lens dust", "optic disk photographically invisible", "low image quality", "image offset"]
# A dictionary mapping medical keywords to class labels.
"""
Processes a string of diagnostic keywords and returns a single standardized label based on predefined mappings.

Parameters:
- 'diagnostic_keywords' (str): A comma-separated string of diagnostic keywords.

Logic:
1. Splits the keywords into a list.
2. Checks for specific conditions:
    - If 'normal' is found, marks it as potentially normal.
    - If any other decisive keyword is found, assigns the corresponding label.
3. If no decisive labels are found but non-decisive labels are present, returns `'O'` (Others).
4. Defaults to returning `'N'` if only `normal` is found.

Returns:
- A single label from `class_labels`.
"""
def get_individual_labels(diagnostic_keywords):
    keywords = [ keyword  for keyword in diagnostic_keywords.split(',')]
    contains_normal = False
    for k in keywords:
        for label in keyword_label_mapping.keys():
            if label in k:
                if label == 'normal':
                    contains_normal = True # if found a 'normal' keyword, check if there are other keywords but keep in mind that a normal keyword was found
                else:
                    return keyword_label_mapping[label] # found a proper keyword label, use the first occurence

    # did not find a proper keyword label, see if there are labels other than non-decisive labels, if so, categorize them as 'others'
    decisive_label = False
    for k in keywords:
        if k not in non_decisive_labels and (('normal' not in k) or ('abnormal' in k)):
            decisive_label = True
    if decisive_label:
        # contains decisive label other than the normal and abnormal categories
        return 'O'
    if contains_normal:
        return 'N'
    
    return keywords[0] # u

# write test cases
# if both left and right are normal, then the final diagnosis is also normal
def test_normal(row):
    l,r = row['Left-label'], row['Right-label']
    if l == 'N' and r == 'N' and row['N'] != 1:
        return False
    else:
        return True

def test_others(row):
    l,r = row['Left-label'], row['Right-label']
    if row['O'] == 1:
        if l == 'O' or r == 'O':
            return True
        else:
            return False
    return True

