import random
import itertools
from collections import defaultdict
import evaluate

accuracy_metric = evaluate.load('accuracy')













# def conform_masked_texts(tokenizer, sample_group, mask=mask_swm):
#     names = sample_group['name']
#     starting_text = sample_group['starting_text']
#     relationship = sample_group['relationship']
#     subjects = sample_group['subjects']
#     subjects_flat = [subject['FSN'] for subject in subjects]
#     subjects_flat += [subject['children'] for subject in subjects if 'children' in subject]
#
#     # masking left
#     names_masked, masks_names = zip(*[mask(tokenizer, name) for name in names])
#     names_masked, masks_names = group(names_masked, masks_names)
#
#     # masking right
#     subjects_masked, masks_subjects = zip(*[mask(tokenizer, subject) for subject in subjects_flat])
#     subjects_masked, masks_subjects = group(subjects_masked, masks_subjects)
#
#     # generate left and right samples
#     texts_generated, masks = zip(
#         *generate_texts(starting_text, names_masked, relationship, subjects_flat, masks_names, how='left'))
#     _texts_generated, _masks = zip(
#         *generate_texts(starting_text, names, relationship, subjects_masked, masks_subjects, how='right'))
#     texts_generated = texts_generated + _texts_generated
#     masks = masks + _masks
#     return texts_generated, masks



