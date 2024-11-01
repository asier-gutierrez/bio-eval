import random
import numpy as np


def generate_texts(starting_text, part_a, relationship, options):
    if starting_text:
        starting_text = starting_text + " "
    else:
        starting_text = ""
    return f'{starting_text}{random.choice(part_a)} {relationship} ', options




def conform_generative_texts(sample_group):
    names = sample_group['a_concepts']
    relationship = sample_group['relation']
    subjects = sample_group['b_concepts']

    # generate samples
    texts_generated, masks = generate_texts(None, names, relationship, subjects)
    return texts_generated, masks


# https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.group_beam_search.example



