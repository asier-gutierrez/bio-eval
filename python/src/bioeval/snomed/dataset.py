import re
import linecache
import json
import tqdm
import math
import random
import functools
import collections

import nltk.corpus

import bioeval.constants.general


def load_dataset(
        snomed_path: str,
        sampling: float
):
    if sampling != 1.0:
        n_lines = 0
        with open(snomed_path, 'r', encoding='utf-8') as f:
            for _ in f:
                n_lines += 1

        n_elements = math.ceil(n_lines * sampling)
        indices = list(range(n_elements-1))
        random.shuffle(indices)
        indices = indices[:n_elements]
        chunks = []
        for idx in indices:
            line = linecache.getline(snomed_path, idx+1)
            chunk = json.loads(line)
            chunks.append(chunk)
    else:
        chunks = []
        with open(snomed_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk)
    return chunks


def apply_casing(
        text:str,
        upper_not_lower: bool
):
    words = text.split(" ")
    first_word = words[0]
    if upper_not_lower and (first_word.lower() == first_word) and (first_word[1:].lower() == first_word[1:]):
        words[0] = first_word.title()
    elif not upper_not_lower and (first_word.lower() != first_word) and (first_word[1:].lower() == first_word[1:]):
        words[0] = first_word.lower()
    return " ".join(words)

class DatasetCLM:
    def __init__(
            self,
            snomed_path: str,
            tokenizer: str,
            sampling: float,
            language: str,
            **kwargs
    ):

        self.snomed_path = snomed_path
        self.tokenizer = tokenizer
        self.sampling = sampling
        self.data = load_dataset(
            snomed_path=snomed_path,
            sampling=sampling)

    def prepare_dataset(self):
        for chunk in self.data:
            texts_generated, masks = DatasetCLM.conform_filling(chunk)
            chunk['generations'] = []
            for text_generated, mask in zip(texts_generated, masks):
                generation = {
                    'text_generated': text_generated,
                    'mask': mask,
                }
                chunk['generations'].append(generation)

    @staticmethod
    def conform_filling(chunk):
        names = chunk['b_concepts']
        relationship = chunk['relation']
        subjects = chunk['a_concepts']

        # generate samples
        texts_generated, masks = [], []
        for name in names:
            for subject in subjects:
                name_upper = apply_casing(name, upper_not_lower=True)
                subject_lower = apply_casing(subject, upper_not_lower=False)
                text_generated = f'{name_upper} {relationship} '
                mask = subject_lower
                texts_generated.append(text_generated)
                masks.append(mask)

        assert len(texts_generated) == len(masks)
        return texts_generated, masks


class DatasetMLM:
    def __init__(
            self,
            snomed_path: str,
            tokenizer: str,
            sampling: float,
            language: str,
            **kwargs
    ):

        self.snomed_path = snomed_path
        self.tokenizer = tokenizer
        self.sampling = sampling
        self.language = language
        self.stop_words = nltk.corpus.stopwords.words(self.language)

        self.data = load_dataset(
            snomed_path=snomed_path,
            sampling=sampling)

    def prepare_dataset(self):
        self.mask_func = functools.partial(
            self.mask_wwm,
            tokenizer=self.tokenizer,
            stop_words=self.stop_words)

        for chunk in self.data:
            texts_generated, masks = DatasetMLM.conform_filling(
                chunk,
                mask_func=self.mask_func,
                tokenizer=self.tokenizer)
            chunk['generations'] = []
            for text_generated, mask in zip(texts_generated, masks):
                generation = {
                    'text_generated': text_generated,
                    'mask': mask,
                }
                chunk['generations'].append(generation)

    @staticmethod
    def group_same_text_masks(texts, masks):
        texts_masks = collections.defaultdict(list)
        assert(len(texts) == len(masks))
        for k, v in zip(texts, masks):
            texts_masks[k].append(v)
        return texts_masks.keys(), texts_masks.values()

    @staticmethod
    def conform_filling(chunk, mask_func, tokenizer):
        names = chunk['b_concepts']
        relationship = chunk['relation']
        subjects = chunk['a_concepts']

        names = [apply_casing(text=name, upper_not_lower=True) for name in names]
        subjects = [apply_casing(text=subject, upper_not_lower=False) for subject in subjects]

        # masking left
        names_masked = []
        masks_names = []
        for name in names:
            masked, mask = mask_func(name)
            if masked and mask:
                names_masked.append(masked)
                masks_names.append(mask)
        # names_masked, masks_names = zip(*[mask_func(name) for name in names])
        names_masked, masks_names = DatasetMLM.group_same_text_masks(names_masked, masks_names)

        # masking right
        subjects_masked = []
        masks_subjects = []
        for subject in subjects:
            masked, mask = mask_func(subject)
            if masked and mask:
                subjects_masked.append(masked)
                masks_subjects.append(mask)
        # subjects_masked, masks_subjects = zip(*[mask_func(subject) for subject in subjects])
        subjects_masked, masks_subjects = DatasetMLM.group_same_text_masks(subjects_masked, masks_subjects)

        # generate left and right samples
        texts_generated, masks, _texts_generated, _masks = (), (), (), ()
        if len(names_masked):
            texts_generated, masks = zip(
                *DatasetMLM.generate_texts(None, names_masked, relationship, subjects,
                                           masks_names, how='left', tokenizer=tokenizer))
        if len(subjects_masked):
            _texts_generated, _masks = zip(
                *DatasetMLM.generate_texts(None, names, relationship, subjects_masked,
                                           masks_subjects, how='right', tokenizer=tokenizer))
        texts_generated = texts_generated + _texts_generated
        masks = masks + _masks
        return texts_generated, masks

    @staticmethod
    def generate_texts(starting_text, part_a, relationship, part_b, masks, how, tokenizer):
        if starting_text:
            starting_text = starting_text + " "
        else:
            starting_text = ""

        texts_generated = []
        if how == 'left':
            for element, _mask in zip(part_a, masks):
                for pb in part_b:
                    text_generated = f'{starting_text}{element} {relationship} {pb}.'
                    tokens = tokenizer.encode(text_generated)
                    mask_idx = tokens.index(tokenizer.mask_token_id)
                    start = len(tokenizer.encode(tokenizer.decode(tokens[:mask_idx], skip_special_tokens=True), add_special_tokens=False))
                    end = start + len(tokenizer.encode(_mask[0], add_special_tokens=False))
                    tokens = tokens[:mask_idx] + tokenizer.encode(_mask[0], add_special_tokens=False) + tokens[mask_idx+1:]
                    text_generated = tokenizer.decode(tokens, skip_special_tokens=True)
                    texts_generated.append((text_generated, (start, end, _mask[0])))
        elif how == 'right':
            for element, _mask in zip(part_b, masks):
                for pa in part_a:
                    text_generated = f'{starting_text}{pa} {relationship} {element}.'
                    tokens = tokenizer.encode(text_generated)
                    mask_idx = tokens.index(tokenizer.mask_token_id)
                    start = len(tokenizer.encode(tokenizer.decode(tokens[:mask_idx], skip_special_tokens=True), add_special_tokens=False))
                    end = start + len(tokenizer.encode(_mask[0], add_special_tokens=False))
                    tokens = tokens[:mask_idx] + tokenizer.encode(_mask[0], add_special_tokens=False) + tokens[mask_idx+1:]
                    text_generated = tokenizer.decode(tokens, skip_special_tokens=True)
                    texts_generated.append((text_generated, (start, end, _mask[0])))
        return texts_generated

    @staticmethod
    def mask_wwm(text, tokenizer, stop_words):
        tokenized_text = tokenizer(text)
        encodings = tokenized_text.encodings[0]
        words = list(set(filter(lambda x: x is not None, encodings.words)))
        selection = ''
        attempts = 0
        while (len(selection) <= 2) or (selection in stop_words):
            word_choice = random.choice(words)
            tokens_to_mask = [token for (token, word_id) in zip(encodings.ids, encodings.word_ids) if
                              word_id == word_choice]
            selection = tokenizer.decode(tokens_to_mask)
            attempts = attempts + 1
            if attempts > 10:
                return None, None
        word_start = True
        new_tokens = []
        for token, word_id in zip(encodings.ids, encodings.word_ids):
            if word_id != word_choice:
                new_tokens.append(token)
            elif word_start:
                new_tokens.append(tokenizer.mask_token_id)
                word_start = False
        masked_text = tokenizer.decode(new_tokens[1:-1])
        masked_text = masked_text if masked_text[0] != ' ' else masked_text[1:]
        return masked_text, tokenizer.decode(tokens_to_mask)
