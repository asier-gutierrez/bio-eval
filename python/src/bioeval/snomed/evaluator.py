import json
import copy

import numpy as np
import tqdm

import bioeval.snomed.dataset
import bioeval.constants.general


class ModelEvaluatorMLM:
    def __init__(
            self,
            model,
            tokenizer,
            pipeline,
            dataset: bioeval.snomed.dataset.DatasetMLM,
            device: str,
            output_file: str,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.dataset = dataset
        self.device = device
        self.output_file = output_file

    def evaluate(self):
        if self.dataset.mask == bioeval.constants.general.MaskingStrategy.SWM:
            eval_func = self.evaluate_mlm_swm
        else:
            eval_func = self.evaluate_mlm_wwm
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for data in tqdm.tqdm(self.dataset.data):
                d = copy.deepcopy(data)
                texts_generated = []
                masks = []
                for generation in d['generations']:
                    texts_generated.append(generation['text_generated'])
                    masks.append(generation['mask'])
                accuracies = eval_func(self.pipeline, texts_generated, masks)
                for generation, accuracy in zip(d['generations'], accuracies):
                    generation['accuracy'] = accuracy
                f.write(f"{json.dumps(d)}\n")

    @staticmethod
    def evaluate_mlm_swm(pipeline, texts, refs):
        accuracies = []
        for text, ref in zip(texts, refs):
            preds = pipeline(text, targets=ref)
            accuracies.extend([pred['score'] for pred in preds])
        return accuracies

    @staticmethod
    def evaluate_mlm_wwm(pipeline, texts, refs):
        accuracies = []
        for text, ref in zip(texts, refs):
            for _ref in ref:
                preds = pipeline(text, targets=_ref, top_k=len(_ref))
                for i in range(len(_ref)):
                    if type(preds[i]) is dict:
                        accuracies.extend([preds[i]['score']])
                    else:
                        accuracies.extend([preds[i][i]['score']])
        return accuracies


class ModelEvaluatorCLM:
    def __init__(
            self,
            model,
            tokenizer,
            pipeline,
            dataset: bioeval.snomed.dataset.DatasetCLM,
            device: str,
            output_file: str,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.dataset = dataset
        self.device = device
        self.output_file = output_file

    def evaluate(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for data in tqdm.tqdm(self.dataset.data):
                d = copy.deepcopy(data)
                d['scores'] = []
                for generation in d['generations']:
                    text_generated = generation['text_generated']
                    mask = generation['mask']
                    score = self.evaluate_inner(
                        text_generated=text_generated,
                        mask=mask
                    )
                    d['scores'].append(score)
                f.write(f"{json.dumps(d)}\n")

    def evaluate_inner(self, text_generated, mask):
        inputs = self.tokenizer(
            [text_generated],
            return_tensors="pt"
        )
        inputs.to(self.device)
        expected_preds = self.tokenizer(mask).input_ids
        outputs = self.model.generate(
            **inputs,
            force_words_ids=[expected_preds],
            num_beams=2,
            max_new_tokens=len(expected_preds),
            return_dict_in_generate=True,
            output_scores=True
        )
        transition_scores = self.model.compute_transition_scores(
            sequences=outputs.sequences,
            scores=outputs.scores,
            normalize_logits=True
        )
        score = np.exp(transition_scores[0].to('cpu').numpy()).tolist()
        return score[1:] # Skip the first score (from previous text to start_of_sequence)
