import copy

import numpy as np

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
            output_path: str,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.dataset = dataset
        self.device = device
        self.output_path = output_path

    def evaluate(self):
        if self.dataset.mask == bioeval.constants.general.MaskingStrategy.SWM:
            eval_func = self.evaluate_mlm_swm
        else:
            eval_func = self.evaluate_mlm_wwm
        for data in self.dataset.data:
            d = copy.deepcopy(data)

    def evaluate_mlm_swm(self, pipeline, texts, refs):
        accuracies = []
        for text, ref in zip(texts, refs):
            preds = pipeline(text, targets=ref)
            accuracies.extend([pred['score'] for pred in preds])
        return accuracies

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
            output_path: str,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.dataset = dataset
        self.device = device
        self.output_path = output_path

    def evaluate(self):
        for data in self.dataset.data:
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
            num_beams=5,
            max_new_tokens=len(expected_preds),
            return_dict_in_generate=True,
            output_scores=True
        )
        transition_scores = self.model.compute_transition_scores(
            sequences=outputs.sequences,
            scores=outputs.scores,
            normalize_logits=True
        )
        score = np.exp(transition_scores.to('cpu').numpy()).tolist()
        return score
