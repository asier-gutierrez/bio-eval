import argparse

import transformers

import bioeval.snomed.dataset
import bioeval.snomed.evaluator
import bioeval.constants.general


def load_evaluation(
        args
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, padding=True)
    if type == bioeval.constants.general.ModelType.CLM:
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
        pipeline = transformers.TextGenerationPipeline(model, tokenizer, device=args.device)
        eval_cls = bioeval.snomed.evaluator.ModelEvaluatorCLM
        dataset_cls = bioeval.snomed.dataset.DatasetCLM
    elif type == bioeval.constants.general.ModelType.MLM:
        model = transformers.AutoModelForMaskedLM.from_pretrained(args.model)
        pipeline = transformers.FillMaskPipeline(model, tokenizer, device=args.device)
        eval_func = bioeval.snomed.evaluator.ModelEvaluatorMLM
        dataset_cls = bioeval.snomed.dataset.DatasetMLM
    else:
        raise NotImplementedError(f'{type} is not implemented yet.')

    kwargs = {}
    if args.masking:
        kwargs['masking'] = args.masking

    dataset = dataset_cls(snomed_path=args.snomed_path,
                          tokenizer=tokenizer,
                          sampling=args.sampling,
                          **kwargs)

    return tokenizer, model, pipeline, eval_cls, dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--type',
                        type=bioeval.constants.general.ModelType,
                        choices=list(bioeval.constants.general.ModelType),
                        required=True)
    parser.add_argument('--snomed_path', type=str, required=True)
    parser.add_argument('--sampling', type=float, default=0.1)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--masking',
                        type=bioeval.constants.general.MaskingStrategy,
                        choices=list(bioeval.constants.general.MaskingStrategy),
                        required=False)
    args = parser.parse_args()

    tokenizer, model, pipeline, eval_cls, dataset = load_evaluation(args)

    evaluator = eval_cls(model=model,
              tokenizer=tokenizer,
              pipeline=pipeline,
              dataset=dataset,
              device=args.device,
              output_path=args.output_path)
    evaluator.evaluate()
