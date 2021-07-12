import sys
import logging

import depccg.parsing
from depccg.types import Token
from depccg.cat import Category
from depccg.printer import print_
from depccg.instance_models import load_model
from depccg.argparse import parse_args
from depccg.lang import set_global_language_to
from depccg.annotator import (
    english_annotator, japanese_annotator, annotate_XX
)
from depccg.allennlp.utils import read_params

# disable lengthy allennlp logs
logging.getLogger('filelock').setLevel(logging.ERROR)
logging.getLogger('allennlp').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def get_annotator(args):
    if args.lang == 'en':
        if (
            args.format in ['ccg2lambda', 'jigg_xml_ccg2lambda']
            and args.annotator is None
        ):
            raise RuntimeError(
                ('Specify --annotator argument in '
                 f'using "{args.format}" output format')
            )
        return english_annotator.get(args.annotator, annotate_XX)

    elif args.lang == 'ja':
        if (
            args.format in ['ccg2lambda', 'jigg_xml_ccg2lambda']
            and not args.tokenize
        ):
            raise RuntimeError(
                ('Cannot specify --pre-tokenized '
                 f'argument using "{args.format}" output format')
            )
        if args.tokenize:
            return japanese_annotator[args.annotator]
        return annotate_XX

    raise RuntimeError(f'unsupported language: {args.lang}')


def main(args):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.CRITICAL if args.silent else logging.INFO
    )

    set_global_language_to(args.lang)
    annotator_fun = get_annotator(args)
    supertagger, config = load_model(args.model, args.gpu)

    (
        apply_binary_rules,
        apply_unary_rules,
        category_dict,
        _
    ) = read_params(config.config, args)

    root_categories = [
        Category.parse(category)
        for category in args.root_cats.split('|')
    ]

    semantic_templates = (
        args.semantic_templates or config.semantic_templates
    )

    kwargs = dict(
        unary_penalty=args.unary_penalty,
        nbest=args.nbest,
        pruning_size=args.pruning_size,
        beta=args.beta,
        use_beta=not args.disable_beta,
        max_length=args.max_length,
        max_step=args.max_step,
        processes=args.num_processes,
    )

    if args.input is not None:
        input_type = open(args.input)
    elif not sys.stdin.isatty():
        input_type = sys.stdin
    else:
        # reading from keyboard
        input_type = None
        sys.stdout.flush()
        sys.stderr.flush()
        logging.getLogger().setLevel(logging.CRITICAL)

    categories = None
    while True:
        fin = [
            line for line in map(str.strip, input_type or [input()])
            if len(line) > 0
        ]
        if len(fin) == 0:
            break

        if args.input_format == 'POSandNERtagged':
            doc = [
                [
                    Token.of_piped(token)
                    for token in sent.split(' ')
                ] for sent in fin
            ]

        else:
            doc = annotator_fun(
                [
                    [word for word in sentence.split(' ')]
                    for sentence in fin
                    if len(sentence) > 0
                ],
                tokenize=args.tokenize,
            )

        logger.info("supertagging")
        score_result, categories_ = supertagger.predict_doc(
            [[token.word for token in sentence] for sentence in doc]
        )
        if categories is None:
            categories = [
                Category.parse(category) for category in categories_
            ]

        if category_dict is not None:
            doc, score_result = depccg.parsing.apply_category_filters(
                doc,
                score_result,
                categories,
                category_dict,
            )

        logger.info("parsing")
        results = depccg.parsing.run(
            doc,
            score_result,
            categories,
            root_categories,
            apply_binary_rules,
            apply_unary_rules,
            **kwargs,
        )

        print_(
            results,
            format=args.format,
            semantic_templates=semantic_templates
        )

        if input_type is None:
            sys.stdout.flush()
        else:
            break


if __name__ == '__main__':
    parse_args(main)
