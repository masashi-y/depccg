import argparse
from depccg.instance_models import download, AVAILABLE_MODEL_VARIANTS
from depccg.annotator import (
    english_annotator, japanese_annotator
)


def add_common_parser_arguments(parser, main_fun):
    parser.add_argument(
        '-c',
        '--config',
        help='json config file specifying the set of unary rules used, etc.')
    parser.add_argument(
        '-m',
        '--model',
        help='path to model directory')
    parser.add_argument(
        '-p',
        '--num-processes',
        default=4,
        type=int,
        help='number of processes used for parsing')
    parser.add_argument(
        '-i',
        '--input',
        default=None,
        help='a file with tokenized sentences in each line')
    # parser.add_argument(
    #     '-w',
    #     '--weights',
    #     default=None,
    #     help='a file that contains weights (p_tag, p_dep)')
    parser.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help='specify gpu id')
    parser.add_argument(
        '--batchsize',
        type=int,
        default=32,
        help='batchsize in supertagger')
    parser.add_argument(
        '--nbest',
        type=int,
        default=1,
        help='output N best parses')
    parser.add_argument(
        '-I',
        '--input-format',
        default='raw',
        choices=['raw', 'POSandNERtagged'],
        # choices=['raw', 'POSandNERtagged', 'json', 'partial'],
        help='input format')
    parser.add_argument(
        '--unary-penalty',
        default=0.1,
        type=float,
        help='penalty to use a unary rule')
    parser.add_argument(
        '--beta',
        default=0.00001,
        type=float,
        help='parameter used to filter categories with lower probabilities')
    parser.add_argument(
        '--pruning-size',
        default=50,
        type=int,
        help='use only the most probable supertags per word')
    parser.add_argument(
        '--disable-beta',
        action='store_true',
        help='disable the use of the beta value')
    parser.add_argument(
        '--disable-category-dictionary',
        action='store_true',
        help=('disable a category dictionary that maps'
              ' words to most likely supertags'))
    parser.add_argument(
        '--disable-seen-rules',
        action='store_true',
        help='')
    parser.add_argument(
        '--max-length',
        default=250,
        type=int,
        help=('give up parsing a sentence that contains'
              ' more words than this value'))
    parser.add_argument(
        '--max-step',
        default=10000000,
        type=int,
        help=('give up parsing when the number of times'
              ' of popping agenda items exceeds this value'))
    parser.add_argument(
        '--semantic-templates',
        help='semantic templates used in "ccg2lambda" format output')
    parser.add_argument(
        '--silent',
        action='store_true')
    parser.set_defaults(func=main_fun)

    subparsers = parser.add_subparsers()
    download_parser = subparsers.add_parser('download')
    download_parser.add_argument(
        'VARIANT',
        nargs='?',
        default=None,
        choices=AVAILABLE_MODEL_VARIANTS[parser.get_default('lang')])

    download_parser.set_defaults(
        func=lambda args: download(args.lang, args.VARIANT)
    )


def parse_args(main_fun):
    parser = argparse.ArgumentParser('depccg')
    parser.set_defaults(func=lambda _: parser.print_help())
    subparsers = parser.add_subparsers()

    english_parser = subparsers.add_parser('en')
    english_parser.set_defaults(lang='en')
    add_common_parser_arguments(english_parser, main_fun)
    english_parser.add_argument(
        '-a',
        '--annotator',
        default=None,
        help='annotate POS, named entity, and lemmas using this library',
        choices=english_annotator.keys())
    english_parser.add_argument(
        '-f',
        '--format',
        default='auto',
        choices=[
            'auto', 'auto_extended', 'deriv', 'xml',
            'conll', 'html', 'prolog', 'jigg_xml', 'ptb',
            'ccg2lambda', 'jigg_xml_ccg2lambda', 'json'
        ],
        help='output format')
    english_parser.add_argument(
        '--root-cats',
        default='S[dcl]|S[wq]|S[q]|S[qem]|NP',
        help=('"|" separated list of categories '
              'allowed to be at the root of a tree.')
    ),
    english_parser.add_argument(
        '--tokenize',
        action='store_true',
        help='tokenize input sentences')

    japanese_parser = subparsers.add_parser('ja')
    japanese_parser.set_defaults(lang='ja')
    add_common_parser_arguments(japanese_parser, main_fun)
    japanese_parser.add_argument(
        '-a',
        '--annotator',
        default='janome',
        help=('annotate POS, named entity,'
              ' and lemmas using this library'),
        choices=japanese_annotator.keys())
    japanese_parser.add_argument(
        '-f',
        '--format',
        default='ja',
        choices=[
            'auto', 'deriv', 'ja', 'conll',
            'html', 'jigg_xml', 'ptb', 'ccg2lambda',
            'jigg_xml_ccg2lambda', 'json', 'prolog'
        ],
        help='output format')
    japanese_parser.add_argument(
        '--root-cats',
        default=(
            'NP[case=nc,mod=nm,fin=f]|'
            'NP[case=nc,mod=nm,fin=t]|'
            'S[mod=nm,form=attr,fin=t]|'
            'S[mod=nm,form=base,fin=f]|'
            'S[mod=nm,form=base,fin=t]|'
            'S[mod=nm,form=cont,fin=f]|'
            'S[mod=nm,form=cont,fin=t]|'
            'S[mod=nm,form=da,fin=f]|'
            'S[mod=nm,form=da,fin=t]|'
            'S[mod=nm,form=hyp,fin=t]|'
            'S[mod=nm,form=imp,fin=f]|'
            'S[mod=nm,form=imp,fin=t]|'
            'S[mod=nm,form=r,fin=t]|'
            'S[mod=nm,form=s,fin=t]|'
            'S[mod=nm,form=stem,fin=f]|'
            'S[mod=nm,form=stem,fin=t]'

        ),
        help=('"|" separated list of categories '
              'allowed to be at the root of a tree.')
    )
    japanese_parser.add_argument(
        '--pre-tokenized',
        dest='tokenize',
        action='store_false',
        help=('the input is pre-tokenized'
              ' (for running parsing experiments etc.)'))

    args = parser.parse_args()
    args.func(args)
