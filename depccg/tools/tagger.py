import argparse

from depccg.tools.reader import read_trees_guess_extension
from depccg.printer import print_
from depccg.lang import set_global_language_to
from depccg.annotator import english_annotator
from depccg.instance_models import SEMANTIC_TEMPLATES
from depccg.types import ScoringResult

LANG = 'en'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'PATH',
        help='path to either of *.auto, *.xml, *.jigg.xml, *.ptb')
    parser.add_argument(
        '--annotator',
        default='spacy',
        choices=english_annotator.keys(),
        help='annotate POS, named entity, and lemmas using this library')
    parser.add_argument(
        '-f',
        '--format',
        default='xml',
        choices=[
            'auto', 'xml', 'prolog', 'jigg_xml',
            'jigg_xml_ccg2lambda', 'json'],
        help='output format')
    parser.add_argument(
        '--semantic-templates',
        help='semantic templates used in "ccg2lambda" format output')
    args = parser.parse_args()

    trees = [
        [ScoringResult(tree, 0.0)]
        for _, _, tree in read_trees_guess_extension(args.PATH)
    ]

    set_global_language_to(LANG)
    semantic_templates = args.semantic_templates or SEMANTIC_TEMPLATES[LANG]
    print_(
        trees,
        format=args.format,
        semantic_templates=semantic_templates
    )
