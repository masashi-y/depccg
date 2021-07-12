
import argparse
from depccg.tools.reader import read_trees_guess_extension
from depccg.printer import print_
from depccg.annotator import japanese_annotator
from depccg.instance_models import SEMANTIC_TEMPLATES
from depccg.types import ScoringResult
import logging

logger = logging.getLogger(__name__)

LANG = 'ja'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'PATH',
        help='path to either of *.auto, *.xml, *.jigg.xml, *.ptb')
    parser.add_argument(
        '--annotator',
        default='janome',
        choices=japanese_annotator.keys(),
        help='annotate POS, named entity, and lemmas using this library')
    parser.add_argument(
        '-f',
        '--format',
        default='jigg_xml',
        choices=[
            'prolog', 'jigg_xml',
            'jigg_xml_ccg2lambda', 'json', 'deriv'],
        help='output format')
    parser.add_argument(
        '--semantic-templates',
        help='semantic templates used in "ccg2lambda" format output')
    args = parser.parse_args()

    trees = [
        [ScoringResult(tree, 0)]
        for _, _, tree in read_trees_guess_extension(args.PATH, lang='ja')
    ]
    semantic_templates = args.semantic_templates or SEMANTIC_TEMPLATES[LANG]
    print_(
        trees,
        format=args.format,
        semantic_templates=semantic_templates
    )
