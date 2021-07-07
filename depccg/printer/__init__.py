from typing import List, Optional

import json
from io import StringIO
from lxml import etree

from depccg.tree import ScoredTree
from depccg.types import Token
from depccg.download import SEMANTIC_TEMPLATES
from depccg.semantics.ccg2lambda import parse as ccg2lambda
from depccg.lang import GLOBAL_LANG_NAME

from depccg.printer.html import to_mathml
from depccg.printer.jigg_xml import to_jigg_xml
from depccg.printer.prolog import to_prolog_en, to_prolog_ja
from depccg.printer.xml import xml_of
from depccg.printer.ja import ja_of
from depccg.printer.conll import conll_of
from depccg.printer.json import json_of
from depccg.printer.deriv import deriv_of
from depccg.printer.ptb import ptb_of
from depccg.printer.auto import auto_of, auto_extended_of


def _process_xml(xml_node):
    return etree \
        .tostring(xml_node, encoding='utf-8', pretty_print=True) \
        .decode('utf-8')


_formatters = {
    'conll': lambda tree, tokens: conll_of(tree, tokens),
    'auto': lambda tree, tokens: auto_of(tree, tokens),
    'auto_extended': lambda tree, tokens: auto_extended_of(tree, tokens),
    'ja': lambda tree, tokens: ja_of(tree, tokens),
    'deriv': lambda tree, _: deriv_of(tree),
    'ptb': lambda tree, _: ptb_of(tree),
}


def to_string(
    nbest_trees: List[List[ScoredTree]],
    tagged_doc: Optional[List[List[Token]]],
    format: str = 'auto',
    semantic_templates: Optional[str] = None,
) -> str:
    """convert parsing results into one string representation

    Args:
        nbest_trees (List[List[ScoredTree]]): parsed results for multiple sentences
        tagged_doc (Optional[List[List[Token]]]): tokens for the sentences
        format (str, optional): format type. Defaults to 'auto'.
        available options are: 'auto', 'auto_extended', 'conll', 'deriv', 'html', 'ja',
        'json', 'ptb', 'jigg_xml', 'jigg_xml_ccg2lambda', 'ccg2lambda', 'prolog'.
        semantic_templates (Optional[str], optional): semantic template used for
        obtaining semantic formula using ccg2lambda. Defaults to None.

    Raises:
        KeyError: if the format option is not supported, this error occurs.

    Returns:
        str: string in the target format
    """

    if format in ('jigg_xml_ccg2lambda', 'ccg2lambda'):
        templates = semantic_templates or SEMANTIC_TEMPLATES.get(
            GLOBAL_LANG_NAME)
        assert templates is not None, \
            f'semantic_templates must be specified for language: {GLOBAL_LANG_NAME}'

    if format == 'conll':
        header = '# ID={}\n# log probability={:.4e}'
    else:
        header = 'ID={}, log probability={:.4e}'

    if format == 'xml':
        return _process_xml(xml_of(nbest_trees, tagged_doc))

    elif format == 'jigg_xml':
        return _process_xml(
            to_jigg_xml(
                nbest_trees,
                tagged_doc,
                use_symbol=GLOBAL_LANG_NAME == 'ja',
            )
        )

    elif format == 'jigg_xml_ccg2lambda':

        jigg_xml = to_jigg_xml(nbest_trees, tagged_doc)
        result_xml_str, _ = ccg2lambda.parse(jigg_xml, str(templates))
        return result_xml_str.decode('utf-8')

    elif format == 'prolog':  # print end=''

        if GLOBAL_LANG_NAME == 'en':
            return to_prolog_en(nbest_trees, tagged_doc)
        elif GLOBAL_LANG_NAME == 'ja':
            return to_prolog_ja(nbest_trees, tagged_doc)
        else:
            raise KeyError(
                f'prolog format is not supported for language {GLOBAL_LANG_NAME}'
            )

    elif format == 'html':
        return to_mathml(nbest_trees)

    elif format == 'json':
        results = {}
        for sentence_index, (trees, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
            results[sentence_index] = []
            for tree, log_prob in trees:
                tree_dict = json_of(tree, tokens=tokens)
                tree_dict['log_prob'] = log_prob
                results[sentence_index].append(tree_dict)
        return json.dumps(results)

    elif format == 'ccg2lambda':

        with StringIO() as file:
            jigg_xml = to_jigg_xml(nbest_trees, tagged_doc)
            _, formulas_list = ccg2lambda.parse(jigg_xml, str(templates))
            for sentence_index, (trees, formulas) in enumerate(zip(nbest_trees, formulas_list), 1):
                for (tree, log_prob), formula in zip(trees, formulas):
                    print(header.format(sentence_index, log_prob), file=file)
                    print(formula, file=file)
            return file.getvalue()

    try:
        formatter = _formatters[format]
    except KeyError:
        raise KeyError(
            f'unsupported format type: {format}'
        )

    with StringIO() as file:
        for sentence_index, (trees, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
            for tree, log_prob in trees:
                print(header.format(sentence_index, log_prob), file=file)
                print(formatter(tree, tokens), file=file)

        return file.getvalue()


def print_(
    nbest_trees: List[List[ScoredTree]],
    tagged_doc: Optional[List[List[Token]]],
    format: str = 'auto',
    semantic_templates: Optional[str] = None,
    **kwargs,
) -> None:
    """print parsing results into one string representation

    Args:
        nbest_trees (List[List[ScoredTree]]): parsed results for multiple sentences
        tagged_doc (Optional[List[List[Token]]]): tokens for the sentences
        format (str, optional): format type. Defaults to 'auto'.
        available options are: 'auto', 'auto_extended', 'conll', 'deriv', 'html', 'ja',
        'json', 'ptb', 'jigg_xml', 'jigg_xml_ccg2lambda', 'ccg2lambda', 'prolog'.
        semantic_templates (Optional[str], optional): semantic template used for
        obtaining semantic formula using ccg2lambda. Defaults to None.

    other keyword arguments for Python 'print' function are also available.

    Raises:
        KeyError: if the format option is not supported, this error occurs.
    """

    print(
        to_string(
            nbest_trees,
            tagged_doc,
            format=format,
            semantic_templates=semantic_templates,
        ),
        **kwargs,
    )
