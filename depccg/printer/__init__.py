from typing import List, Optional, Union

import json
from io import StringIO
from lxml import etree

from depccg.tree import ScoredTree
from depccg.instance_models import SEMANTIC_TEMPLATES
from depccg.semantics.ccg2lambda import parse as ccg2lambda
from depccg.lang import get_global_language

from depccg.printer.html import to_mathml
from depccg.printer.jigg_xml import to_jigg_xml
from depccg.printer.prolog import to_prolog_en, to_prolog_ja
from depccg.printer.xml import xml_of
from depccg.printer.ja import ja_of
from depccg.printer.conll import conll_of
from depccg.printer.my_json import json_of
from depccg.printer.deriv import deriv_of
from depccg.printer.ptb import ptb_of
from depccg.printer.auto import auto_of, auto_extended_of


def _process_xml(xml_node):
    return etree \
        .tostring(xml_node, encoding='utf-8', pretty_print=True) \
        .decode('utf-8')


_formatters = {
    'conll': conll_of,
    'auto': auto_of,
    'auto_extended': auto_extended_of,
    'ja': ja_of,
    'deriv': deriv_of,
    'ptb': ptb_of,
}


def to_string(
    nbest_trees: List[Union[List[ScoredTree], ScoredTree]],
    format: str = 'auto',
    semantic_templates: Optional[str] = None,
) -> str:
    """convert parsing results into one string representation

    Args:
        nbest_trees (List[Union[List[ScoredTree], ScoredTree]]): 
            parsed results for multiple sentences
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
    if isinstance(nbest_trees[0], ScoredTree):
        nbest_trees = [nbest_trees]
    elif not (
        isinstance(nbest_trees[0], list)
        and isinstance(nbest_trees[0][0], ScoredTree)
    ):
        raise RuntimeError('invalid argument type for stringifying trees')

    if format in ('jigg_xml_ccg2lambda', 'ccg2lambda'):
        lang = get_global_language()
        templates = semantic_templates or SEMANTIC_TEMPLATES.get(lang)
        assert templates is not None, \
            f'semantic_templates must be specified for language: {lang}'

    if format == 'conll':
        header = '# ID={}\n# log probability={:.8f}'
    else:
        header = 'ID={}, log probability={:.8f}'

    if format == 'xml':
        return _process_xml(xml_of(nbest_trees))

    elif format == 'jigg_xml':
        return _process_xml(
            to_jigg_xml(
                nbest_trees,
                use_symbol=get_global_language() == 'ja',
            )
        )

    elif format == 'jigg_xml_ccg2lambda':

        jigg_xml = to_jigg_xml(nbest_trees)
        result_xml_str, _ = ccg2lambda.parse(
            jigg_xml, str(templates), ncores=1
        )
        return result_xml_str.decode('utf-8')

    elif format == 'prolog':  # print end=''
        lang = get_global_language()
        if lang == 'en':
            return to_prolog_en(nbest_trees)
        elif lang == 'ja':
            return to_prolog_ja(nbest_trees)
        else:
            raise KeyError(
                f'prolog format is not supported for language {lang}'
            )

    elif format == 'html':
        return to_mathml(nbest_trees)

    elif format == 'json':
        results = {}
        for sentence_index, trees in enumerate(nbest_trees, 1):
            results[sentence_index] = []
            for tree, log_prob in trees:
                tree_dict = json_of(tree)
                tree_dict['log_prob'] = log_prob
                results[sentence_index].append(tree_dict)
        return json.dumps(results, indent=4)

    elif format == 'ccg2lambda':

        with StringIO() as file:
            jigg_xml = to_jigg_xml(nbest_trees)
            _, formulas_list = ccg2lambda.parse(
                jigg_xml, str(templates), ncores=1
            )
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
        for sentence_index, trees in enumerate(nbest_trees, 1):
            for tree, log_prob in trees:
                print(header.format(sentence_index, log_prob), file=file)
                print(formatter(tree), file=file)

        return file.getvalue()


def print_(
    nbest_trees: List[Union[List[ScoredTree], ScoredTree]],
    format: str = 'auto',
    semantic_templates: Optional[str] = None,
    **kwargs,
) -> None:
    """print parsing results into one string representation

    Args:
        nbest_trees (List[Union[List[ScoredTree], ScoredTree]]):
         parsed results for multiple sentences
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
            format=format,
            semantic_templates=semantic_templates,
        ),
        **kwargs,
    )
