# C&C NLP tools
# Copyright (c) Universities of Edinburgh, Oxford and Sydney
# Copyright (c) James R. Curran
#
# This software is covered by a non-commercial use licence.
# See LICENCE.txt for the full text of the licence.
#
# If LICENCE.txt is not included in this distribution
# please email candc@it.usyd.edu.au to obtain a copy.

import sys
import re
import logging
import argparse
import tempfile
import os
import subprocess
from pathlib import Path

from depccg.tools.reader import read_auto
from depccg import utils
from depccg.printer.auto import auto_flattened_of

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


IGNORE_RULES = r"""
rule_id 7
rule_id 11
rule_id 12
rule_id 14
rule_id 15
rule_id 16
rule_id 17
rule_id 51
rule_id 52
rule_id 56
rule_id 91
rule_id 92
rule_id 95
rule_id 96
rule_id 98
conj 1 0
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 0
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 2
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 3
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 6
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 9
((S[b]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 6
((S[b]{_}\NP{Y}<1>){_}/PP{Z}<2>){_} 1 6
(((S[b]{_}\NP{Y}<1>){_}/PP{Z}<2>){_}/NP{W}<3>){_} 1 6
(S[X]{Y}/S[X]{Y}<1>){_} 1 13
(S[X]{Y}/S[X]{Y}<1>){_} 1 5
(S[X]{Y}/S[X]{Y}<1>){_} 1 55
((S[X]{Y}/S[X]{Y}){Z}\(S[X]{Y}/S[X]{Y}){Z}<1>){_} 2 97
((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_} 2 4
((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_} 2 93
((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_} 2 8
((S[X]{Y}\NP{Z}){Y}/(S[X]{Y}<1>\NP{Z}){Y}){_} 2 94
((S[X]{Y}\NP{Z}){Y}/(S[X]{Y}<1>\NP{Z}){Y}){_} 2 18
been ((S[pt]{_}\NP{Y}<1>){_}/(S[ng]{Z}<2>\NP{Y*}){Z}){_} 1 0
been ((S[pt]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 there 0
been ((S[pt]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 There 0
be ((S[b]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 there 0
be ((S[b]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 There 0
been ((S[pt]{_}\NP{Y}<1>){_}/(S[pss]{Z}<2>\NP{Y*}){Z}){_} 1 0
been ((S[pt]{_}\NP{Y}<1>){_}/(S[adj]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[pss]{Z}<2>\NP{Y*}){Z}){_} 1 0
have ((S[b]{_}\NP{Y}<1>){_}/(S[pt]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[adj]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[ng]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[pss]{Z}<2>\NP{Y*}){Z}){_} 1 0
going ((S[ng]{_}\NP{Y}<1>){_}/(S[to]{Z}<2>\NP{Y*}){Z}){_} 1 0
have ((S[b]{_}\NP{Y}<1>){_}/(S[to]{Z}<2>\NP{Y*}){Z}){_} 1 0
Here (S[adj]{_}\NP{Y}<1>){_} 1 0
# this is a dependency Julia doesn't have but looks okay
from (((NP{Y}\NP{Y}<1>){_}/(NP{Z}\NP{Z}){W}<3>){_}/NP{V}<2>){_} 1 0
"""

IGNORE = {tuple(rule.split())
          for rule in IGNORE_RULES.split('\n') if not rule.startswith('#')}


def die(msg):
    logger.error(msg)
    sys.exit(1)


def get_deps_from_auto(auto_file):
    candc_dir = os.environ.get('CANDC', None)
    if not candc_dir:
        die('did not find C&C parser at CANDC environmental variable.')
    CANDC_DIR = Path(candc_dir).resolve()
    GENERATE = CANDC_DIR / 'bin' / 'generate'
    MARKEDUP = CANDC_DIR / 'src' / 'data' / 'ccg' / 'cats' / 'markedup'
    CATS = CANDC_DIR / 'src' / 'data' / 'ccg' / 'cats'
    if not GENERATE.exists():
        logger.error(
            'Currently the evalution script requires C&C parser compiled from its source.')
        die('expected: $CANDC/bin/generate')
    elif not MARKEDUP.exists() or not CATS.exists:
        logger.error('The C&C directory is not configured expectedly.')
        die('expected: $CANDC/src/data/ccg/cats/markedup')

    tmp = tempfile.mktemp()
    print(tmp)
    with open(tmp, 'w') as f:
        for _, _, tree in read_auto(auto_file):
            print(auto_flattened_of(tree), file=f)

    command = f'{GENERATE} -j {CATS} {MARKEDUP} {tmp}'
    proc = subprocess.Popen(command,
                            shell=True,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    results, error = proc.communicate()
    if len(error.decode('utf-8')) > 0:
        die(
            f'caught error in running $CANDC/bin/generate: {error.decode("utf-8")}')

    lines = iter(results.decode('utf-8').split('\n'))
    deps, udeps = set(), set()
    rule_ids = {}
    line = next(lines)
    while line != '':
        line = next(lines)

    for line in lines:
        line = line.strip()
        if len(line) == 0:
            # If 0, no dependencies for this sentence - probably a conversion script error.
            parsed = len(rule_ids) > 0
            yield parsed, deps, udeps, rule_ids
            deps, udeps = set(), set()
            rule_ids = {}
            continue
        fields = line.split()
        pred, cat, slot, arg, rule_id = fields[:5]
        pred_word = pred.rsplit('_')[0]
        arg_word = arg.rsplit('_')[0]
        if not ignore(pred_word, cat, slot, arg_word, rule_id):
            cat = strip_markup(cat)
            deps.add((pred, cat, slot, arg))
            rule_ids[(pred, cat, slot, arg)] = rule_id
            udeps.add((pred, arg))


def get_pargs(file):
    try:
        lines = open(file)
    except IOError as e:
        die(f'could not open gold_deps file ({e.strerror})')

    deps, udeps = set(), set()
    for line in lines:
        line = line.strip()
        if line.startswith('<s '):
            continue
        if line.startswith('<\\s>'):
            yield deps, udeps
            deps, udeps = set(), set()
            continue
        arg_index, pred_index, cat, slot, arg, pred = line.split()[:6]
        pred = f'{utils.normalize(pred)}_{int(pred_index) + 1}'
        arg = f'{utils.normalize(arg)}_{int(arg_index) + 1}'
        deps.add((pred, cat, slot, arg))
        udeps.add((pred, arg))
    assert len(deps) == 0 and len(udeps) == 0


DEPS_IGNORED = 0


def ignore(pred, cat, slot, arg, rule_id):
    global DEPS_IGNORED
    res = ('rule_id', rule_id) in IGNORE or \
          (cat, slot, rule_id) in IGNORE or \
          (pred, cat, slot, rule_id) in IGNORE or \
          (pred, cat, slot, arg, rule_id) in IGNORE
    DEPS_IGNORED += res
    return res


MARKUP = re.compile(r'<[0-9]>|\{[A-Z_]\*?\}|\[X\]')


def strip_markup(cat):
    cat = MARKUP.sub('', cat)
    return cat[1:-1] if cat[0] == '(' else cat


def score_deps(gold_deps, test_deps, rule_ids, verbose, relations,
               correct_relations, incorrect_relations, missing_relations):
    correct = gold_deps.intersection(test_deps)
    if verbose:
        for dep in correct:
            print('correct: %s %s %s %s %s' % (dep + (rule_ids[dep],)))
    if relations:
        for dep in correct:
            correct_relations[dep[1:3]] = correct_relations.setdefault(
                dep[1:3], 0) + 1

    incorrect = test_deps.difference(gold_deps)
    if verbose:
        for dep in incorrect:
            print('incorrect: %s %s %s %s %s' % (dep + (rule_ids[dep],)))
    if relations:
        for dep in incorrect:
            incorrect_relations[dep[1:3]] = incorrect_relations.setdefault(
                dep[1:3], 0) + 1

    missing = gold_deps.difference(test_deps)
    if verbose:
        for dep in missing:
            print('missing:   %s %s %s %s ?' % dep)
    if relations:
        for dep in missing:
            missing_relations[dep[1:3]] = missing_relations.setdefault(
                dep[1:3], 0) + 1

    if verbose:
        print()
    return len(correct), len(incorrect), len(missing)


def score_udeps(gold_deps, test_deps):
    correct = gold_deps.intersection(test_deps)
    incorrect = test_deps.difference(gold_deps)
    missing = gold_deps.difference(test_deps)
    return len(correct), len(incorrect), len(missing)


def percentage(val, total):
    return 100.0 * val / total if val else 0.0


def print_acc(name, desc, correct, total):
    acc = percentage(correct, total)
    print(f'{name:6s}: {acc:5.2f}% ({correct} of {total} {desc})')


def print_stats(name, correct, incorrect, missing):
    test = correct + incorrect
    gold = correct + missing
    prec = percentage(correct, test)
    recall = percentage(correct, gold)
    fscore = (2 * prec * recall) / (prec + recall) if prec and recall else 0.0
    print(
        f'{name[0]}p:    {prec:5.2f}% ({correct} of {test} {name} deps precision)')
    print(
        f'{name[0]}r:    {recall:5.2f}% ({correct} of {gold} {name} deps recall)')
    print(f'{name[0]}f:    {fscore:5.2f}% ({name} deps f-score)')


def print_rel_stats(relation, correct, incorrect, missing):
    relation = "%s %s" % relation
    test = correct + incorrect
    prec = percentage(correct, test)
    gold = correct + missing
    recall = percentage(correct, gold)
    fscore = (2 * prec * recall) / (prec + recall) if prec and recall else 0.0
    print(f'{relation:50s}: {prec:6.2f}% {recall:6.2f}% {fscore:6.2f}% {test:6d} {gold:6d} {correct:6d}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('PARG_FILENAME', type=Path,
                        help='gold parg file in ccgbank')
    parser.add_argument('TEST', type=Path,
                        help='parse results in auto file format')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='produces verbose output')
    parser.add_argument('-r', '--relations', action='store_true',
                        help='produces per relation output')
    args = parser.parse_args()

    preface = f'# this file was generated by the following command(s):\n# {" ".join(sys.argv)}\n'

    nsentences, parse_failures = 0, 0
    deps_sent_correct, deps_correct, deps_incorrect, deps_missing = 0, 0, 0, 0
    udeps_sent_correct, udeps_correct, udeps_incorrect, udeps_missing = 0, 0, 0, 0
    relations_correct, relations_incorrect, relations_missing = {}, {}, {}

    TEST = get_deps_from_auto(args.TEST)
    for gold_deps, gold_udeps in get_pargs(args.PARG_FILENAME):
        parsed, test_deps, test_udeps, test_rule_ids = next(TEST)
        nsentences += 1
        if not parsed:
            parse_failures += 1
            continue

        correct, incorrect, missing = score_deps(gold_deps,
                                                 test_deps,
                                                 test_rule_ids,
                                                 args.verbose,
                                                 args.relations,
                                                 relations_correct,
                                                 relations_incorrect,
                                                 relations_missing)
        deps_correct += correct
        deps_incorrect += incorrect
        deps_missing += missing
        deps_sent_correct += incorrect == 0 and missing == 0

        correct, incorrect, missing = score_udeps(gold_udeps, test_udeps)
        udeps_correct += correct
        udeps_incorrect += incorrect
        udeps_missing += missing
        udeps_sent_correct += incorrect == 0 and missing == 0

    print(f'''{preface}
note: all these statistics are over just those sentences
      for which the parser returned an analysis, and
      dependency extraction script is successful
''')

    if args.relations:
        relations = relations_correct.copy()
        for relation, freq in relations_missing.items():
            relations[relation] = relations.get(relation, 0) + freq

        for relation, freq in sorted(relations.items()):
            print_rel_stats(relation,
                            relations_correct.get(relation, 0),
                            relations_incorrect.get(relation, 0),
                            relations_missing.get(relation, 0))
        print()

    nparsed = nsentences - parse_failures
    print_acc('cover', 'sentences evaluated  - this includes dependency extraction script errors and parse failures', nparsed, nsentences)
    print()
    print_stats('labelled', deps_correct, deps_incorrect, deps_missing)
    print_acc('lsent', 'labelled deps sentences correct',
              deps_sent_correct, nparsed)
    print()
    print_stats('unlabelled', udeps_correct, udeps_incorrect, udeps_missing)
    print_acc('usent', 'unlabelled deps sentences correct',
              udeps_sent_correct, nparsed)
    print()
    print_acc('skip', 'ignored deps (to ensure compatibility with CCGbank)',
              DEPS_IGNORED, deps_correct + deps_incorrect + DEPS_IGNORED)


if __name__ == '__main__':
    main()
