
from .reader import read_trees_guess_extension
from ..printer import to_mathml
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


index_html = '''\
<html>
<head>
  <title>parse results</title>
</head>
<body>
<table border=1>
  <tr><td>id</td><td>file</td><td>sentence</td></tr>
  {}
</table>
</body>
</html>
'''

def to_mathml_separate_files(trees, directory_name='results'):
    out_directory = Path(directory_name)
    if out_directory.exists():
        raise RuntimeError(f'Directory "{directory_name}" already exists. Use other name.')
    out_directory.mkdir()
    trs = []
    for i, nbest in enumerate(trees):
        filename = f'{i}.html'
        mathml_html = to_mathml([nbest])
        with (out_directory / filename).open('w') as f:
            print(mathml_html, file=f)
        words = nbest[0][0].word if isinstance(nbest[0], tuple) else nbest[0].word
        trs.append(f'<tr><td>{i}</td><td><a href="{filename}">{filename}</a></td><td>{words}</td></tr>')
    with (out_directory / 'index.html').open('w') as f:
        print(index_html.format('\n'.join(trs)), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('PATH', help='path to either of *.auto, *.xml, *.jigg.xml, *.ptb')
    parser.add_argument('--mkdir', default=None, help='if specified, creates a directory')
    args = parser.parse_args()

    trees = [[tree] for _, _, tree in read_trees_guess_extension(args.PATH)]
    if not args.mkdir:
        print(to_mathml(trees))
    else:
        to_mathml_separate_files(trees, args.mkdir)
