# depccg
A\* CCG Parser with a Supertag and Dependency Factored Model

#### Requirements
* Python 2.7
* Chainer
* Cython
* A C++ compiler supporting C++11 standard
* OpenMP (optional)
* CMake

#### Build
    mkdir build
    cd build
    cmake ..
    make

#### Pretrained models
Pretrained models are available:
* [English](http://cl.naist.jp/~masashi-y/resources/depccg/en_hf_tri.tar.gz)
* [Japanese](http://cl.naist.jp/~masashi-y/resources/depccg/ja_hf_ccgbank.tar.gz)

#### Running parser
Having successfully built the sources, you'll see `depccg.so` in `build/src` directory.
In python,
```python
from depccg import PyAStarParser
model = "/path/to/model/directory"
parser = PyAStarParser(model)
res = parser.parse("this is a test sentence .")
# print res.deriv
#   NP   ((S\NP)/NP)  (NP/N)  (N/N)     N      . 
#  this      is         a     test   sentence  . 
#                            ----------------->
#                                  N ->
#                    ------------------------->
#                              NP ->
#       -------------------------------------->
#                     (S\NP) ->
# --------------------------------------------<
#                     S ->
# -----------------------------------------------<rp>
#                      S ->

# parser.parse_doc performs A* search in threads (using OpenMP), which is highly efficient. 
res = praser.parse_doc(sents) # sents: list of str
for tree in res:
    print tree.deriv
```
For Japanese CCG parsing, use `depccg.PyJaAStarParser`,
which has the exactly same interface.  
Note that the Japanese parser accepts pre-tokenized sentences as input.

`src/run.py` implements example running code. Please refer to it for
the detailed usage of the parser.
#### Training model
TODO

```
$ python lstm_parser_bi_fast.py create
usage: CCG parser's LSTM supertag tagger create [-h]
                                                [--cat-freq-cut CAT_FREQ_CUT]
                                                [--word-freq-cut WORD_FREQ_CUT]
                                                [--afix-freq-cut AFIX_FREQ_CUT]
                                                [--subset {train,test,dev,all}]
                                                [--mode {train,test}]
                                                path out
```

```
$ python lstm_parser_bi_fast.py train
usage: CCG parser's LSTM supertag tagger train [-h] [--gpu GPU]
                                               [--tritrain TRITRAIN]
                                               [--tri-weight TRI_WEIGHT]
                                               [--batchsize BATCHSIZE]
                                               [--epoch EPOCH]
                                               [--word-emb-size WORD_EMB_SIZE]
                                               [--afix-emb-size AFIX_EMB_SIZE]
                                               [--nlayers NLAYERS]
                                               [--hidden-dim HIDDEN_DIM]
                                               [--dep-dim DEP_DIM]
                                               [--dropout-ratio DROPOUT_RATIO]
                                               [--initmodel INITMODEL]
                                               [--pretrained PRETRAINED]
                                               model train val
```

We make tri-training dataset publicly available:
[English Tri-training Dataset](http://cl.naist.jp/~masashi-y/resources/depccg/headfirst_parsed.conll.stagged.gz)

#### Evaluation
You can evaluate the performance of a supertagger with `src/py/eval_tagger.py`:
```
$ python eval_tagger.py 
usage: evaluate lstm tagger [-h] [--save SAVE] model defs_dir test_data
```

For the evaluation in CCG-based dependencies, please use
evaluation scripts in [EasyCCG](https://github.com/mikelewis0/easyccg) and
[C&C](http://www.cl.cam.ac.uk/~sc609/candc-1.00.html).

#### Licence
MIT Licence

#### Contact
For questions and usage issues, please contact yoshikawa.masashi.yh8@is.naist.jp .

#### Acknowledgement
In creating the parser, I owe very much to:
- [EasyCCG](https://github.com/mikelewis0/easyccg): from which I learned everything
- [NLTK](http://www.nltk.org/): for nice pretty printing for parse derivation

