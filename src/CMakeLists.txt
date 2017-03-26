
set_source_files_properties(
  ${DEPCCG_SOURCE_DIR}/src/depccg.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )

cython_add_module( depccg 
           depccg.pyx
           cat.cpp
           cat_loader.cpp
           chainer_tagger.cpp
           chart.cpp
           combinator.cpp
           dep.cpp
           en_grammar.cpp
           feat.cpp
           ja_grammar.cpp
           logger.cpp
           parser.cpp
           parser_tools.cpp
           tree.cpp
           utils.cpp
           )

