# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
  ${DEPCCG_SOURCE_DIR}/src/depccg.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
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
           py/tagger.cpp
           tree.cpp
           utils.cpp
           )

add_subdirectory( py )
