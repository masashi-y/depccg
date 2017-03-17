# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
  ${DEPCCG_SOURCE_DIR}/src/py/py_utils.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( py_utils py_utils.pyx )

