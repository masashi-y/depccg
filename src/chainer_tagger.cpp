
#include <Python.h>
#include <iostream>
#include "py/tagger.h"
#include "chainer_tagger.h"
#include "utils.h"

namespace myccg {
namespace tagger {

char PYPATH[] = "PYTHONPATH=.";

std::unique_ptr<float[]> ChainerTagger::predict(const std::string& tokens) const {
    putenv(PYPATH);
    if ( !Py_IsInitialized() )
        Py_Initialize();
    initchainer_tagger();
    std::unique_ptr<float[]> res(
            new float[ this->TargetSize() * tokens.size() ]);
    tag(this->model_.c_str(), tokens.c_str(), tokens.size(), res.get());
    return res;
}

std::unique_ptr<float*[]> ChainerTagger::predict(const std::vector<std::string>& doc) const {
    putenv(PYPATH);
    if ( !Py_IsInitialized() )
        Py_Initialize();
    initchainer_tagger();
    float** res = new float*[doc.size()];
    const char** c_sents = new const char*[doc.size()];
    int* lengths = new int[doc.size()];
    for (unsigned i = 0; i < doc.size(); i++) {
        auto& sent = doc[i];
        c_sents[i] = sent.c_str();
        lengths[i] = (int)sent.size();
        res[i] = new float[this->TargetSize() * sent.size()];
    }
    tag_doc(this->model_.c_str(), c_sents, lengths, doc.size(), res);
    delete[] c_sents;
    delete[] lengths;
    return std::unique_ptr<float*[]>(res);
}

} // namespace tagger
} // namespace myccg

