
#include <Python.h>
#include <iostream>
#include "tagger.h"
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
    for (int i = 0; i < doc.size(); i++) {
        auto& sent = doc[i];
        c_sents[i] = sent.c_str();
        lengths[i] = (int)sent.size();
        res[i] = new float[this->TargetSize() * sent.size()];
    }
    tag_doc(this->model_.c_str(), c_sents, lengths, doc.size(), res);
    return std::unique_ptr<float*[]>(res);
}

using namespace myccg;

#define RANGE(array, i, size) (array) + ((i) * (size)), \
                        (array) + ((i) * (size) + (size) - 1)

void test()
{
    int max_idx;
    std::cout << "----" << __FILE__ << "----" << std::endl;
    const std::string path = "/home/masashi-y/myccg/myccg/model";
    tagger::ChainerTagger tagg(path);
    const std::string sent = "this is a new sentence .";
    print(sent);
    std::vector<std::string> tokens = utils::split(sent, ' ');
    auto res = tagg.predict(sent);
    for (unsigned i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res.get(), i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }

    const std::string sent2 = "Ed saw briefly his friend .";
    print(sent2);
    std::vector<std::string> tokens2 = utils::split(sent2, ' ');
    auto res2 = tagg.predict(sent2);
    for (unsigned i = 0; i < tokens2.size(); i++) {
        std::cout << tokens2[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res2.get(), i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }

    const std::string sent3 = "Darth Vador , also known as Anakin Skywalker is a fictional character .";
    print(sent3);
    std::vector<std::string> tokens3 = utils::split(sent3, ' ');
    auto res3 = tagg.predict(sent3);
    for (unsigned i = 0; i < tokens3.size(); i++) {
        std::cout << tokens3[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res3.get(), i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }

    std::cout << "batch experiment" << std::endl;
    std::vector<std::string> doc{sent, sent2, sent3};
    auto res4 = tagg.predict(doc);
    auto res5 = res4[0];
    for (unsigned i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res5, i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }
    auto res6 = res4[1];
    for (unsigned i = 0; i < tokens2.size(); i++) {
        std::cout << tokens2[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res6, i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }

    auto res7 = res4[2];
    for (unsigned i = 0; i < tokens3.size(); i++) {
        std::cout << tokens3[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res7, i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }
}

} // namespace tagger
} // namespace myccg

