
#include <Python.h>
#include <iostream>
#include "tagger.h"
#include "chainer_tagger.h"
#include "utils.h"

namespace myccg {
namespace tagger {

char PYPATH[] = "PYTHONPATH=.";

std::unique_ptr<float[]> ChainerTagger::predict(const std::string& tokens) {
    putenv(PYPATH);
    if ( !Py_IsInitialized() )
        Py_Initialize();
    initchainer_tagger();
    std::unique_ptr<float[]> res(
            new float[ this->TargetSize() * tokens.size() * sizeof(float) ]);
    tag(this->model_.c_str(), tokens.c_str(), tokens.size(), res.get());
    return res;
}

using namespace myccg;

#define RANGE(array, i, size) (array) + ((i) * (size)), \
                        (array) + ((i) * (size) + (size) - 1)

int test()
{
    int max_idx;
    std::cout << "----" << __FILE__ << "----" << std::endl;
    const std::string path = "/home/masashi-y/myccg/myccg/model";
    tagger::ChainerTagger tagg(path);
    const std::string sent = "this is a new sentence .";
    print(sent);
    std::vector<std::string> tokens = utils::split(sent, ' ');
    auto res = tagg.predict(sent);
    for (int i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res.get(), i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }

    const std::string sent2 = "Ed saw briefly his friend .";
    print(sent2);
    std::vector<std::string> tokens2 = utils::split(sent2, ' ');
    auto res2 = tagg.predict(sent2);
    for (int i = 0; i < tokens2.size(); i++) {
        std::cout << tokens2[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res2.get(), i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }
}

} // namespace tagger
} // namespace myccg

