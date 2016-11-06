
#ifndef INCLUDE_CHAINER_TAGGER_H_
#define INCLUDE_CHAINER_TAGGER_H_

#include <memory>
#include <vector>
#include <string>
#include <stdlib.h>
#include "cat.h"
#include "utils.h"

namespace myccg {
namespace tagger {


class Tagger
{
    virtual std::unique_ptr<float[]> predict(const std::string& tokens) {};

    virtual std::vector<std::unique_ptr<float[]>> predict(const std::vector<std::string>& doc) {};
};

class ChainerTagger: public Tagger
{
public:
    ChainerTagger(const std::string& model): model_(model) {
        const std::string targetfile = model + "/target.txt";
        targets_ = utils::load_category_list(targetfile);
    }
 
    std::unique_ptr<float[]> predict(const std::string& tokens);

    int TargetSize() { return this->targets_.size(); }

    const cat::Category* TagAt(int idx) { return targets_[idx]; }

    // std::vector<std::unique_ptr<float[]>> predict(const std::vector<std::string>& doc);

private:
    const std::string& model_;
    std::vector<const cat::Category*> targets_;

};

int test();
        
} // namespace tagger
} // namespace myccg

#endif
