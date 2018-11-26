
#ifndef INCLUDE_CHAINER_TAGGER_H_
#define INCLUDE_CHAINER_TAGGER_H_

#include <memory>
#include <vector>
#include <utility>
#include <string>
#include "cat.h"
#include "cat_loader.h"
#include "debug.h"

namespace myccg {

typedef std::unique_ptr<float*[]> Probs;
typedef const std::vector<std::string> Doc;

class Tagger
{
public:
    Tagger(const std::string& model)
        : model_(model),
        targets_(utils::LoadCategoryList(model + "/target.txt")) {}

    void SetEnv(const char* path) const;

    int TargetSize() const { return this->targets_.size(); }
    Cat TagAt(int idx) const { return targets_[idx]; }
    std::vector<Cat> Targets() const { return targets_; }

    virtual Probs PredictTags(Doc& doc) const NO_IMPLEMENTATION;
    virtual std::pair<Probs, Probs> PredictTagsAndDeps(Doc& doc) const NO_IMPLEMENTATION;

protected:
    std::string model_;
    std::vector<Cat> targets_;

};

class ChainerTagger: public Tagger
{
public:
    ChainerTagger(const std::string& model): Tagger(model) {}

    Probs PredictTags(Doc& doc) const;
    std::pair<Probs, Probs> PredictTagsAndDeps(Doc& doc) const NO_IMPLEMENTATION

};

class ChainerDependencyTagger: public Tagger
{
public:
    ChainerDependencyTagger(const std::string& model): Tagger(model) {}

    Probs PredictTags(Doc& tokens) const NO_IMPLEMENTATION;
    std::pair<Probs, Probs> PredictTagsAndDeps(Doc& doc) const;
};

} // namespace myccg

#endif
