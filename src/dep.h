
#ifndef INCLUDE_DEP_H_
#define INCLUDE_DEP_H_

#include <iostream>
#include "tree.h"
#include "chainer_tagger.h"
#include "combinator.h"
#include "parser_tools.h"
#include "parser.h"
#include "cat.h"
#include "cat_loader.h"
#include "logger.h"

namespace myccg {

template<typename Lang>
class DepAStarParser: public AStarParser<Lang>
{
public:

    typedef AStarParser<Lang> Base;

    DepAStarParser(
            Tagger* tagger,
            const std::string& model,
            const std::unordered_set<Cat>& possible_root_cats,
            Comparator comparator,
           std::vector<Op> binary_rules,
            float beta,
            int pruning_size,
            LogLevel loglevel)
    : AStarParser<Lang>(tagger, model, possible_root_cats,
                comparator, binary_rules, beta, pruning_size, loglevel) {}

    std::vector<NodeType> Parse(const std::vector<std::string>& doc);
    std::vector<NodeType> Parse(const std::vector<std::string>& doc,
                                float** tag_scores, float** dep_scores);
    NodeType Parse(int id, const std::string& sent, float* tag_scores, float* dep_scores);
    NodeType Parse(int id, const std::string& sent, float* tag_scores) {
        return Base::Parse(id, sent, tag_scores);
    }
    std::vector<NodeType> ParseDoc(const std::vector<std::string>& doc,
                                float** tag_scores, float** dep_scores) {
        return Parse(doc, tag_scores, dep_scores);
    }
};

} // namespace myccg
#endif
