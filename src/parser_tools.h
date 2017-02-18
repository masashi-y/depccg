
#ifndef INCLUDE_PARSER_TOOLS_H_
#define INCLUDE_PARSER_TOOLS_H_

#include <vector>
#include <utility>
#include <queue>
#include <unordered_map>
#include "cat.h"
#include "tree.h"

namespace myccg {


struct RuleCache
{
    RuleCache(Cat result, bool left_is_head, Op combinator)
    : result(result), left_is_head(left_is_head), combinator(combinator) {}

    Cat result;
    bool left_is_head;
    Op combinator;
};

struct AgendaItem
{
    AgendaItem(int id, NodeType parse_, float in_prob_, float out_prob_,
            int start_of_span_, int span_length_)
    : fin(false), id(id), parse(parse_), in_prob(in_prob_), out_prob(out_prob_),
    prob(in_prob_ + out_prob_), start_of_span(start_of_span_),
    span_length(span_length_) {}

    AgendaItem(bool fin, int id, NodeType parse_, float in_prob_, float out_prob_,
            int start_of_span_, int span_length_)
    : fin(fin), id(id), parse(parse_), in_prob(in_prob_), out_prob(out_prob_),
    prob(in_prob_ + out_prob_), start_of_span(start_of_span_),
    span_length(span_length_) {}

    ~AgendaItem() {}

    bool fin;
    int id;
    NodeType parse;
    float in_prob;
    float out_prob;
    float prob;
    int start_of_span;
    int span_length;

};

typedef std::priority_queue<AgendaItem,
                            std::vector<AgendaItem>,
                            bool (*)(const AgendaItem&, const AgendaItem&)
                            > AgendaType;

bool NormalComparator(const AgendaItem& left, const AgendaItem& right);
bool JapaneseComparator(const AgendaItem& left, const AgendaItem& right);
bool LongerDependencyComparator(const AgendaItem& left, const AgendaItem& right);

struct CompareFloatCat {
    bool operator()
    (const std::pair<float, Cat>& left, const std::pair<float, Cat>& right) const {
        return left.first < right.first;
    }
};


void ComputeOutsideProbs(float* probs, int sent_size, float* out);

}

#endif
