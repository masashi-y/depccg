
#ifndef INCLUDE_PARSER_TOOLS_H_
#define INCLUDE_PARSER_TOOLS_H_

#include <vector>
#include <utility>
#include <queue>
#include <limits>
#include <unordered_map>
#include "cat.h"
#include "tree.h"

namespace myccg {
namespace parser {

using cat::Cat;
using tree::NodeType;

bool IsModifier(Cat cat);

bool IsVerb(NodeType tree);

bool IsAdjective(Cat cat);

bool IsAdverb(Cat cat);

bool IsAUX(Cat cat);

bool IsPeriod(Cat cat);

struct AgendaItem
{
    AgendaItem(int id, NodeType parse_, float in_prob_, float out_prob_,
            int start_of_span_, int span_length_)
    : parse(parse_), in_prob(in_prob_), out_prob(out_prob_),
    prob(in_prob_ + out_prob_), start_of_span(start_of_span_),
    span_length(span_length_) {}

    ~AgendaItem() {}

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

typedef std::pair<NodeType, float> ChartItem;
struct ChartCell
{
    ChartCell():
    items(std::unordered_map<Cat, ChartItem>()),
    best_prob(std::numeric_limits<float>::lowest()), best(NULL) {}

    ~ChartCell() {}

    bool IsEmpty() const { return items.size() == 0; }

    NodeType GetBestParse() { return best; }

    bool update(NodeType parse, float prob) {
        Cat cat = parse->GetCategory();
        if (items.count(cat) > 0)
            return false;
        items.emplace(cat, std::make_pair(parse, prob));
        if (best_prob < prob) {
            best_prob = prob;
            best = parse;
        }
        return true;
    }

    std::unordered_map<Cat, ChartItem> items;
    float best_prob;
    NodeType best;
};

struct CompareFloatCat {
    bool operator()
    (const std::pair<float, Cat>& left, const std::pair<float, Cat>& right) const {
        return left.first < right.first;
    }
};

}
}

#endif
