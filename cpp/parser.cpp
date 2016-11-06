
#include "parser.h"
#include <cmath>
#include <unordered_map>
#include <queue>
#include <utility>
#include <limits>

namespace myccg {
namespace parser {

struct AgendaItem
{
    AgendaItem(const tree::Node* parse, float in_prob, float out_prob,
            int start_of_span, int span_length)
    : parse(parse), in_prob(in_prob), out_prob(out_prob),
    cost(in_prob + out_prob), start_of_span(start_of_span),
    span_length(span_length) {}

    const tree::Node* parse;
    const float in_prob;
    const float out_prob;
    const float cost;
    const int start_of_span;
    const int span_length;

};

bool operator<(const AgendaItem& item1, const AgendaItem& item2) {
    if ( fabs(item1.cost - item2.cost) > 0.0000001 )
        return item1.cost < item2.cost;
    return item1.parse->GetDependencyLength() > item2.parse->GetDependencyLength();
}

typedef std::pair<const tree::Node*, float> ChartItem;
class ChartCell
{
    ChartCell():
    items_(std::unordered_map<const cat::Category*, ChartItem>()),
    best_cost_(std::numeric_limits<float>::max()), best_(NULL) {}

    bool update(const tree::Node* parse, float cost) {
        return true;
    }

    std::unordered_map<const cat::Category*, ChartItem> items_;
    float best_cost_;
    const tree::Node* best_;
};

struct RuleCache
{
    const cat::Category* lchild;
    const cat::Category* rchild;
    const cat::Category* result;
    const bool left_is_head;
    const combinator::Combinator* combinator;
};

tree::Node* AStarParser::parse(std::string& sent) {
    return NULL;
}

bool is_normal_form(combinator::RuleType rule_type,
        const tree::Node* left, const tree::Node* right) {
    if ( (left->GetRuleType() == combinator::FC ||
                left->GetRuleType() == combinator::GFC) &&
            (rule_type == combinator::FA ||
             rule_type == combinator::FC ||
             rule_type == combinator::GFC) )
        return false;
    if ( (right->GetRuleType() == combinator::BX ||
                left->GetRuleType() == combinator::GBX) &&
            (rule_type == combinator::BA ||
             rule_type == combinator::BX ||
             left->GetRuleType() == combinator::GBX) )
        return false;
    if ( left->GetRuleType() == combinator::UNARY &&
            rule_type == combinator::FA &&
            left->GetCategory()->IsForwardTypeRaised() )
        return false;
    if ( right->GetRuleType() == combinator::UNARY &&
            rule_type == combinator::BA &&
            right->GetCategory()->IsBackwardTypeRaised() )
        return false;
    return true;
}

void compute_outside_probs(float* probs, int sentSize, float* out) {
    int j;
    float* from_left = new float[sentSize + 1];
    float* from_right = new float[sentSize + 1];
    from_left[0] = 0.0;
    from_right[sentSize] = 0.0;

    for (int i = 0; i < sentSize; i++) {
        j = sentSize - i;
        from_left[i + 1] = from_left[i] + probs[i];
        from_right[j - 1] = from_right[j] + probs[j - 1];
    }

    for (int i = 0; i < sentSize + 1; i++) {
        for (int j = 0; j < sentSize + 1; j++) {
            out[i * sentSize + j] = from_left[i] + from_right[j];
        }
    }
    delete[] from_left;
    delete[] from_right;
}

} // namespace parser
} // namespace myccg

int main()
{
    myccg::tagger::test();
    myccg::tree::test();
}

