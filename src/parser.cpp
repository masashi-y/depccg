
#include <cmath>
#include <queue>
#include <utility>
#include <limits>
#include <memory>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parser.h"
#include "configure.h"

#define DEBUG(x) std::cerr << #x": " << (x) << std::endl;
namespace myccg {
namespace parser {

struct AgendaItem
{
    AgendaItem(int id, NodePtr parse_, float in_prob_, float out_prob_,
            int start_of_span_, int span_length_)
    : parse(parse_), in_prob(in_prob_), out_prob(out_prob_),
    prob(in_prob_ + out_prob_), start_of_span(start_of_span_),
    span_length(span_length_) {}

    ~AgendaItem() {}

    int id;
    NodePtr parse;
    float in_prob;
    float out_prob;
    float prob;
    int start_of_span;
    int span_length;

    bool operator<(const AgendaItem& other) const {
        if ( fabs(this->prob - other.prob) > 0.00001 )
            return this->prob < other.prob;
        if (this->parse->GetDependencyLength() != other.parse->GetDependencyLength())
            return this->parse->GetDependencyLength() > other.parse->GetDependencyLength();
        return this->id > other.id;
    }
};

typedef std::pair<NodePtr, float> ChartItem;
struct ChartCell
{
    ChartCell():
    items(std::unordered_map<Cat, ChartItem>()),
    best_prob(std::numeric_limits<float>::lowest()), best(NULL) {}

    ~ChartCell() {}

    bool IsEmpty() const { return items.size() == 0; }

    NodePtr GetBestParse() { return best; }

    bool update(NodePtr parse, float prob) {
        Cat cat = parse->GetCategory();
        if (items.count(cat) > 0 && prob <= best_prob)
            return false;
        items.emplace(cat, std::make_pair(parse, prob));;
        if (best_prob < prob) {
            best_prob = prob;
            best = parse;
        }
        return true;
    }

    std::unordered_map<Cat, ChartItem> items;
    float best_prob;
    NodePtr best;
};


bool AStarParser::IsAcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const {
    if (span_len == s_len)
        return (possible_root_cats_.count(cat) > 0);
    return true;
}

bool AStarParser::IsSeen(Cat left, Cat right) const {
    return (seen_rules_.count(
                std::make_pair(left->StripFeat(), right->StripFeat())) > 0);
}

bool IsNormalForm(combinator::RuleType rule_type, NodePtr left, NodePtr right) {
    return true;
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

std::vector<RuleCache>& AStarParser::GetRules(Cat left, Cat right) {
    auto key = std::make_pair(left, right);
    if (rule_cache_.count(key) > 0)
        return rule_cache_[key];
    std::vector<RuleCache> tmp;
    for (auto rule: binary_rules_) {
        if (rule->CanApply(left, right)) {
            tmp.emplace_back(rule->Apply(left, right),
                        rule->HeadIsLeft(left, right), rule);
        }
    }
    #pragma omp critical(GetRules)
    rule_cache_.emplace(key, std::move(tmp));
    return rule_cache_[key];
}

void ComputeOutsideProbs(float* probs, int sent_size, float* out) {
    float from_left[MAX_LENGTH + 1];
    float from_right[MAX_LENGTH + 1];
    from_left[0] = 0.0;
    from_right[sent_size] = 0.0;

    for (int i = 0; i < sent_size - 1; i++) {
        int j = sent_size - i;
        from_left[i + 1] = from_left[i] + probs[i];
        from_right[j - 1] = from_right[j] + probs[j - 1];
    }

    for (int i = 0; i < sent_size + 1; i++) {
        for (int j = i; j < sent_size + 1; j++) {
            out[i * sent_size + j] = from_left[i] + from_right[j];
        }
    }
}

NodePtr AStarParser::Parse(const std::string& sent, float beta) {
    std::unique_ptr<float[]> scores = tagger_->predict(sent);
    NodePtr res = Parse(sent, scores.get(), beta);
    return res;
}

std::vector<NodePtr>
AStarParser::Parse(const std::vector<std::string>& doc, float beta) {
    std::unique_ptr<float*[]> scores = tagger_->predict(doc);
    std::vector<NodePtr> res(doc.size());
    // #pragma omp parallel for schedule(PARALLEL_SCHEDULE)
    for (unsigned i = 0; i < doc.size(); i++)
        res[i] = Parse(doc[i], scores[i], beta);
    return res;
}

struct CompareFloatCat {
    bool operator()
    (const std::pair<float, Cat>& left, const std::pair<float, Cat>& right) const {
        return left.first < right.first;
    }
};

#define NORMALIZED_PROB(x, y) std::log( std::exp((x)) / (y) )

NodePtr AStarParser::Parse(const std::string& sent, float* scores, float beta) {
    int pruning_size = 50;
    std::vector<std::string> tokens = utils::Split(sent, ' ');
    int sent_size = (int)tokens.size();
    float best_in_probs[MAX_LENGTH];
    float out_probs[(MAX_LENGTH + 1) * (MAX_LENGTH + 1)];
    std::priority_queue<AgendaItem> agenda;
    int agenda_id = 0;

    float totals[MAX_LENGTH];

    std::priority_queue<std::pair<float, Cat>,
                        std::vector<std::pair<float, Cat>>,
                        CompareFloatCat> scored_cats[MAX_LENGTH];

#ifdef JAPANESE
    for (unsigned i = 0; i < tokens.size(); i++) {
        std::cerr << tokens[i] << " --> ";
        std::cerr << tagger_->TagAt( utils::ArgMax(scores + (i * TagSize()),
                    scores + (i * TagSize() + TagSize() - 1)))->ToStr() << std::endl;
    }
#endif

    for (int i = 0; i < sent_size; i++) {
        totals[i] = 0.0;
        for (int j = 0; j < TagSize(); j++) {
            float score = scores[i * TagSize() + j];
            totals[i] += std::exp(score);
            scored_cats[i].emplace(score, TagAt(j));
        }
        best_in_probs[i] = NORMALIZED_PROB( scored_cats[i].top().first, totals[i] );
    }
    ComputeOutsideProbs(best_in_probs, sent_size, out_probs);

    for (int i = 0; i < sent_size; i++) {
        float threshold = scored_cats[i].top().first * beta;
        float out_prob = out_probs[i * sent_size + (i + 1)];

        for (int j = 0; j < pruning_size; j++) {
            auto prob_and_cat = scored_cats[i].top();
            scored_cats[i].pop();
            if (prob_and_cat.first > threshold) {
                float in_prob = NORMALIZED_PROB( prob_and_cat.first, totals[i] );
                agenda.emplace(agenda_id++, std::make_shared<const tree::Leaf>(
                            tokens[i], prob_and_cat.second, i), in_prob, out_prob, i, 1);
            } else
                break;
        }
    }

    ChartCell chart[MAX_LENGTH * MAX_LENGTH];

    while (chart[sent_size - 1].IsEmpty() && agenda.size() > 0) {

        const AgendaItem item = agenda.top();
        agenda.pop();
        NodePtr parse = item.parse;
        ChartCell& cell = chart[item.start_of_span * sent_size + (item.span_length - 1)];

        if (cell.update(parse, item.in_prob)) {
            if (item.span_length != sent_size) {
                for (Cat unary: unary_rules_[parse->GetCategory()]) {
                    NodePtr subtree = std::make_shared<const tree::Tree>(unary, parse);
                    agenda.emplace(agenda_id++, subtree, item.in_prob, item.out_prob,
                                        item.start_of_span, item.span_length);
                }
            }
            for (int span_length = item.span_length + 1
                ; span_length < 1 + sent_size - item.start_of_span
                ; span_length++) {
                ChartCell& other = chart[(item.start_of_span + item.span_length) * 
                                sent_size + (span_length - item.span_length - 1)];
                for (auto&& pair: other.items) {
                    NodePtr right = pair.second.first;
                    float prob = pair.second.second;
                    if (! IsSeen(parse->GetCategory(), right->GetCategory())) continue;
                    for (auto&& rule: GetRules(parse->GetCategory(), right->GetCategory())) {
                        if (IsNormalForm(rule.combinator->GetRuleType(), parse, right) &&
                                IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodePtr subtree = std::make_shared<const tree::Tree>(
                                    rule.result, rule.left_is_head, parse, right, rule.combinator);
                            float in_prob = item.in_prob + prob;
                            float out_prob = out_probs[item.start_of_span *
                                            sent_size + item.start_of_span + span_length];
                            agenda.emplace(agenda_id++, subtree, in_prob, out_prob,
                                                item.start_of_span, span_length);
                        }
                    }
                }
            }
            for (int start_of_span = 0; start_of_span < item.start_of_span; start_of_span++) {
                int span_length = item.start_of_span + item.span_length - start_of_span;
                ChartCell& other = chart[start_of_span * sent_size +
                                    (span_length - item.span_length - 1)];
                for (auto&& pair: other.items) {
                    NodePtr left = pair.second.first;
                    float prob = pair.second.second;
                    if (! IsSeen(left->GetCategory(), parse->GetCategory())) continue;
                    for (auto&& rule: GetRules(left->GetCategory(), parse->GetCategory())) {
                        if (IsNormalForm(rule.combinator->GetRuleType(), left, parse) &&
                                IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodePtr subtree = std::make_shared<const tree::Tree>(
                                    rule.result, rule.left_is_head, left, parse, rule.combinator);
                            float in_prob = item.in_prob + prob;
                            float out_prob = out_probs[start_of_span * sent_size +
                                            start_of_span + span_length];
                            agenda.emplace(agenda_id++, subtree, in_prob, out_prob,
                                                start_of_span, span_length);
                        }
                    }
                }
            }
        }
    }
    if (chart[sent_size - 1].IsEmpty())
        return failure_node;
    auto res = chart[sent_size - 1].GetBestParse();
    return res;
}

} // namespace parser
} // namespace myccg

