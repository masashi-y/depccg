
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
#include "debug.h"
#include "grammar.h"

namespace myccg {


template<typename Lang>
bool AStarParser<Lang>::IsAcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const {
    if (span_len == s_len)
        return (possible_root_cats_.count(cat) > 0);
    return true;
}

template<typename Lang>
bool AStarParser<Lang>::IsSeen(Cat left, Cat right) const {
#ifdef JAPANESE
    return true;
#endif
    return (seen_rules_.count(
                std::make_pair(left->StripFeat(), right->StripFeat())) > 0);
}


template<typename Lang>
std::vector<RuleCache>& AStarParser<Lang>::GetRules(Cat left, Cat right) {
    auto key = std::make_pair(left, right);
    if (rule_cache_.count(key) > 0)
        return rule_cache_[key];
    std::vector<RuleCache> tmp;
    for (auto rule: Lang::binary_rules) {
        if (rule->CanApply(left, right)) {
            tmp.emplace_back(rule->Apply(left, right),
                        rule->HeadIsLeft(left, right), rule);
        }
    }
    #pragma omp critical(GetRules)
    rule_cache_.emplace(key, std::move(tmp));
    return rule_cache_[key];
}

template<typename Lang>
NodeType AStarParser<Lang>::Parse(const std::string& sent) {
    std::unique_ptr<float[]> scores = tagger_->predict(sent);
    NodeType res = Parse(sent, scores.get());
    return res;
}

template<typename Lang>
std::vector<NodeType> AStarParser<Lang>::Parse(const std::vector<std::string>& doc) {
    std::unique_ptr<float*[]> scores = tagger_->predict(doc);
    std::vector<NodeType> res(doc.size());
    #pragma omp parallel for schedule(PARALLEL_SCHEDULE)
    for (unsigned i = 0; i < doc.size(); i++) {
        res[i] = Parse(doc[i], scores[i]);
    }
    return res;
}

#define NORMALIZED_PROB(x, y) std::log( std::exp((x)) / (y) )

template<typename Lang>
NodeType AStarParser<Lang>::Parse(const std::string& sent, float* scores) {
    std::vector<std::string> tokens = utils::Split(sent, ' ');
    int sent_size = (int)tokens.size();
    float best_in_probs[MAX_LENGTH];
    float out_probs[(MAX_LENGTH + 1) * (MAX_LENGTH + 1)];
    AgendaType agenda(comparator_);
    int agenda_id = 0;

    float totals[MAX_LENGTH];

    std::priority_queue<std::pair<float, Cat>,
                        std::vector<std::pair<float, Cat>>,
                        CompareFloatCat> scored_cats[MAX_LENGTH];

#ifdef DEBUGGING

    for (unsigned i = 0; i < tokens.size(); i++) {
        std::cerr << tokens[i] << " --> ";
        std::cerr << tagger_->TagAt( utils::ArgMax(scores + (i * TagSize()),
                    scores + (i * TagSize() + TagSize() - 1)))->ToStr() << std::endl;
    }
    std::cerr << std::endl;
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
        float threshold = scored_cats[i].top().first * beta_;
        float out_prob = out_probs[i * sent_size + (i + 1)];

        for (int j = 0; j < pruning_size_; j++) {
            auto prob_and_cat = scored_cats[i].top();
            scored_cats[i].pop();
            if (prob_and_cat.first > threshold) {
                float in_prob = NORMALIZED_PROB( prob_and_cat.first, totals[i] );
                agenda.emplace(agenda_id++, std::make_shared<const Leaf>(
                            tokens[i], prob_and_cat.second, i), in_prob, out_prob, i, 1);
            } else
                break;
        }
    }

    ChartCell chart[MAX_LENGTH * MAX_LENGTH];

    while (chart[sent_size - 1].IsEmpty() && agenda.size() > 0) {

        const AgendaItem item = agenda.top();
        agenda.pop();
        NodeType parse = item.parse;

#ifdef DEBUGGING
        POPPED;
        std::cerr << Derivation(parse);
        std::cerr << "score: " << item.prob << std::endl;
        DEBUG(item.start_of_span);
        DEBUG(item.span_length);
        BORDER;
#endif

        ChartCell& cell = chart[item.start_of_span * sent_size + (item.span_length - 1)];

        if (cell.update(parse, item.in_prob)) {
            if (item.span_length != sent_size) {
                for (Cat unary: unary_rules_[parse->GetCategory()]) {
                    NodeType subtree = std::make_shared<const Tree>(unary, parse);
                    agenda.emplace(agenda_id++, subtree, item.in_prob, item.out_prob,
                                        item.start_of_span, item.span_length);
#ifdef DEBUGGING
        TREETYPE("UNARY");
        std::cerr << Derivation(subtree);
        BORDER;
#endif
                }
            }
            for (int span_length = item.span_length + 1
                ; span_length < 1 + sent_size - item.start_of_span
                ; span_length++) {
                ChartCell& other = chart[(item.start_of_span + item.span_length) * 
                                sent_size + (span_length - item.span_length - 1)];
                for (auto&& pair: other.items) {
                    NodeType right = pair.second.first;
                    float prob = pair.second.second;
#ifdef DEBUGGING
        TREETYPE("RIGHT ARG");
        std::cerr << Derivation(right);
        BORDER;
#endif
                    if (! IsSeen(parse->GetCategory(), right->GetCategory())) continue;
                    for (auto&& rule: GetRules(parse->GetCategory(), right->GetCategory())) {
                        if (Lang::IsAcceptable(rule.combinator->GetRuleType(), parse, right) &&
                                IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodeType subtree = std::make_shared<const Tree>(
                                    rule.result, rule.left_is_head, parse, right, rule.combinator);
                            float in_prob = item.in_prob + prob;
                            float out_prob = out_probs[item.start_of_span *
                                            sent_size + item.start_of_span + span_length];
                            agenda.emplace(agenda_id++, subtree, in_prob, out_prob,
                                                item.start_of_span, span_length);
#ifdef DEBUGGING
        ACCEPT;
        std::cerr << Derivation(subtree);
        BORDER;
#endif
                        }
                    }
                }
            }
            for (int start_of_span = 0; start_of_span < item.start_of_span; start_of_span++) {
                int span_length = item.start_of_span + item.span_length - start_of_span;
                ChartCell& other = chart[start_of_span * sent_size +
                                    (span_length - item.span_length - 1)];
                for (auto&& pair: other.items) {
                    NodeType left = pair.second.first;
                    float prob = pair.second.second;
#ifdef DEBUGGING
        TREETYPE("LEFT ARG");
        std::cerr << Derivation(left);
        BORDER;
#endif
                    if (! IsSeen(left->GetCategory(), parse->GetCategory())) continue;
                    for (auto&& rule: GetRules(left->GetCategory(), parse->GetCategory())) {
                        if (Lang::IsAcceptable(rule.combinator->GetRuleType(), left, parse) &&
                                IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodeType subtree = std::make_shared<const Tree>(
                                    rule.result, rule.left_is_head, left, parse, rule.combinator);
                            float in_prob = item.in_prob + prob;
                            float out_prob = out_probs[start_of_span * sent_size +
                                            start_of_span + span_length];
                            agenda.emplace(agenda_id++, subtree, in_prob, out_prob,
                                                start_of_span, span_length);
                        }
                    }
#ifdef DEBUGGING
        ACCEPT;
        std::cerr << Derivation(subtree);
        BORDER;
#endif
                }
            }
        }
    }
    if (chart[sent_size - 1].IsEmpty())
        return failure_node;
    auto res = chart[sent_size - 1].GetBestParse();
    std::cerr << ".";
    return res;
}

template class AStarParser<En>;
template class AStarParser<Ja>;

} // namespace myccg

