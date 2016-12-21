
#include <cmath>
#include <queue>
#include <utility>
#include <memory>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parser.h"
#include "configure.h"
#include "debug.h"
#include "parser_tools.h"
#include "grammar.h"

namespace myccg {



template <typename Lang>
NodeType DepAStarParser<Lang>::Parse(const std::string& sent) NO_IMPLEMENTATION

template <typename Lang>
std::vector<NodeType>
DepAStarParser<Lang>::Parse(const std::vector<std::string>& doc) {
    std::unique_ptr<float*[]> cat_scores, dep_scores;
    std::tie(cat_scores, dep_scores) = dep_tagger_->predict(doc);
    std::vector<NodeType> res(doc.size());
    #pragma omp parallel for schedule(PARALLEL_SCHEDULE)
    for (unsigned i = 0; i < doc.size(); i++) {
        res[i] = Parse(doc[i], cat_scores[i], dep_scores[i]);
    }
    return res;
}

#define NORMALIZED_PROB(x, y) std::log( std::exp((x)) / (y) )

template <typename Lang>
NodeType DepAStarParser<Lang>::Parse(
        const std::string& sent,
        float* tag_scores,
        float* dep_scores) {
    std::vector<std::string> tokens = utils::Split(sent, ' ');
    int sent_size = (int)tokens.size();
    float best_in_probs[MAX_LENGTH];
    float out_probs[(MAX_LENGTH + 1) * (MAX_LENGTH + 1)];
    AgendaType agenda(Base::comparator_);
    int agenda_id = 0;

    float tag_totals[MAX_LENGTH], dep_totals[MAX_LENGTH];

    std::priority_queue<std::pair<float, Cat>,
                        std::vector<std::pair<float, Cat>>,
                        CompareFloatCat> scored_cats[MAX_LENGTH];

#ifdef DEBUGGING
    for (int i = 0; i < sent_size; i++) {
        std::cerr << tokens[i] << " --> ";
        std::cerr << dep_tagger_->TagAt( utils::ArgMax(tag_scores + (i * TagSize()),
                    tag_scores + (i * TagSize() + TagSize() - 1))) << std::endl;
    }
    std::cerr << std::endl;

    for (int i = 0; i < sent_size; i++) {
        std::cerr << i + 1 << " " << tokens[i] << " --> ";
        std::cerr << utils::ArgMax(dep_scores + (i * (sent_size + 1)),
                    dep_scores + (i * (sent_size + 1) + (sent_size + 1) - 1)) << std::endl;
    }
    std::cerr << std::endl;
#endif

    for (int i = 0; i < sent_size; i++) {
        tag_totals[i] = 0.0;
        dep_totals[i] = 0.0;
        for (int j = 0; j < TagSize(); j++) {
            float score = tag_scores[i * TagSize() + j];
            tag_totals[i] += std::exp(score);
            scored_cats[i].emplace(score, TagAt(j));
        }
        best_in_probs[i] = NORMALIZED_PROB( scored_cats[i].top().first, tag_totals[i] );
    }
    ComputeOutsideProbs(best_in_probs, sent_size, out_probs);

    for (int i = 0; i < sent_size; i++) {
        float threshold = scored_cats[i].top().first * Base::beta_;
        float out_prob = out_probs[i * sent_size + (i + 1)];
        float dep_score = std::log(0.000000000001); //dep_scores[i * sent_size + i + 1];

        for (int j = 0; j < Base::pruning_size_; j++) {
            auto prob_and_cat = scored_cats[i].top();
            scored_cats[i].pop();
            if (prob_and_cat.first > threshold) {
                float in_prob = dep_score + NORMALIZED_PROB( prob_and_cat.first, tag_totals[i] );
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
                for (Cat unary: Base::unary_rules_[parse->GetCategory()]) {
                    NodeType subtree = std::make_shared<const Tree>(unary, parse);
#ifdef DEBUGGING
        TREETYPE("UNARY");
        std::cerr << Derivation(subtree);
        BORDER;
#endif
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
                    NodeType right = pair.second.first;
#ifdef DEBUGGING
        TREETYPE("RIGHT ARG");
        std::cerr << Derivation(right);
        std::cerr << "start_of_span: " << item.start_of_span + item.span_length << std::endl;
        std::cerr << "span_length: " << span_length - item.span_length << std::endl;
        BORDER;
#endif
                    float prob = pair.second.second;
                    if (! Base::IsSeen(parse->GetCategory(), right->GetCategory())) continue;
                    for (auto&& rule: Base::GetRules(parse->GetCategory(), right->GetCategory())) {
                        if (Lang::IsAcceptable(rule.combinator->GetRuleType(), parse, right) &&
                                Base::IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodeType subtree = std::make_shared<const Tree>(
                                    rule.result, rule.left_is_head, parse, right, rule.combinator);
#ifdef DEBUGGING
        ACCEPT;
        std::cerr << Derivation(subtree);
        BORDER;
#endif
                            int head = rule.left_is_head ? parse->GetHeadId() : right->GetHeadId();
                            int dep  = rule.left_is_head ? right->GetHeadId() : parse->GetHeadId();
                            float dep_score = dep_scores[dep * (sent_size + 1) + head + 1];
                            std::cerr << dep << ", " << head << ", " << sent_size << std::endl;
                            std::cerr << tokens[dep] << " --> " << tokens[head] << ": " << dep_score << std::endl;
                            float in_prob = item.in_prob + prob + dep_score;
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
                    NodeType left = pair.second.first;
                    float prob = pair.second.second;
#ifdef DEBUGGING
        TREETYPE("LEFT ARG");
        std::cerr << Derivation(left);
        std::cerr << "start_of_span: " << start_of_span << std::endl;
        std::cerr << "span_length: " << span_length - item.span_length << std::endl;
        BORDER;
#endif
                    if (! Base::IsSeen(left->GetCategory(), parse->GetCategory())) continue;
                    for (auto&& rule: Base::GetRules(left->GetCategory(), parse->GetCategory())) {
                        if (Lang::IsAcceptable(rule.combinator->GetRuleType(), left, parse) &&
                                Base::IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodeType subtree = std::make_shared<const Tree>(
                                    rule.result, rule.left_is_head, left, parse, rule.combinator);
#ifdef DEBUGGING
        ACCEPT;
        std::cerr << Derivation(subtree);
        BORDER;
#endif
                            int head  = rule.left_is_head ? left->GetHeadId() : parse->GetHeadId();
                            int dep = rule.left_is_head ? parse->GetHeadId() : left->GetHeadId();
                            float dep_score = dep_scores[dep * (sent_size + 1) + head + 1];
                            std::cerr << dep << ", " << head << ", " << sent_size << std::endl;
                            std::cerr << tokens[dep] << " --> " << tokens[head] << ": " << dep_score << std::endl;
                            float in_prob = item.in_prob + prob + dep_score;
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
        return Base::failure_node;
    auto res = chart[sent_size - 1].GetBestParse();
    return res;
}

template class DepAStarParser<En>;
template class DepAStarParser<Ja>;

} // namespace myccg

