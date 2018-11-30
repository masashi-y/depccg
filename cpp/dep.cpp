
#include <cmath>
#include <queue>
#include <utility>
#include <memory>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dep.h"
#include "debug.h"
#include "parser_tools.h"
#include "grammar.h"
#include "matrix.h"
#include "chart.h"

namespace myccg {


const unsigned MAX_LENGTH = 250;

template <typename Lang>
std::vector<std::vector<ScoredNode>>
DepAStarParser<Lang>::Parse(const std::vector<std::string>& doc) {
    std::unique_ptr<float*[]> cat_scores, dep_scores;

    Base::logger_.InitStatistics(doc.size());

    Base::logger_.RecordTimeStartRunning();
    std::tie(cat_scores, dep_scores) = Base::tagger_->PredictTagsAndDeps(doc);
    Base::logger_.RecordTimeEndOfTagging();
    std::vector<std::vector<ScoredNode>> res = Parse(doc, cat_scores.get(), dep_scores.get());
    Base::logger_.RecordTimeEndOfParsing();
    Base::logger_.Report();
    return res;
}

template <typename Lang>
std::vector<std::vector<ScoredNode>>
DepAStarParser<Lang>::Parse(const std::vector<std::string>& doc,
                            float** tag_scores, float** dep_scores) {
    std::vector<std::vector<ScoredNode>> res(doc.size());
    #pragma omp parallel for schedule(dynamic, 1)
    for (unsigned i = 0; i < doc.size(); i++) {
        if ( Base::keep_going )
            res[i] = Parse(i, doc[i], tag_scores[i], dep_scores[i]);
    }
    return res;
}

template <typename Lang>
std::vector<ScoredNode> DepAStarParser<Lang>::Parse(
        int id,
        const std::string& sent,
        float* tag_scores,
        float* dep_scores) {
    std::vector<std::string> tokens = utils::Split(sent, ' ');
    int sent_size = (int)tokens.size();
    if (sent_size >= MAX_LENGTH)
        return Base::Failed(sent, "input sentence exceeding max length");

    float best_tag_probs[MAX_LENGTH];
    float best_dep_probs[MAX_LENGTH];
    float p_tag_out[(MAX_LENGTH + 1) * (MAX_LENGTH + 1)];
    float p_dep_out[(MAX_LENGTH + 1) * (MAX_LENGTH + 1)];

    Matrix<float> tag_out_probs(p_tag_out, sent_size + 1, sent_size + 1);
    Matrix<float> dep_out_probs(p_dep_out, sent_size + 1, sent_size + 1);
    Matrix<float> tag_in_probs(tag_scores, sent_size, Base::TagSize());
    Matrix<float> dep_in_probs(dep_scores, sent_size, sent_size + 1);

    AgendaType agenda(Base::comparator_);
    int agenda_id = 0;

    std::priority_queue<std::pair<float, Cat>,
                        std::vector<std::pair<float, Cat>>,
                        CompareFloatCat> scored_cats[MAX_LENGTH];

    Base::logger_.ShowTaggingOneBest(Base::tagger_, tag_scores, tokens);
    Base::logger_.ShowDependencyOneBest(dep_scores, tokens);

    float dep_leaf_out_prob = 0.0;
    for (int i = 0; i < sent_size; i++) {
        bool do_pruning = Base::use_category_dict_ &&
                            Base::category_dict_.count(tokens[i]) > 0;
        for (int j = 0; j < Base::TagSize(); j++) {
            if ( ! do_pruning ||
                    (do_pruning && Base::category_dict_[tokens[i]][j])) {
                float score = tag_in_probs(i, j);
                scored_cats[i].emplace(score, Base::TagAt(j));
            }
        }
        best_tag_probs[i] = scored_cats[i].top().first;
        int idx = dep_in_probs.ArgMax(i);
        best_dep_probs[i] = dep_in_probs(i, idx);
        dep_leaf_out_prob += dep_in_probs(i, idx);
    }

    ComputeOutsideProbs(best_tag_probs, sent_size, p_tag_out);
    ComputeOutsideProbs(best_dep_probs, sent_size, p_dep_out);

    for (int i = 0; i < sent_size; i++) {
        float threshold = Base::use_beta_ ?
            scored_cats[i].top().first * Base::beta_ : std::numeric_limits<float>::lowest();
        float out_prob = tag_out_probs(i, i + 1) + dep_leaf_out_prob;

        int j = 0;
        while (j++ < Base::pruning_size_ && 0 < scored_cats[i].size()) {
            auto prob_and_cat = scored_cats[i].top();
            scored_cats[i].pop();
            if (std::exp(prob_and_cat.first) > threshold) {
                float in_prob = prob_and_cat.first;
                agenda.emplace(agenda_id++, std::make_shared<const Leaf>(
                            tokens[i], prob_and_cat.second, i), in_prob, out_prob, i, 1);
            } else
                break;
        }
    }

    Chart chart(sent_size, Base::nbest_ > 1);
    Chart goal(1, Base::nbest_ > 1);
    ChartCell* goal_cell = goal(0, 0);

    while (Base::keep_going && Base::nbest_ > goal.Size() && agenda.size() > 0) {

        const AgendaItem item = agenda.top();
        if (item.fin) {
            goal_cell->update(item.parse, item.in_prob);
            agenda.pop();
            continue;
        }
        agenda.pop();
        NodeType parse = item.parse;
        Base::logger_.RecordAgendaItem("POPPED", item);

        ChartCell* cell = chart(item.start_of_span, item.span_length - 1);

        if (cell->update(parse, item.in_prob)) {

            if ( parse->GetLength() == sent_size &&
                    Base::possible_root_cats_.count(parse->GetCategory()) ) {
                float dep_score = dep_in_probs(parse->GetHeadId(), 0);
                float in_prob = item.in_prob +  dep_score;
                agenda.emplace(true, agenda_id++, parse, in_prob, 0.0,
                                    item.start_of_span, item.span_length);
            }

            if (sent_size == 1 || item.span_length != sent_size) {
                for (Cat unary: Base::unary_rules_[parse->GetCategory()]) {
                    if (Lang::IsAcceptableUnary(unary, parse)) {
                        NodeType subtree = std::make_shared<const Tree>(unary, parse);
                        agenda.emplace(agenda_id++, subtree, item.in_prob - 0.1, item.out_prob,
                                            item.start_of_span, item.span_length);
                        Base::logger_.RecordTree("UNARY", subtree);
                    }
                }
            }
            for (auto&& other: chart.GetCellsStartingAt(item.start_of_span + item.span_length)) {
                for (auto&& pair: other->items) {
                    NodeType right = pair.second.first;
                    float prob = pair.second.second;
                    int span_length = parse->GetLength() + right->GetLength();
                    Base::logger_.RecordTree("RIGHT", right);

                    if (! Base::IsSeen(parse->GetCategory(), right->GetCategory())) continue;

                    for (auto&& rule: Base::GetRules(parse->GetCategory(), right->GetCategory())) {
                        if (Lang::IsAcceptableBinary(rule.combinator->GetRuleType(), rule.result, parse, right)) {
                            NodeType subtree = std::make_shared<const Tree>(
                                    rule.result, rule.left_is_head, parse, right, rule.combinator);
                            NodeType head = rule.left_is_head ? parse : right;
                            NodeType dep  = rule.left_is_head ? right : parse;
                            float dep_score = dep_in_probs(dep->GetHeadId(), head->GetHeadId() + 1);
                            float in_prob = item.in_prob + prob + dep_score;
                            float out_prob = tag_out_probs(item.start_of_span,
                                            item.start_of_span + span_length)
                                           + dep_out_probs(item.start_of_span,
                                            item.start_of_span + span_length)
                                           - best_dep_probs[head->GetHeadId()];
                            agenda.emplace(agenda_id++, subtree, in_prob, out_prob,
                                                item.start_of_span, span_length);
                            Base::logger_.RecordTree("ACCEPTED", subtree);
                        }
                    }
                }
            }
            for (auto&& other: chart.GetCellsEndingAt(item.start_of_span)) {
                for (auto&& pair: other->items) {
                    NodeType left = pair.second.first;
                    float prob = pair.second.second;
                    int span_length = parse->GetLength() + left->GetLength();
                    int start_of_span = left->GetStartOfSpan();
                    Base::logger_.RecordTree("LEFT", left);

                    if (! Base::IsSeen(left->GetCategory(), parse->GetCategory())) continue;
                    for (auto&& rule: Base::GetRules(left->GetCategory(), parse->GetCategory())) {
                        if (Lang::IsAcceptableBinary(rule.combinator->GetRuleType(), rule.result, left, parse)) {
                            NodeType subtree = std::make_shared<const Tree>(
                                    rule.result, rule.left_is_head, left, parse, rule.combinator);
                            NodeType head  = rule.left_is_head ? left : parse;
                            NodeType dep   = rule.left_is_head ? parse : left;
                            float dep_score = dep_in_probs(dep->GetHeadId(), head->GetHeadId() + 1);
                            float in_prob = item.in_prob + prob + dep_score;
                            float out_prob = tag_out_probs(start_of_span,
                                            start_of_span + span_length)
                                           + dep_out_probs(start_of_span,
                                            start_of_span + span_length)
                                           - best_dep_probs[head->GetHeadId()];
                            agenda.emplace(agenda_id++, subtree, in_prob, out_prob,
                                                start_of_span, span_length);
                            Base::logger_.RecordTree("ACCEPTED", subtree);
                        }
                    }
                }
            }
        }
    }
    Base::logger_.CompleteOne(id, agenda_id);

    if (goal.IsEmpty())
        return Base::Failed(sent, "no candidate parse found");

    auto res = goal_cell->GetNBestParses();
    Base::logger_.CalculateNumOneBestTags(
            id, Base::tagger_, tag_scores, res[0].first);
    return res;
}

template class DepAStarParser<En>;
template class DepAStarParser<Ja>;

} // namespace myccg

