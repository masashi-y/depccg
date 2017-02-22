
#include <string.h>
#include "logger.h"
#include "utils.h"
#include "matrix.h"

#define BORDER "####################################"

namespace myccg {

ParserLogger::ParserLogger(LogLevel level)
    : level_(level), cur_level_(Info), nprocessed_(0),
      do_statistics_(false),
      num_sents_(0), agenda_sizes_(nullptr),
      num_one_best_tags_(nullptr) {}

ParserLogger::~ParserLogger() {
    delete[] agenda_sizes_;
}

void ParserLogger::InitStatistics(int num_sents) {
    do_statistics_ = true;
    num_sents_ = num_sents;
    agenda_sizes_ = new int[num_sents];
    memset(agenda_sizes_, 0, sizeof(int) * num_sents);
    num_one_best_tags_ = new double[num_sents];
    memset(num_one_best_tags_, 0, sizeof(double) * num_sents);
}

void ParserLogger::ShowStatistics() {
    if (! do_statistics_)
        return;
    double average_agenda_size = 0.0;
    for (int i = 0; i < num_sents_; i++)
        average_agenda_size += (double)agenda_sizes_[i];
    double average_num_one_best = 0.0;
    for (int i = 0; i < num_sents_; i++)
        average_num_one_best += num_one_best_tags_[i];
    average_agenda_size /= (double)num_sents_;
    average_num_one_best /= (double)num_sents_;
    std::cerr << "average agenda size: " << average_agenda_size << std::endl;
    std::cerr << "average # of tokens with one best tag: " << average_num_one_best << std::endl;
}

void ParserLogger::RecordTime(const char* name) {
    auto now = std::chrono::system_clock::now();
    times_.emplace(name, now);
}

void ParserLogger::RecordTime(const std::string& name) {
    RecordTime(name.c_str());
}

void ParserLogger::RecordTimeStartRunning() {
    (*this)(Info) << "running super tagger ...";
    RecordTime("time_start_running");
}

void ParserLogger::RecordTimeEndOfTagging() {
    (*this)(Info) << "finished";
    (*this)(Info) << "running A* parser ...";
    RecordTime("time_end_of_tagging");
}

void ParserLogger::RecordTimeEndOfParsing() {
    (*this)(Info) << "finished";
    RecordTime("time_end_of_parsing");
}

void ParserLogger::RecordTree(const char* message, NodeType tree) {
    if (level_ == Debug)
        std::cerr << Red(message) << std::endl
                  << Derivation(tree, false) << std::endl
                  << Cyan(BORDER) << std::endl;
}

void ParserLogger::RecordAgendaItem(const char* message, const AgendaItem& item) {
    NodeType tree = item.parse;
    if (level_ == Debug)
        std::cerr << Blue(message) << std::endl
                  << "id: " << item.id << std::endl
                  << Derivation(tree, false) << std::endl
                  << "s: " << item.in_prob << std::endl
                  << "h: " << item.out_prob << std::endl
                  << "s + h: " << item.prob << std::endl
                  << Cyan(BORDER) << std::endl;
}

void ParserLogger::CompleteOne() {
    if (++nprocessed_ % 10 == 0 && level_ <= Info) {
        std::cerr << ".";
        if (nprocessed_ % 500 == 0)
            std::cerr << nprocessed_ << std::endl;
    }
}

void ParserLogger::CompleteOne(int id, int agenda_size) {
    if (! do_statistics_)
        return;
    agenda_sizes_[id] = agenda_size;
    CompleteOne();
}
void ParserLogger::CalculateNumOneBestTags(
        int id, Tagger* tagger, float* probs, NodeType parse) {
    if (! do_statistics_)
        return;
    int res = 0;
    auto leaves = GetLeaves()(parse.get());
    Matrix<float> mat(probs, leaves.size(), tagger->TargetSize());
    for (unsigned i = 0; i < leaves.size(); i++) {
        auto one_best = tagger->TagAt( mat.ArgMax(i) );
        if ( *one_best == *leaves[i]->GetCategory() )
            res++;
    }
    num_one_best_tags_[id] = (double)res / (double)leaves.size();
}

void ParserLogger::ShowTaggingOneBest(
        Tagger* tagger, float* probs, std::vector<std::string>& tokens) {
    if (level_ > Debug) return;
    Matrix<float> mat(probs, tokens.size(), tagger->TargetSize());
    for (unsigned i = 0; i < tokens.size(); i++) {
        int idx = mat.ArgMax(i);
        std::cerr << tokens[i] << "\t"
                  << tagger->TagAt( idx ) << "\t" << mat(i, idx) << std::endl;
    }
    std::cerr << std::endl;
}

void ParserLogger::ShowDependencyOneBest(
        float* probs, std::vector<std::string>& tokens) {
    if (level_ > Debug) return;
    unsigned sent_size = tokens.size();
    Matrix<float> mat(probs, sent_size, sent_size + 1);
    for (unsigned i = 0; i < sent_size; i++) {
        int idx = mat.ArgMax(i);
        std::cerr << i + 1 << "\t" << tokens[i] << "\t"
                  << idx << "\t" << mat(i, idx) << std::endl;
    }
    std::cerr << std::endl;
}

void ParserLogger::Report() {
    if (cur_level_ < level_) return;
    double tagging_time= std::chrono::duration_cast<std::chrono::seconds>(
            times_["time_end_of_tagging"]-times_["time_start_running"]).count();
    double parsing_time= std::chrono::duration_cast<std::chrono::seconds>(
            times_["time_end_of_parsing"]-times_["time_end_of_tagging"]).count();
    double total = tagging_time + parsing_time;

    std::cerr << std::endl
              << "tagging time: " << tagging_time << " seconds" << std::endl
              << "parsing time: " << parsing_time << " seconds" << std::endl
              << "total elapsed time: " << total << " seconds" << std::endl;
    ShowStatistics();
}

} // namespace myccg
