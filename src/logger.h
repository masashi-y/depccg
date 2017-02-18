
#ifndef INCLUDE_LOGGER_H_
#define INCLUDE_LOGGER_H_

#include <chrono>
#include <string>
#include "tree.h"
#include "chainer_tagger.h"
#include "parser_tools.h"

#define GREENCOLOR "\033[32m"
#define BLACKCOLOR "\033[93m"
#define REDCOLOR "\033[31m"
#define BLUECOLOR "\033[34m"
#define CYANCOLOR "\033[36m"

namespace myccg {

template<typename T>
struct Color
{
    Color(T& ob, const char* color)
        :ob_(ob), color_(color) {}

    T ob_;
    const char* color_;
};

template<typename T> Color<T> Blue(T ob) { return Color<T>(ob, BLUECOLOR); }
template<typename T> Color<T> Red(T ob) { return Color<T>(ob, REDCOLOR); }
template<typename T> Color<T> Green(T ob) { return Color<T>(ob, GREENCOLOR); }
template<typename T> Color<T> Cyan(T ob) { return Color<T>(ob, CYANCOLOR); }

template<typename T>
std::ostream& operator<<(std::ostream& out, const Color<T>& color) {
    return out << color.color_ << color.ob_ << "\033[0m";
}

enum LogLevel { Debug, Info, Warn, Error };

class ParserLogger
{
public:
    ParserLogger(LogLevel level);
    ~ParserLogger();

    void InitStatistics(int num_sents);
    void ShowStatistics();
    void RecordTime(const char* name);
    void RecordTime(const std::string& name);
    void RecordTimeStartRunning();
    void RecordTimeEndOfTagging();
    void RecordTimeEndOfParsing();

    void RecordTree(const char* message, NodeType tree);

    void RecordAgendaItem(const char* message, const AgendaItem& item);

    void Report();

    void CompleteOne();
    void CompleteOne(int id, int agenda_size);

    void CalculateNumOneBestTags(
            int id, Tagger* tagger, float* probs, NodeType res);

    void ShowTaggingOneBest(
            Tagger* tagger, float* probs, std::vector<std::string>& tokens);

    void ShowDependencyOneBest(
        float* probs, std::vector<std::string>& tokens);

    template<typename T> ParserLogger& operator<<(const T& message) {
        std::cerr << "[LOG] " << message << std::endl;
        return *this;
    }

    template<typename T>
    ParserLogger& operator<<(const Color<T>& color) {
        std::cerr << "[LOG] " << color.color_ << color.ob_ << "\033[0m" << std::endl;
        return *this;
    }

    typedef std::ostream& (*stream_function)(std::ostream&);
    ParserLogger& operator<<(stream_function func) {
        func(std::cout);
        return *this;
    }

private:
    LogLevel level_;
    int nprocessed_;
    std::unordered_map<std::string,
        std::chrono::system_clock::time_point> times_;

    bool do_statistics_;
    int num_sents_;
    int* agenda_sizes_;
    double* num_one_best_tags_;
};

} // namespace myccg

#endif
