
#ifndef INCLUDE_FEAT_H_
#define INCLUDE_FEAT_H_

#include <iostream>
#include "cacheable.h"


namespace myccg {
namespace cat {

class Feature;
typedef const Feature* Feat;

#ifdef JAPANESE
class Feature: public Cacheable
{
public:
    static Feat Parse(const std::string& string);

    ~Feature() {}

    std::string ToStr() const;
    bool IsEmpty() const { return values_.empty(); }
    bool Matches(const Feature* other) const;
    bool ContainsWildcard() const { return contains_wildcard_; }
    std::string SubstituteWildcard(const std::string& string) const;

private:
    Feature(const std::string& value);
private:
    std::vector<std::pair<std::string, std::string>> values_;
    bool contains_wildcard_;
};

#else
class Feature: public Cacheable
{
public:
    static Feat Parse(const std::string& string);

    ~Feature() {}

    std::string ToStr() const { return IsEmpty() ? "" : "[" + value_ + "]"; }
    bool IsEmpty() const { return value_.empty(); }
    bool Matches(const Feature* other) const {
        return (GetId() == other->GetId() ||
                this->ContainsWildcard() ||
                other->ContainsWildcard());
    }
    bool ContainsWildcard() const { return value_ == "X"; }

    std::string SubstituteWildcard(const std::string& string) const;
private:
    Feature(const std::string& value): value_(value) {}
private:
    std::string value_;
};
#endif

} // namespace cat
} // namespace myccg

#endif
