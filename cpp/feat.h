
#ifndef INCLUDE_FEAT_H_
#define INCLUDE_FEAT_H_

#include <vector>
#include "cacheable.h"
#include "debug.h"


namespace myccg {

class Feature;
typedef const Feature* Feat;

class Feature: public Cacheable<Feature>
{
public:
    static Feat Parse(const std::string& string);

    virtual ~Feature() {}
    virtual std::string ToStr() const = 0;
    virtual bool IsEmpty() const = 0;
    virtual bool Matches(Feat other) const = 0;
    virtual bool ContainsWildcard() const = 0;
    virtual std::string SubstituteWildcard(const std::string& string) const = 0;
    virtual bool ContainsKeyValue(const std::string& key, const std::string& value) const = 0;
    virtual Feat ToMultiValue() const = 0;
    virtual std::unordered_map<std::string, std::string> Values() const = 0;
    Feature() {}
};

class MultiValueFeature: public Feature
{
public:
    MultiValueFeature(const std::string& value);
    ~MultiValueFeature() {}
    std::string ToStr() const;
    bool IsEmpty() const { return values_.empty(); }
    bool Matches(Feat other) const;
    bool ContainsWildcard() const { return contains_wildcard_; }
    std::string SubstituteWildcard(const std::string& string) const;
    bool ContainsKeyValue(const std::string& key, const std::string& value) const;
    Feat ToMultiValue() const { return this; }
    std::unordered_map<std::string, std::string> Values() const {
        std::unordered_map<std::string, std::string> res;
        for(auto&& key_val : values_)
            res[key_val.first] = key_val.second;
        return res;                
    }

private:
    std::vector<std::pair<std::string, std::string>> values_;
    bool contains_wildcard_;
};

class SingleValueFeature: public Feature
{
public:
    SingleValueFeature(const std::string& value): value_(value) {}
    ~SingleValueFeature() {}

    std::string ToStr() const { return IsEmpty() ? "" : "[" + value_ + "]"; }
    bool IsEmpty() const { return value_.empty(); }
    bool Matches(Feat other) const;
    bool ContainsWildcard() const { return value_ == "X"; }
    std::string SubstituteWildcard(const std::string& string) const;
    bool ContainsKeyValue(const std::string& key, const std::string& value) const NO_IMPLEMENTATION
    Feat ToMultiValue() const;
    std::unordered_map<std::string, std::string> Values() const {
        std::unordered_map<std::string, std::string> res;
        if (value_ != "")
            res[value_] = "true";
        return res;
    }

private:
    std::string value_;
};

} // namespace myccg

#endif
