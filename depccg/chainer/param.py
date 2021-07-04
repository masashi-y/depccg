
import json


class Param:
    def __init__(self, obj):
        self.__dict__["ps"] = {"model": obj.__class__.__name__}
        self.__dict__["obj"] = obj

    def __setattr__(self, key, value):
        self.__dict__["ps"][key] = value
        self.__dict__["obj"].__dict__[key] = value

    def dump(self, out):
        json.dump(self.ps, open(out, "w"))

    @staticmethod
    def load(obj, paramfile):
        p = Param(obj)
        params = json.load(open(paramfile))
        p.__dict__["ps"].update(params)
        obj.__dict__.update(params)
        return p
