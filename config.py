import collections
import os
import json

class MetaCfg(type): # since __repr__,__str__ are defines for instances we need metaclass for class Cfg

    def __repr__(cls):
        return f'{cls.__name__}=collections.OrderedDict([' \
            f'{",".join([f"({repr(n)},{repr(v)})" for (n,v) in cls.cfg_dict().items()])}' \
            f'])'

    def __str__(cls):
        s = "\n\t".join([f"{n}={repr(v)}" for (n,v) in cls.cfg_dict().items()])
        return f'{cls.__name__}:\n\t{s}'


class Cfg(metaclass=MetaCfg): # user configurations will be derived from this class

    @classmethod
    def cfg_dict(cls):
        return collections.OrderedDict([(name,val)
                for name,val in cls.__dict__.items()
                if not name.startswith('__') and not callable(getattr(cls, name))])

    @classmethod
    def save_json(cls, fname_path):
        folder = os.path.split(fname_path)[0]
        assert   os.path.isdir(folder) | (not folder), f'folder {folder} does not exist'
        with open(fname_path,'w') as file:
            json.dump(cls.cfg_dict(),file,indent=4)
        print(f'configuration {cls.__name__} saved to file {fname_path}')

    @classmethod
    def load_json(cls,fname_path):
        assert os.path.exists(fname_path),f'file {fname_path} does not exist'
        with open(fname_path,'r') as file:
            cfg = json.load(file)
        for name,value in cfg.items(): # change value of loaded variables only
            setattr(cls,name,value)

if __name__ == '__main__':
    class TestCfg(Cfg):
        f1 = True
        f2 = 'val2'
        f3 = 125
        f4 = 'val3'

    print(TestCfg)
    print(repr(TestCfg))

    TestCfg.save_json('../tmp/new.json')

    print(f'old value:',TestCfg.f1)
    TestCfg.f1 = False
    print(f'new value:',TestCfg.f1)

    TestCfg.load_json('../tmp/new.json')
    print(f'when loaded:',TestCfg.f1)

