import six
import pprint


class AttrDict(object):
    '''
    fuck attrdict
    '''

    def __init__(self, d=None):
        self.__dict__ = d if d is not None else dict()

        for k, v in list(six.iteritems(self.__dict__)):
            if isinstance(v, dict):
                self.__dict__[k] = AttrDict(v)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def __iter__(self):
        for k, v in six.iteritems(self.__dict__):
            yield k, v