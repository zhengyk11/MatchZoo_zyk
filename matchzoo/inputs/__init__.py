# note 
import six
from keras.utils.generic_utils import deserialize_keras_object

from PairGenerator import PairGenerator
from DSSM_PairGenerator import DSSM_PairGenerator
from DRMM_PairGenerator import DRMM_PairGenerator
from Duet_PairGenerator import Duet_PairGenerator

from ListGenerator import ListGenerator
from DSSM_ListGenerator import DSSM_ListGenerator
from DRMM_ListGenerator import DRMM_ListGenerator
from Duet_ListGenerator import Duet_ListGenerator


def serialize(generator):
    return generator.__name__

def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')

def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)

