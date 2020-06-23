from .basic_model import Network
from .search_model import Network_search


def robnet(genotype_list, **kwargs):
    return Network(genotype_list=genotype_list, **kwargs)

def robnet_search(**kwargs):
    return Network_search(**kwargs)