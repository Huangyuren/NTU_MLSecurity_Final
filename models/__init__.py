from models.robnet import *


def model_entry(config, genotype_list):
    return globals()['robnet'](genotype_list, **config.model_param)

def model_search_entry(config):
    return globals()['robnet_search'](**config.model_param)
