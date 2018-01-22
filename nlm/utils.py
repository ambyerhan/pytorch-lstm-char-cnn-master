# Proj : attNMT by AmbyerHan
# ambyer
# 2017.11.01
# this file contains functions that used in the self-attention NMT

import json
import six

from datetime import datetime


def load_json(filename):
    jdic = {}
    try:
        with open(filename, 'rb') as fjson:
            jdic = json.load(fjson)
    except Exception as e:
        print('The error message : {0}'.format(str(e)))
        exit()

    return jdic

def dump_json(filename, jdic, ind = 2, en_as = False):
    json.dump(jdic, filename, indent = ind, ensure_ascii = en_as)

def disp_hparam(args):
    date = datetime.now().strftime("%Y.%m.%d - %H:%M:%S")
    print("|------------------------------------------------------|")
    print("|                    Configuration                     |")
    print("|------------------------------------------------------|")
    print("|--%-13s = %-34s--|" % ('DATE AND TIME', date))
    tmp = sorted(six.iteritems(args), key = lambda d: d[0], reverse = False)
    for key, val in tmp:
        print("|--%-13s = %-34s--|" % (key, val))
    print("|------------------------------------------------------|")
    print("|                 Configuration  End                   |")
    print("|------------------------------------------------------|")
