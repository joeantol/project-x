#!/usr/bin/python

get_ipython().__class__.__name__

try:
    get_ipython().__class__.__name__
    print('In Jupyter')
except:
    print('NOT in Jupyter')

