from importlib.resources import path
import os
import sys
sys.path.append('../src')
import csv
from supportfunctions import *
sys.stdout.flush()
import petsc4py
#petsc4py.init(sys.argv)
from petsc4py import PETSc
import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime
import argparse
import numpy as np


from main import tol
from main import epsilon
from main import fraction
from main import max_iter
from main import path_name
from main import test_code



open(path_name+"gamma_0.0_eta_0.1_test"+'.csv','w+')