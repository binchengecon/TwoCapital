import csv
import numpy as np
import os
import itertools


Paramter_Matrix  = np.zeros((3,2))
Paramter_Header  = ['Paramter','Parameter']

LinearSolver_List = ['petsc', 'petsc4py', 'eigen']
LinearSolver_Header = ['LinearSolverMethod']

ROOT_DIR = os.path.abspath(os.curdir)
# write a list into csv file
filename = "ListWriting"
with open("../data/PostJump/" + filename + '.csv', 'w+',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(LinearSolver_Header)
    writer.writerows(LinearSolver_List)

# Write a number array into csv file

filename2 = "MatrixWriting"

with open(filename2 + '.csv', 'w+', newline='') as f:

    writer = csv.writer(f)
    writer.writerow(Paramter_Header)
    writer.writerows(Paramter_Matrix)



# stack array

col1 = range(6)
col2 = range(6)

data = list(zip(col1,col2,LinearSolver_List))

# 

filename3 = "CombineWriting"

with open(filename3 + '.csv', 'w+', newline='') as f:

    writer = csv.writer(f)
    writer.writerow(Paramter_Header)
    writer.writerows(data)

