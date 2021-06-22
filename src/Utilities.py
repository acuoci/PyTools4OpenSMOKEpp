'''
MODULE: Utilities
@Authors:
    Alberto Cuoci [1]
    [1]: CRECK Modeling Lab, Department of Chemistry, Materials, and Chemical Engineering, Politecnico di Milano
@Contacts:
    alberto.cuoci@polimi.it
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.cuoci@polimi.it
'''

import numpy as np

# Function: check for the existence of multiple elements
def CheckForMultipleElements(v):
    for i in range(len(v)):
        count = v.count(v[i])
        if (count > 1): print(v[i], " appearing ", count)

            
# Function: check for the existence of same elements in 2 different vectors
def CheckForCrossingValues(v1,v2):
    for i in range(len(v1)):
        for j in range(len(v2)):
            if (v1[i] == v2[j]): print(v1[i])   

