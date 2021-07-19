'''
SCRIPT: ReadStoichiometry
@Authors:
    Alberto Cuoci [1]
    [1]: CRECK Modeling Lab, Department of Chemistry, Materials, and Chemical Engineering, Politecnico di Milano
@Contacts:
    alberto.cuoci@polimi.it
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.cuoci@polimi.it
'''

# Import main libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import xml.etree.ElementTree as ET
from scipy.sparse import csr_matrix, find
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
plt.style.use('default')

# Define path to PyTools classes
import sys
sys.path.insert(0, '../src')

# Import PyTools classes
from KineticMechanism import KineticMechanism
from OpenSMOKEppXMLFile import OpenSMOKEppXMLFile
from PolimiSootModule import *
from Utilities import *


# --------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------- 

def ReadVectorInt(vector, pos):
    length = int(vector[pos])
    subvector = vector[pos+1:pos+length+1]
    return np.int32(subvector), pos+length+1

def ReadVectorFloat64(vector, pos):
    length = int(vector[pos])
    subvector = vector[pos+1:pos+length+1]
    return np.float64(subvector), pos+length+1
    

# --------------------------------------------------------------------------------------
# Read kinetic mechanism
# -------------------------------------------------------------------------------------- 

kin_xml_folder_name="C:\\Users\\acuoci\\Aachen-Data\\Kinetics\\GRI30\\kinetics-GRI30\\"
kinetics = KineticMechanism(kin_xml_folder_name + "kinetics.xml")
kinetics.ReadKinetics(kin_xml_folder_name + "reaction_names.xml")


# --------------------------------------------------------------------------------------
# Import stoichiometry
# -------------------------------------------------------------------------------------- 

tree = ET.parse(kin_xml_folder_name + "kinetics.xml")
root = tree.getroot()
kinetics_element = root.find('Kinetics')
stoichiometry_element = kinetics_element.find('Stoichiometry')
stoichiometry = (stoichiometry_element.text).split()


# --------------------------------------------------------------------------------------
# Process stoichiometric data
# -------------------------------------------------------------------------------------- 

pos = 0
numDir1, pos = ReadVectorInt(stoichiometry, pos)
numDir2, pos = ReadVectorInt(stoichiometry, pos)
numDir3, pos = ReadVectorInt(stoichiometry, pos)
numDir4, pos = ReadVectorInt(stoichiometry, pos)
numDir5, pos = ReadVectorInt(stoichiometry, pos)

numRevTot1, pos = ReadVectorInt(stoichiometry, pos)
numRevTot2, pos = ReadVectorInt(stoichiometry, pos)
numRevTot3, pos = ReadVectorInt(stoichiometry, pos)
numRevTot4, pos = ReadVectorInt(stoichiometry, pos)
numRevTot5, pos = ReadVectorInt(stoichiometry, pos)

numRevEq1, pos = ReadVectorInt(stoichiometry, pos)
numRevEq2, pos = ReadVectorInt(stoichiometry, pos)
numRevEq3, pos = ReadVectorInt(stoichiometry, pos)
numRevEq4, pos = ReadVectorInt(stoichiometry, pos)
numRevEq5, pos = ReadVectorInt(stoichiometry, pos)

jDir1, pos = ReadVectorInt(stoichiometry, pos)
jDir2, pos = ReadVectorInt(stoichiometry, pos)
jDir3, pos = ReadVectorInt(stoichiometry, pos)
jDir4, pos = ReadVectorInt(stoichiometry, pos)
jDir5, pos = ReadVectorInt(stoichiometry, pos)
valueDir5, pos = ReadVectorFloat64(stoichiometry, pos)

jDir1 = jDir1-1
jDir2 = jDir2-1
jDir3 = jDir3-1
jDir4 = jDir4-1
jDir5 = jDir5-1

jRevTot1, pos = ReadVectorInt(stoichiometry, pos)
jRevTot2, pos = ReadVectorInt(stoichiometry, pos)
jRevTot3, pos = ReadVectorInt(stoichiometry, pos)
jRevTot4, pos = ReadVectorInt(stoichiometry, pos)
jRevTot5, pos = ReadVectorInt(stoichiometry, pos)
valueRevTot5, pos = ReadVectorFloat64(stoichiometry, pos)

jRevTot1 = jRevTot1-1
jRevTot2 = jRevTot2-1
jRevTot3 = jRevTot3-1
jRevTot4 = jRevTot4-1
jRevTot5 = jRevTot5-1

jRevEq1, pos = ReadVectorInt(stoichiometry, pos)
jRevEq2, pos = ReadVectorInt(stoichiometry, pos)
jRevEq3, pos = ReadVectorInt(stoichiometry, pos)
jRevEq4, pos = ReadVectorInt(stoichiometry, pos)
jRevEq5, pos = ReadVectorInt(stoichiometry, pos)
valueRevEq5, pos = ReadVectorFloat64(stoichiometry, pos)

jRevEq1 = jRevEq1-1
jRevEq2 = jRevEq2-1
jRevEq3 = jRevEq3-1
jRevEq4 = jRevEq4-1
jRevEq5 = jRevEq5-1

changeOfMoles, pos = ReadVectorFloat64(stoichiometry, pos)
explicit_reaction_orders = int(stoichiometry[pos])

# Elementary reactions only
if (explicit_reaction_orders == 0):
    
    lambda_numDir1 = numDir1;
    lambda_numDir2 = numDir2;
    lambda_numDir3 = numDir3;
    lambda_numDir4 = numDir4;
    lambda_numDir5 = numDir5;

    lambda_numRevEq1 = numRevEq1;
    lambda_numRevEq2 = numRevEq2;
    lambda_numRevEq3 = numRevEq3;
    lambda_numRevEq4 = numRevEq4;
    lambda_numRevEq5 = numRevEq5;

    lambda_jDir1 = jDir1;
    lambda_jDir2 = jDir2;
    lambda_jDir3 = jDir3;
    lambda_jDir4 = jDir4;
    lambda_jDir5 = jDir5;
    lambda_valueDir5 = valueDir5;

    lambda_jRevEq1 = jRevEq1;
    lambda_jRevEq2 = jRevEq2;
    lambda_jRevEq3 = jRevEq3;
    lambda_jRevEq4 = jRevEq4;
    lambda_jRevEq5 = jRevEq5;
    lambda_valueRevEq5 = valueRevEq5;
    
else:

    sys.exit("Non-elementary reactions cannot be processed")


# --------------------------------------------------------------------------------------
# Reactant side analysis (stoichiometric coefficients)
# --------------------------------------------------------------------------------------   

react_species = []
react_reaction = []
react_nu = []

count1=0
count2=0
count3=0
count4=0
count5=0
for i in range(kinetics.ns):

    for k in range(numDir1[i]):
        react_species.append(i)
        react_reaction.append(jDir1[count1])
        react_nu.append(1.)
        count1 = count1+1
        
    for k in range(numDir2[i]):
        react_species.append(i)
        react_reaction.append(jDir2[count2])
        react_nu.append(2.)
        count2 = count2+1
        
    for k in range(numDir3[i]):
        react_species.append(i)
        react_reaction.append(jDir3[count3])
        react_nu.append(3.)
        count3 = count3+1
        
    for k in range(numDir4[i]):
        react_species.append(i)
        react_reaction.append(jDir4[count4])
        react_nu.append(0.5)
        count4 = count4+1
        
    for k in range(numDir5[i]):
        react_species.append(i)
        react_reaction.append(jDir5[count5])
        react_nu.append(valueDir5[count5])
        count5 = count5+1

nur = sparse.coo_matrix((react_nu,(react_reaction,react_species)),shape=(kinetics.nr,kinetics.ns))


# --------------------------------------------------------------------------------------
# Product side analysis (stoichiometric coefficients)
# --------------------------------------------------------------------------------------   

react_species = []
react_reaction = []
react_nu = []

count1=0
count2=0
count3=0
count4=0
count5=0
for i in range(kinetics.ns):

    for k in range(numRevTot1[i]):
        react_species.append(i)
        react_reaction.append(jRevTot1[count1])
        react_nu.append(1.)
        count1 = count1+1
        
    for k in range(numRevTot2[i]):
        react_species.append(i)
        react_reaction.append(jRevTot2[count2])
        react_nu.append(2.)
        count2 = count2+1
        
    for k in range(numRevTot3[i]):
        react_species.append(i)
        react_reaction.append(jRevTot3[count3])
        react_nu.append(3.)
        count3 = count3+1
        
    for k in range(numRevTot4[i]):
        react_species.append(i)
        react_reaction.append(jRevTot4[count4])
        react_nu.append(0.5)
        count4 = count4+1
        
    for k in range(numRevTot5[i]):
        react_species.append(i)
        react_reaction.append(jRevTot5[count5])
        react_nu.append(valueRevTot5[count5])
        count5 = count5+1

nup = sparse.coo_matrix((react_nu,(react_reaction,react_species)),shape=(kinetics.nr,kinetics.ns))


# --------------------------------------------------------------------------------------
# Reactant side analysis (reaction orders)
# --------------------------------------------------------------------------------------   

react_species = []
react_reaction = []
react_lambda = []

count1=0
count2=0
count3=0
count4=0
count5=0
for i in range(kinetics.ns):

    for k in range(lambda_numDir1[i]):
        react_species.append(i)
        react_reaction.append(lambda_jDir1[count1])
        react_lambda.append(1.)
        count1 = count1+1
        
    for k in range(lambda_numDir2[i]):
        react_species.append(i)
        react_reaction.append(lambda_jDir2[count2])
        react_lambda.append(2.)
        count2 = count2+1
        
    for k in range(lambda_numDir3[i]):
        react_species.append(i)
        react_reaction.append(lambda_jDir3[count3])
        react_lambda.append(3.)
        count3 = count3+1
        
    for k in range(lambda_numDir4[i]):
        react_species.append(i)
        react_reaction.append(lambda_jDir4[count4])
        react_lambda.append(0.5)
        count4 = count4+1
        
    for k in range(lambda_numDir5[i]):
        react_species.append(i)
        react_reaction.append(lambda_jDir5[count5])
        react_lambda.append(lambda_valueDir5[count5])
        count5 = count5+1

lambdar = sparse.coo_matrix((react_lambda,(react_reaction,react_species)),shape=(kinetics.nr,kinetics.ns))


# --------------------------------------------------------------------------------------
# Product side analysis (reaction orders)
# --------------------------------------------------------------------------------------   

react_species = []
react_reaction = []
react_lambda = []

count1=0
count2=0
count3=0
count4=0
count5=0
for i in range(kinetics.ns):

    for k in range(lambda_numRevEq1[i]):
        react_species.append(i)
        react_reaction.append(lambda_jRevEq1[count1])
        react_lambda.append(1.)
        count1 = count1+1
        
    for k in range(lambda_numRevEq2[i]):
        react_species.append(i)
        react_reaction.append(lambda_jRevEq2[count2])
        react_lambda.append(2.)
        count2 = count2+1
        
    for k in range(lambda_numRevEq3[i]):
        react_species.append(i)
        react_reaction.append(lambda_jRevEq3[count3])
        react_lambda.append(3.)
        count3 = count3+1
        
    for k in range(lambda_numRevEq4[i]):
        react_species.append(i)
        react_reaction.append(lambda_jRevEq4[count4])
        react_lambda.append(0.5)
        count4 = count4+1
        
    for k in range(lambda_numRevEq5[i]):
        react_species.append(i)
        react_reaction.append(lambda_jRevEq5[count5])
        react_lambda.append(lambda_valueRevEq5[count5])
        count5 = count5+1

lambdap = sparse.coo_matrix((react_lambda,(react_reaction,react_species)),shape=(kinetics.nr,kinetics.ns))



# Final checks (to be removed)
diff_nur = nur - kinetics.nur
print(diff_nur)
diff_nup = nup - kinetics.nup
print(diff_nup)
diff_lambdar = lambdar - kinetics.nur
print(diff_lambdar)
#diff_lambdap = lambdap - kinetics.nup
#print(diff_lambdap)


# Final output (to be removed)
#for j in range(kinetics.nr):
#    indices1 = sparse.find(kinetics.nup.getrow(j))[1]
#    indices2 = sparse.find(nup.getrow(j))[1]
#    print(indices1, indices2)

