'''
SCRIPT: CleaningKineticMechanisms
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

# Define path to PyTools classes
import sys
sys.path.insert(0, '../src')

# Import PyTools classes
from KineticMechanism import KineticMechanism
from OpenSMOKEppXMLFile import OpenSMOKEppXMLFile
from PolimiSootModule import *
from Utilities import *


# Read list of species
listofspecies_file = open('listofspecies', 'r', encoding='utf-8')
listofspecies = (listofspecies_file.read()).split()
ns = len(listofspecies)


# --------------------------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------------------------
tran_file = open('transport.uncleaned', 'r', encoding='utf-8')
tran_data = tran_file.readlines()
tran_lines = len(tran_data)

data = []
found = [False]*ns
for i in range(ns):
    species = listofspecies[i]
    for j in range(tran_lines):
        if (found[i] == False):
            words = tran_data[j].split()
            if (len(words) != 0):
                if (words[0] == species):
                    found[i] = True
                    single_data = { 'name': species, 'coeffs': words[1:7] }
                    data.append(single_data)

f = open('transport.cleaned', 'w')
for i in range(ns):
    f.write('%-30s %-16s %-16s %-16s %-16s %-16s %-16s\n' % (data[i]['name'], data[i]['coeffs'][0],             data[i]['coeffs'][1],data[i]['coeffs'][2],data[i]['coeffs'][3],data[i]['coeffs'][4],data[i]['coeffs'][5]) )
f.close()


# --------------------------------------------------------------------------------------------
# Thermodynamics
# --------------------------------------------------------------------------------------------

thermo_file = open('thermo.uncleaned', 'r', encoding='utf-8')
thermo_data = thermo_file.readlines()
thermo_lines = len(thermo_data)

data = []
found = [False]*ns
for i in range(ns):
    species = listofspecies[i]
    for j in range(thermo_lines):
        if (found[i] == False):
            words = thermo_data[j].split()
            if (len(words) != 0):
                if (words[0] == species):
                    found[i] = True
                    if (thermo_data[j][80] == '&'):
                        nlines = 5;
                        first_line = thermo_data[j][0:81]
                        second_line = thermo_data[j+1].strip()
                        last_line = thermo_data[j+4][0:80]
                    else:
                        nlines = 4
                        first_line = thermo_data[j][0:80]
                        second_line = thermo_data[j+1][0:80]
                        last_line = ''
                    
                    single_data = { 'name': species, 'nlines': nlines,                                     'line1': first_line+'\n',                                       'line2': second_line+'\n',                                     'line3': thermo_data[j+2][0:80]+'\n',                                     'line4': thermo_data[j+3][0:80]+'\n',                                     'line5': last_line+'\n' }
                    
                    data.append(single_data)

f = open('thermo.cleaned', 'w')
for i in range(ns):
    f.write(data[i]['line1'])
    f.write(data[i]['line2'])
    f.write(data[i]['line3'])
    f.write(data[i]['line4'])
    if (data[i]['nlines'] == 5):
        f.write(data[i]['line5'])
f.close()
