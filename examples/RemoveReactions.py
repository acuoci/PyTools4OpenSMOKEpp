'''
SCRIPT: RemoveReactions
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
from scipy.sparse import csr_matrix, find
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
plt.style.use('default')

# Define path to PyTools classes
import sys
sys.path.insert(0, '../src')

# Import PyTools classes
from KineticMechanism import KineticMechanism
from OpenSMOKEppXMLFile import OpenSMOKEppXMLFile
from Utilities import *



# Official CRECK2012 Soot BINJOnly (no heavy fuels version)
folder_github = "C:\\Users\\acuoci\\OneDrive - Politecnico di Milano\\My Projects\\GitHub\\"
kin_xml_folder_name=folder_github + "CRECK_DiscreteSectionalModel_v2012\\CRECK_2012_Soot_OnlyBINJ_NoHeavyFuels\\kinetics-CRECK_2012_SootOnlyBINJ-SP-AGG\\"
kinetics = KineticMechanism(kin_xml_folder_name + "kinetics.xml")
kinetics.ReadKinetics(kin_xml_folder_name + "reaction_names.xml")


# Remove reactions containing a subset of species
species_to_be_removed = [   'BIN21AJ','BIN21BJ','BIN21CJ', 'BIN22AJ','BIN22BJ','BIN22CJ', 'BIN23AJ','BIN23BJ','BIN23CJ', \
                            'BIN24AJ','BIN24BJ','BIN24CJ', 'BIN25AJ','BIN25BJ','BIN25CJ' ]
reactions = kinetics.ReactionsWithoutMultipleSpecies(species_to_be_removed, ['RP']*len(species_to_be_removed), 'OR')
species = kinetics.SpeciesInSelectedReactions(reactions)
kinetics.PrintKineticMechanism("Kinetics.WithoutBIN21-25.CKI", species, reactions)

