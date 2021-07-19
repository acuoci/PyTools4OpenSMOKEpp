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
from PolimiSootModule import *
from Utilities import *



# Official CRECK2012 Soot BINJOnly (no heavy fuels version)
folder_github = "C:\\Users\\acuoci\\OneDrive - Politecnico di Milano\\My Projects\\GitHub\\"
kin_xml_folder_name=folder_github + "CRECK_DiscreteSectionalModel_v2012\\CRECK_2012_Soot_OnlyBINJ_NoHeavyFuels\\kinetics-CRECK_2012_SootOnlyBINJ-SP-AGG\\"
kinetics = KineticMechanism(kin_xml_folder_name + "kinetics.xml")
kinetics.ReadKinetics(kin_xml_folder_name + "reaction_names.xml")

# Create groups
kinetics.AddGroupOfSpecies('PAH12', 'PAHs with 1/2 aromatic rings', DefaultPAH12())
kinetics.AddGroupOfSpecies('PAH34', 'PAHs with 3/4 aromatic rings', DefaultPAH34())
kinetics.AddGroupOfSpecies('PAHLP', 'PAHs with more than 4 aromatic rings (molecular and radical)', DefaultPAHLP(kinetics.species))
kinetics.AddGroupOfSpecies('SP', 'BIN sections corresponding to spherical particles (molecular and radical)', DefaultSP(kinetics.species))
kinetics.AddGroupOfSpecies('AGG', 'BIN sections corresponding to aggregates (molecular and radical)', DefaultAGG(kinetics.species))

# Select species to be kept
species_to_be_kept = [   'BIN21AJ','BIN21BJ','BIN21CJ', 'BIN22AJ','BIN22BJ','BIN22CJ', 'BIN23AJ','BIN23BJ','BIN23CJ', \
                         'BIN24AJ','BIN24BJ','BIN24CJ', 'BIN25AJ','BIN25BJ','BIN25CJ' ]
reactions = kinetics.ReactionsWithMultipleSpecies(species_to_be_kept, ['RP']*len(species_to_be_kept), 'OR')
species = kinetics.SpeciesInSelectedReactions(reactions)
kinetics.PrintKineticMechanism("Kinetics.WithBIN21-25.CKI", species, reactions)

