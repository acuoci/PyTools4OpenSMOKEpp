'''
SCRIPT: ManipulatingSootKinetics
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


# Hierarchical partitioning reactions according to classification in terms of PAHs and soot particles/aggregates
list_of_pahs = kinetics.Group('PAH12')['list'] + kinetics.Group('PAH34')['list'] + kinetics.Group('PAHLP')['list']
reactions_pahs_all, reactions_pahs_pah12, reactions_pahs_pah34, reactions_pahs_pahlp, reactions_pahs_sp, reactions_pahs_agg = kinetics.PartitionSootPrecursorsReactions(list_of_pahs)

list_of_pahs = kinetics.Group('PAH12')['list'] + kinetics.Group('PAH34')['list'] + kinetics.Group('PAHLP')['list']
reactions_pahs_all, reactions_pahs_pah12, reactions_pahs_pah34, reactions_pahs_pahlp, reactions_pahs_sp, reactions_pahs_agg = kinetics.PartitionSootPrecursorsReactions(list_of_pahs)

# Partitioning reactions according to classification in terms of PAHs and soot particles/aggregates
reactions_all_sp = kinetics.ReactionsWithMultipleSpecies(kinetics.Group('SP')['list'], ["RP"]*len(kinetics.Group('SP')['list']), 'OR')
reactions_all_agg = kinetics.ReactionsWithMultipleSpecies(kinetics.Group('AGG')['list'], ["RP"]*len(kinetics.Group('AGG')['list']), 'OR')
reactions_all_sp_agg = (np.unique(reactions_all_sp + reactions_all_agg)).tolist()
reactions_minuspahs_sp_agg = list(set(reactions_all_sp_agg) - set(reactions_pahs_sp+reactions_pahs_agg))
reactions_only_sp, reactions_sp_agg, reactions_only_agg = kinetics.SplitSphericalAggregatesReactions(reactions_minuspahs_sp_agg)
reactions_only_gas = list(set(range(kinetics.nr)) - set(reactions_pahs_all) - set(reactions_minuspahs_sp_agg))


# Check 
CheckForCrossingValues(reactions_pahs_pah34, reactions_pahs_pahlp)
CheckForCrossingValues(reactions_only_gas, reactions_pahs_all)
CheckForCrossingValues(reactions_only_gas, reactions_minuspahs_sp_agg)
CheckForCrossingValues(reactions_pahs_all, reactions_minuspahs_sp_agg)

# Summary
print('Reactions ALL with SP:', len(reactions_all_sp))
print('Reactions ALL with AGG:', len(reactions_all_agg))
print('Reactions ALL with SP-AGG:', len(reactions_all_sp_agg))
print('Reactions MINUS-PAHS with SP and AGG:', len(reactions_minuspahs_sp_agg))
print('Reactions MINUS-PAHS with SP only:', len(reactions_only_sp))
print('Reactions MINUS-PAHS with SP/AGG:', len(reactions_sp_agg))
print('Reactions MINUS-PAHS with AGG only:', len(reactions_only_agg))
print('')

# Summary
print('Reactions ALL:', kinetics.nr)
print('Reactions GAS ONLY:', len(reactions_only_gas))
print('Reactions PAH PAH12:', len(reactions_pahs_pah12))
print('Reactions PAH PAH34:', len(reactions_pahs_pah34))
print('Reactions PAH PAHLP:', len(reactions_pahs_pahlp))
print('Reactions PAH SP:', len(reactions_pahs_sp))
print('Reactions PAH AGG:', len(reactions_pahs_agg))
print('Reactions SP:', len(reactions_only_sp))
print('Reactions SP/AGG:', len(reactions_sp_agg))
print('Reactions AGG:', len(reactions_only_agg))
print('Sum: ', len(reactions_only_gas)+len(reactions_pahs_pah12)+len(reactions_pahs_pah34)+len(reactions_pahs_pahlp)+len(reactions_pahs_sp)+len(reactions_pahs_agg)+len(reactions_only_sp)+len(reactions_sp_agg)+len(reactions_only_agg))
print('Sum: ', len(reactions_only_gas)+len(reactions_pahs_all)+len(reactions_only_sp)+len(reactions_sp_agg)+len(reactions_only_agg))
print('')


# Print partial versions of the kinetic mechanisms
# !!! Be careful: only the reactions in which the selected species are included
# !!! These mechanisms are not intended for adoption in simulations but for analysis purposes only

# Print mechanism with gas species only [gas]
species = kinetics.SpeciesInSelectedReactions(reactions_only_gas)
kinetics.PrintKineticMechanism("SubKinetics.01.GasOnly.CKI", species, reactions_only_gas)

# Print mechanism with PAHs12 [PAH12]
species = kinetics.SpeciesInSelectedReactions(reactions_pahs_pah12)
kinetics.PrintKineticMechanism("SubKinetics.02.PAH12.CKI", species, reactions_pahs_pah12)

# Print mechanism with PAHs34 [PAH34]
species = kinetics.SpeciesInSelectedReactions(reactions_pahs_pah34)
kinetics.PrintKineticMechanism("SubKinetics.03.PAH34.CKI", species, reactions_pahs_pah34)

# Print mechanism with PAHsLP [PAHLP]
species = kinetics.SpeciesInSelectedReactions(reactions_pahs_pahlp)
kinetics.PrintKineticMechanism("SubKinetics.04.PAHLP.CKI", species, reactions_pahs_pahlp)

# Print mechanism with spherical particles [SP]
species = kinetics.SpeciesInSelectedReactions(reactions_pahs_sp)
kinetics.PrintKineticMechanism("SubKinetics.05.SP.CKI", species, reactions_pahs_sp)

# Print mechanism with aggregates [AGG]
species = kinetics.SpeciesInSelectedReactions(reactions_pahs_agg)
kinetics.PrintKineticMechanism("SubKinetics.06.AGG.CKI", species, reactions_pahs_agg)



# Print partial versions of the kinetic mechanisms
# !!! Be careful: only the reactions in which the selected species are included
# !!! This is more restrictive the mechanisms above, because reactions involving the selected species and species
# !!! hierarchically lower are excluded
# !!! These mechanisms are not intended for adoption in simulations but for analysis purposes only

species = kinetics.SpeciesInSelectedReactions(reactions_only_sp)
kinetics.PrintKineticMechanism("SubKineticsRestricted.01.SP.CKI", species, reactions_only_sp)

species = kinetics.SpeciesInSelectedReactions(reactions_only_agg)
kinetics.PrintKineticMechanism("SubKineticsRestricted.02.AGG.CKI", species, reactions_only_agg)

species = kinetics.SpeciesInSelectedReactions(reactions_sp_agg)
kinetics.PrintKineticMechanism("SubKineticsRestricted.03.SP-AGG.CKI", species, reactions_sp_agg)



# Partial mechanisms
# They can be really adopted for simulations!

# 01) Mechanism including gas-phase species only
reactions = reactions_only_gas
species = kinetics.SpeciesInSelectedReactions(reactions)
kinetics.PrintKineticMechanism("Kinetics.01.GasOnly.CKI", species, reactions)

# 02) Mechanism up to PAHs12 (without soot)
reactions = reactions_only_gas + reactions_pahs_pah12
reaction_indices = [reactions_only_gas, reactions_pahs_pah12]
reaction_classes = ['GAS_ONLY', 'PAHS_PAH12']
reaction_comments = ['Gas-phase species only (no PAHs, no soot)', 'PAHs12 only (no larger PAHs, no soot)']
species = kinetics.SpeciesInSelectedReactions(reactions)
kinetics.PrintKineticMechanismByClasses("Kinetics.02.UpToPAHs12.CKI", False, species, reaction_indices, reaction_classes, reaction_comments)

# 03) Mechanism up to PAHs34 (without soot)
reactions = reactions_only_gas + reactions_pahs_pah12 + reactions_pahs_pah34
reaction_indices = [reactions_only_gas, reactions_pahs_pah12, reactions_pahs_pah34]
reaction_classes = ['GAS_ONLY', 'PAHS_PAH12', 'PAHS_PAH34']
reaction_comments = ['Gas-phase species only (no PAHs, no soot)', 'PAHs12 only (no larger PAHs, no soot)', 'PAHs34 only (no larger PAHs, no soot)']
species = kinetics.SpeciesInSelectedReactions(reactions)
kinetics.PrintKineticMechanismByClasses("Kinetics.03.UpToPAHs34.CKI", False, species, reaction_indices, reaction_classes, reaction_comments)

# 04) Mechanism up to PAHsLP (without soot)
reactions = reactions_only_gas + reactions_pahs_pah12 + reactions_pahs_pah34 + reactions_pahs_pahlp
reaction_indices = [reactions_only_gas, reactions_pahs_pah12, reactions_pahs_pah34, reactions_pahs_pahlp]
reaction_classes = ['GAS_ONLY', 'PAHS_PAH12', 'PAHS_PAH34', 'PAHS_PAHLP']
reaction_comments = ['Gas-phase species only (no PAHs, no soot)', 'PAHs12 only (no larger PAHs, no soot)', 'PAHs34 only (no larger PAHs, no soot)', 'PAHsLP only (no soot)']
species = kinetics.SpeciesInSelectedReactions(reactions)
kinetics.PrintKineticMechanismByClasses("Kinetics.04.UpToPAHsLP.CKI", False, species, reaction_indices, reaction_classes, reaction_comments)

# 05) Mechanism up to spherical particles (SP)
reactions = reactions_only_gas + reactions_pahs_pah12 + reactions_pahs_pah34 + reactions_pahs_pahlp + reactions_pahs_sp + reactions_only_sp 
reaction_indices = [reactions_only_gas, reactions_pahs_pah12, reactions_pahs_pah34, reactions_pahs_pahlp, reactions_pahs_sp, reactions_only_sp]
reaction_classes = ['GAS_ONLY', 'PAHS_PAH12', 'PAHS_PAH34', 'PAHS_PAHLP', 'PAHS_SP', 'Soot_SP']
reaction_comments = ['Gas-phase species only (no PAHs, no soot)', 'PAHs12 only (no larger PAHs, no soot)', 'PAHs34 only (no larger PAHs, no soot)', 'PAHsLP only (no soot)', 'PAHS and SP', 'SP only']
species = kinetics.SpeciesInSelectedReactions(reactions)
kinetics.PrintKineticMechanismByClasses("Kinetics.05.UpToSP.CKI", False, species, reaction_indices, reaction_classes, reaction_comments)

# 06) Overall mechanism
species = kinetics.SpeciesInSelectedReactions(reactions_only_gas+reactions_pahs_pah12+reactions_pahs_pah34+reactions_pahs_pahlp+reactions_pahs_sp+reactions_pahs_agg+reactions_only_sp+reactions_sp_agg+reactions_only_agg)
reaction_indices = [reactions_only_gas, reactions_pahs_pah12, reactions_pahs_pah34, reactions_pahs_pahlp, reactions_pahs_sp, reactions_pahs_agg, reactions_only_sp, reactions_sp_agg, reactions_only_agg]
reaction_classes = ['GAS_ONLY', 'PAHS_PAH12', 'PAHS_PAH34', 'PAHS_PAHLP', 'PAHS_SP', 'PAHS_AGG', 'Soot_SP', 'Soot_SPAGG', 'Soot_AGG']
reaction_comments = ['GAS_ONLY', 'PAHS_PAH12', 'PAHS_PAH34', 'PAHS_PAHLP', 'PAHS_SP', 'PAHS_AGG', 'Soot_SP', 'Soot_SPAGG', 'Soot_AGG']
kinetics.PrintKineticMechanismByClasses("Kinetics.06.All.CKI", False, species, reaction_indices, reaction_classes, reaction_comments)
