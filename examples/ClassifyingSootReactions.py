'''
SCRIPT: ClassifyingSootReactions
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

# Testing with already existing classification
test_with_file = False

# First reaction belonging to the soot group
first_reaction_soot = 8526-1
first_reaction_soot = 6601-1


# Official CRECK2012 Soot BINJOnly
kin_xml_folder_name="C:\\Users\\acuoci\\OneDrive - Politecnico di Milano\\My Projects\\GitHub\\CRECK_DiscreteSectionalModel_v2012\\CRECK_2012_Soot_OnlyBINJ\\kinetics-CRECK_2012_SootOnlyBINJ\\"
kin_xml_folder_name="/home/chimica2/cuoci/Applications/OfficialKineticMechanisms/CRECK_2012_Soot_OnlyBINJ/kinetics-CRECK_2012_SootOnlyBINJ/"
kin_xml_folder_name="/home/chimica2/cuoci/Applications/OfficialKineticMechanisms/CRECK_2012_Soot_OnlyBINJ_NoHeavyFuels/kinetics-CRECK_2012_SootOnlyBINJ-SP-AGG/"
kinetics = KineticMechanism(kin_xml_folder_name + "kinetics.xml")
kinetics.ReadKinetics(kin_xml_folder_name + "reaction_names.xml")
kinetics.ProcessingReactions()


# Create groups
kinetics.AddGroupOfSpecies('PAH12', 'PAHs with 1/2 aromatic rings', DefaultPAH12())
kinetics.AddGroupOfSpecies('PAH34', 'PAHs with 3/4 aromatic rings', DefaultPAH34())
kinetics.AddGroupOfSpecies('PAHLP', 'PAHs with more than 4 aromatic rings (molecular and radical)', DefaultPAHLP(kinetics.species))
kinetics.AddGroupOfSpecies('PAHLP-M', 'PAHs with more than 4 aromatic rings (molecular)', DefaultPAHLPM(kinetics.species))
kinetics.AddGroupOfSpecies('PAHLP-R', 'PAHs with more than 4 aromatic rings (radical)', DefaultPAHLPR(kinetics.species))
kinetics.AddGroupOfSpecies('SP', 'BIN sections corresponding to spherical particles (molecular and radical)', DefaultSP(kinetics.species))
kinetics.AddGroupOfSpecies('SP-M', 'BIN sections corresponding to spherical particles (molecular)', DefaultSPM(kinetics.species))
kinetics.AddGroupOfSpecies('SP-R', 'BIN sections corresponding to spherical particles (radical)', DefaultSPR(kinetics.species))
kinetics.AddGroupOfSpecies('AGG', 'BIN sections corresponding to aggregates (molecular and radical)', DefaultAGG(kinetics.species))
kinetics.AddGroupOfSpecies('AGG-M', 'BIN sections corresponding to aggregates (molecular)', DefaultAGGM(kinetics.species))
kinetics.AddGroupOfSpecies('AGG-R', 'BIN sections corresponding to aggregates (radical)', DefaultAGGR(kinetics.species))


# List of species and reactions
list_of_pahs = kinetics.Group('PAH12')['list'] + kinetics.Group('PAH34')['list'] + kinetics.Group('PAHLP')['list']
reactions_pahs_all, reactions_pahs_pah12, reactions_pahs_pah34, reactions_pahs_pahlp, reactions_pahs_sp, reactions_pahs_agg = kinetics.PartitionSootPrecursorsReactions(list_of_pahs)


# Select soot reactions only
soot_reactions = []
for reaction in kinetics.reactions:
    if (reaction['index'] >= first_reaction_soot):
        soot_reactions.append(reaction)
        


# Summary from file
if (test_with_file == True):

	classes = kinetics.OrganizeReactionsInClasses(range(kinetics.nr))

	# Inception reactions (0-10)
	inception_pahr_pahr_from_file = classes[kinetics.reaction_class_name.index('INCEP-PAHr+PAHr')]
	inception_pahr_pahm_from_file = classes[kinetics.reaction_class_name.index('INCEP-PAHr+PAHm')]
	inception_pahm_pahm_from_file = classes[kinetics.reaction_class_name.index('INCEP-PAHm+PAHm')]
	inception_pahm_binj_from_file = classes[kinetics.reaction_class_name.index('INCEP-PAHm+BINJ')]
	inception_pahr_bin_from_file = classes[kinetics.reaction_class_name.index('INCEP-PAHr+BIN')]
	inception_pahr_binj_from_file = classes[kinetics.reaction_class_name.index('INCEP-PAHr+BINJ')]
	inception_binj_binj_from_file = classes[kinetics.reaction_class_name.index('INCEP-BINJ+BINJ')]
	inception_bin_binj_from_file = classes[kinetics.reaction_class_name.index('INCEP-BIN+BINJ')]
	inception_bin_bin_from_file = classes[kinetics.reaction_class_name.index('INCEP-BIN+BIN')]
	inception_rr_binj_from_file = classes[kinetics.reaction_class_name.index('INCEP-RR+BINJ')]
	inception_rr_bin_from_file = classes[kinetics.reaction_class_name.index('INCEP-RR+BIN')]
	inception_total_from_file = inception_pahr_pahr_from_file + inception_pahr_pahm_from_file + inception_pahm_pahm_from_file + \
		                    inception_pahm_binj_from_file + inception_pahr_bin_from_file + inception_pahr_binj_from_file + \
		                    inception_binj_binj_from_file + inception_bin_binj_from_file + inception_bin_bin_from_file + \
		                    inception_rr_binj_from_file + inception_rr_bin_from_file

	# HACA reactions (11-13)
	haca_c2h2bin_from_file = classes[kinetics.reaction_class_name.index('HACA-C2H2_2_BIN')] 
	haca_c2h2binj_from_file = classes[kinetics.reaction_class_name.index('HACA-C2H2_2_BINJ')]
	haca_habs_from_file = classes[kinetics.reaction_class_name.index('HACA-HABS')]
	haca_c2h2_from_file = haca_c2h2bin_from_file + haca_c2h2binj_from_file
	haca_total_from_file = haca_c2h2_from_file + haca_habs_from_file

	# Growth reactions (14-17)
	growth_pahm_binj_from_file = classes[kinetics.reaction_class_name.index('GROWTH-PAHm+BINJ')]
	growth_pahr_bin_from_file = classes[kinetics.reaction_class_name.index('GROWTH-PAHr+BIN')] 
	growth_binj_binj_from_file = classes[kinetics.reaction_class_name.index('GROWTH-BINJ+BINJ')]
	growth_bin_bin_from_file = classes[kinetics.reaction_class_name.index('GROWTH-BIN+BIN')]
	growth_total_from_file = growth_pahm_binj_from_file + growth_pahr_bin_from_file + \
		                 growth_binj_binj_from_file + growth_bin_bin_from_file

	# Coalescence/aggregation reactions (18)
	coalagg_reactions_from_file = classes[kinetics.reaction_class_name.index('COALAGG-BIN+BIN')]

	# Dehydrogenation reactions (19-23)
	deh_fisrec_from_file = classes[kinetics.reaction_class_name.index('DEH-CH_FISREC')]
	deh_demeth_from_file = classes[kinetics.reaction_class_name.index('DEH-DEMETH')]
	deh_deh_from_file = classes[kinetics.reaction_class_name.index('DEH-DEH')]
	deh_h2_binj_from_file = classes[kinetics.reaction_class_name.index('DEH-DEH2_BINJ')]
	deh_h2_bin_from_file = classes[kinetics.reaction_class_name.index('DEH-DEH2_BIN')]
	deh_total_from_file = deh_fisrec_from_file + deh_demeth_from_file + deh_deh_from_file + \
		              deh_h2_binj_from_file + deh_h2_bin_from_file

	# Oxidation reactions (24-35)
	ox_o2_pahlp_from_file = classes[kinetics.reaction_class_name.index('OX-O2-LPAH')]
	ox_oh_pahlp_from_file = classes[kinetics.reaction_class_name.index('OX-OH-LPAH')]
	ox_o_pahlp_from_file = classes[kinetics.reaction_class_name.index('OX-O-LPAH')]
	ox_ho2_pahlp_from_file = classes[kinetics.reaction_class_name.index('OX-HO2-LPAH')]

	ox_o2_sp_from_file = classes[kinetics.reaction_class_name.index('OX-O2-SP')]
	ox_oh_sp_from_file = classes[kinetics.reaction_class_name.index('OX-OH-SP')]
	ox_o_sp_from_file = classes[kinetics.reaction_class_name.index('OX-O-SP')]
	ox_ho2_sp_from_file = classes[kinetics.reaction_class_name.index('OX-HO2-SP')]

	ox_o2_agg_from_file = classes[kinetics.reaction_class_name.index('OX-O2-AGG')]
	ox_oh_agg_from_file = classes[kinetics.reaction_class_name.index('OX-OH-AGG')]
	ox_o_agg_from_file = classes[kinetics.reaction_class_name.index('OX-O-AGG')]
	ox_ho2_agg_from_file = classes[kinetics.reaction_class_name.index('OX-HO2-AGG')]

	oxidation_o2_from_file = ox_o2_pahlp_from_file + ox_o2_sp_from_file + ox_o2_agg_from_file
	oxidation_oh_from_file = ox_oh_pahlp_from_file + ox_oh_sp_from_file + ox_oh_agg_from_file
	oxidation_o_from_file = ox_o_pahlp_from_file + ox_o_sp_from_file + ox_o_agg_from_file
	oxidation_ho2_from_file = ox_ho2_pahlp_from_file + ox_ho2_sp_from_file + ox_ho2_agg_from_file
	oxidation_total_from_file = oxidation_o2_from_file + oxidation_oh_from_file + \
		                    oxidation_o_from_file + oxidation_ho2_from_file

	soot_total_from_file = inception_total_from_file + haca_total_from_file + \
		               growth_total_from_file + coalagg_reactions_from_file + \
		               deh_total_from_file + oxidation_total_from_file

	print('Inception reactions:       ', len(inception_total_from_file))
	print('HACA reactions:            ', len(haca_total_from_file))
	print('Growth reactions:          ', len(growth_total_from_file))
	print('Coal/Agg reactions:        ', len(coalagg_reactions_from_file))
	print('Dehydrogenation reactions: ', len(deh_total_from_file))
	print('Oxidation reactions:       ', len(oxidation_total_from_file))
	print('Total soot reactions:      ', len(soot_total_from_file))



# Inception
inception_binbin_reactions = IdentifyInception(soot_reactions, kinetics.Group('PAHLP')['list'], kinetics.Group('PAHLP')['list'])
inception_pahpah_reactions = IdentifyInception(soot_reactions, ['RC9H11'] + kinetics.Group('PAH12')['list'] + kinetics.Group('PAH34')['list'], kinetics.Group('PAH12')['list']+kinetics.Group('PAH34')['list'])
inception_pahbin_reactions = IdentifyInception(soot_reactions, kinetics.Group('PAH12')['list']+kinetics.Group('PAH34')['list'], kinetics.Group('PAHLP')['list'])
inception_rrbin_reactions = IdentifyInception(soot_reactions, ['C3H3', 'C4H5', 'C4H3', 'C5H5'], kinetics.Group('PAHLP')['list']+kinetics.Group('SP')['list']+kinetics.Group('AGG')['list'])
inception_reactions = inception_binbin_reactions + inception_pahpah_reactions + \
                      inception_pahbin_reactions + inception_rrbin_reactions
                      
                      
# HACA reactions
haca_c2h2_reactions = IdentifyHACAC2H2(soot_reactions, kinetics.Group('PAHLP')['list'] + kinetics.Group('SP')['list'] + kinetics.Group('AGG')['list'])

pahlpm = []
pahlpr = []
for pah in kinetics.Group('PAHLP')['list']:
    if (pah[-1] == 'J'): pahlpm.append(pah)
    else: pahlpr.append(pah)

haca_habs_reactions = IdentifyHACAAbstractions(soot_reactions, pahlpm, pahlpr)

haca_reactions = haca_c2h2_reactions + haca_habs_reactions


# Growth reactions
growth_pahbin_reactions = IdentifyGrowth(soot_reactions, kinetics.Group('PAH12')['list']+kinetics.Group('PAH34')['list'], kinetics.Group('SP')['list'] + kinetics.Group('AGG')['list'])
growth_binbin_reactions = IdentifyGrowth(soot_reactions, kinetics.Group('PAHLP')['list'], kinetics.Group('SP')['list'] + kinetics.Group('AGG')['list'])
growth_reactions = growth_pahbin_reactions + growth_binbin_reactions                      
                      
                      
# Coalsescence/Aggregation reactions
coalagg_reactions = IdentifyCoalescenceAndAggregationReactions(soot_reactions, kinetics.Group('SP')['list'] + kinetics.Group('AGG')['list'])                      
                      
                      
# Dehydrogenation reactions
dehydrogenation_fission_reactions = IdentifyDehydrogenationFission(soot_reactions, kinetics.Group('PAHLP')['list'])
dehydrogenation_demeth_reactions = IdentifyDehydrogenationDemethylation(soot_reactions, ['H','CH3'], kinetics.Group('PAHLP')['list'] + kinetics.Group('SP')['list'] + kinetics.Group('AGG')['list'])
dehydrogenation_general_reactions = IdentifyDehydrogenationGeneral(soot_reactions, ['H','H2'], kinetics.Group('PAHLP')['list'] + kinetics.Group('SP')['list'] + kinetics.Group('AGG')['list'])
dehydrogenation_reactions = dehydrogenation_fission_reactions + dehydrogenation_demeth_reactions + dehydrogenation_general_reactions


# Oxidation reactions
ox_o2_pahlp_reactions  = IdentifyOxidationReaction(soot_reactions, ['O2'], kinetics.Group('PAHLP')['list']) 
ox_oh_pahlp_reactions  = IdentifyOxidationReaction(soot_reactions, ['OH'], kinetics.Group('PAHLP')['list']) 
ox_o_pahlp_reactions   = IdentifyOxidationReaction(soot_reactions, ['O'], kinetics.Group('PAHLP')['list']) 
ox_ho2_pahlp_reactions = IdentifyOxidationReaction(soot_reactions, ['HO2'], kinetics.Group('PAHLP')['list']) 

ox_o2_sp_reactions  = IdentifyOxidationReaction(soot_reactions, ['O2'], kinetics.Group('SP')['list']) 
ox_oh_sp_reactions  = IdentifyOxidationReaction(soot_reactions, ['OH'], kinetics.Group('SP')['list']) 
ox_o_sp_reactions   = IdentifyOxidationReaction(soot_reactions, ['O'], kinetics.Group('SP')['list']) 
ox_ho2_sp_reactions = IdentifyOxidationReaction(soot_reactions, ['HO2'], kinetics.Group('SP')['list'])

ox_o2_agg_reactions  = IdentifyOxidationReaction(soot_reactions, ['O2'], kinetics.Group('AGG')['list']) 
ox_oh_agg_reactions  = IdentifyOxidationReaction(soot_reactions, ['OH'], kinetics.Group('AGG')['list']) 
ox_o_agg_reactions   = IdentifyOxidationReaction(soot_reactions, ['O'], kinetics.Group('AGG')['list']) 
ox_ho2_agg_reactions = IdentifyOxidationReaction(soot_reactions, ['HO2'], kinetics.Group('AGG')['list'])

ox_o2_reactions = ox_o2_pahlp_reactions + ox_o2_sp_reactions + ox_o2_agg_reactions
ox_oh_reactions = ox_oh_pahlp_reactions + ox_oh_sp_reactions + ox_oh_agg_reactions
ox_o_reactions = ox_o_pahlp_reactions + ox_o_sp_reactions + ox_o_agg_reactions
ox_ho2_reactions = ox_ho2_pahlp_reactions + ox_ho2_sp_reactions + ox_ho2_agg_reactions
ox_reactions = ox_o2_reactions + ox_oh_reactions + ox_o_reactions + ox_ho2_reactions

# Total soot reactions
soot_total_reactions = inception_reactions + haca_reactions + growth_reactions + \
                       coalagg_reactions + dehydrogenation_reactions + ox_reactions

# Summary
print('Inception reactions:       ', len(inception_reactions))
print('HACA reactions:            ', len(haca_reactions))
print('Growth reactions:          ', len(growth_reactions))
print('Coal/Agg reactions:        ', len(coalagg_reactions))
print('Dehydrogenation reactions: ', len(dehydrogenation_reactions))
print('Oxidation reactions:       ', len(ox_reactions))
print('Total soot reactions:      ', len(soot_total_reactions))                      


# Comparison
if (test_with_file == True):
	print('Inception reactions:       ', len(inception_total_from_file), len(inception_reactions) )
	print('HACA reactions:            ', len(haca_total_from_file), len(haca_reactions))
	print('  - C2H2 reactions:        ', len(haca_c2h2_from_file), len(haca_c2h2_reactions))
	print('  - H-abs reactions:       ', len(haca_habs_from_file), len(haca_habs_reactions))
	print('Growth reactions:          ', len(growth_total_from_file), len(growth_reactions))
	print('Coal/Agg reactions:        ', len(coalagg_reactions_from_file), len(coalagg_reactions))
	print('Dehydrogenation reactions: ', len(deh_total_from_file), len(dehydrogenation_reactions))
	print('Oxidation reactions:       ', len(oxidation_total_from_file), len(ox_reactions))
	print('  - O2 reactions:          ', len(oxidation_o2_from_file), len(ox_o2_reactions))
	print('  - OH reactions:          ', len(oxidation_oh_from_file), len(ox_oh_reactions))
	print('  - O reactions:           ', len(oxidation_o_from_file), len(ox_o_reactions))
	print('  - HO2 reactions:         ', len(oxidation_ho2_from_file), len(ox_ho2_reactions))
	print('Total soot reactions:      ', len(soot_total_from_file), len(soot_total_reactions))

else:
	print('Inception reactions:       ', len(inception_reactions) )
	print('HACA reactions:            ', len(haca_reactions))
	print('  - C2H2 reactions:        ', len(haca_c2h2_reactions))
	print('  - H-abs reactions:       ', len(haca_habs_reactions))
	print('Growth reactions:          ', len(growth_reactions))
	print('Coal/Agg reactions:        ', len(coalagg_reactions))
	print('Dehydrogenation reactions: ', len(dehydrogenation_reactions))
	print('Oxidation reactions:       ', len(ox_reactions))
	print('  - O2 reactions:          ', len(ox_o2_reactions))
	print('  - OH reactions:          ', len(ox_oh_reactions))
	print('  - O reactions:           ', len(ox_o_reactions))
	print('  - HO2 reactions:         ', len(ox_ho2_reactions))
	print('Total soot reactions:      ', len(soot_total_reactions), len(soot_reactions))


# ---------------------------------------------------------------------------------------------- #   
# Print kinetic mechanism with Soot sections
# ---------------------------------------------------------------------------------------------- # 

f = open("kinetics.classified.CKI", "w")
    
# ---------------------------------------------------------------------------------------------- #   
# Write Summary
# ---------------------------------------------------------------------------------------------- #    
f.write('! CRECK Modeling Lab @ Politecnico di Milano \n')
f.write('! Please visit our web-site:  http://creckmodeling.chem.polimi.it/ \n')
f.write('\n')
f.write('! Total number of reactions: ' + str(kinetics.nr) + '\n')
       
# ---------------------------------------------------------------------------------------------- #   
# Write ELEMENTS section
# ---------------------------------------------------------------------------------------------- #    
kinetics.PrintElementsSectionInCHEMKIN(f)
    
# ---------------------------------------------------------------------------------------------- #   
# Write SPECIES section
# ---------------------------------------------------------------------------------------------- #
kinetics.PrintSpeciesSectionInCHEMKIN(f, kinetics.species)

# ---------------------------------------------------------------------------------------------- #   
# Write REACTIONS section
# ---------------------------------------------------------------------------------------------- #
f.write('REACTIONS' + '\n')

# Gas-phase reactions
f.write('\n')
f.write('! Gas-phase reactions' + '\n')
for i in range(first_reaction_soot):
	index = i
	f.write(kinetics.reaction_lines[index])
f.write('\n')
f.write('\n')
f.write('\n')
    

# Soot reactions
f.write('! SOOT MODEL' + '\n')

if (len(inception_reactions)):
	f.write('\n')
	f.write('! [SOOTCLASS] [INCEP]' + '\n')
	for i in range(len(inception_reactions)):
            index = inception_reactions[i]
            f.write(kinetics.reaction_lines[index])
	f.write('\n')

if (len(haca_c2h2_reactions)):
	f.write('\n')
	f.write('! [SOOTCLASS] [HACA-C2H2]' + '\n')
	for i in range(len(haca_c2h2_reactions)):
            index = haca_c2h2_reactions[i]
            f.write(kinetics.reaction_lines[index])
	f.write('\n')

if (len(haca_habs_reactions)):
	f.write('\n')
	f.write('! [SOOTCLASS] [HACA-HABS]' + '\n')
	for i in range(len(haca_habs_reactions)):
            index = haca_habs_reactions[i]
            f.write(kinetics.reaction_lines[index])
	f.write('\n')

if (len(growth_reactions)):
	f.write('\n')
	f.write('! [SOOTCLASS] [GROWTH]' + '\n')
	for i in range(len(growth_reactions)):
            index = growth_reactions[i]
            f.write(kinetics.reaction_lines[index])
	f.write('\n')

if (len(coalagg_reactions)):
	f.write('\n')
	f.write('! [SOOTCLASS] [COALAGG-BIN+BIN]' + '\n')
	for i in range(len(coalagg_reactions)):
            index = coalagg_reactions[i]
            f.write(kinetics.reaction_lines[index])
	f.write('\n')

if (len(dehydrogenation_reactions)):
	f.write('\n')
	f.write('! [SOOTCLASS] [DEH]' + '\n')
	for i in range(len(dehydrogenation_reactions)):
            index = dehydrogenation_reactions[i]
            f.write(kinetics.reaction_lines[index])
	f.write('\n')

if (len(ox_o2_reactions)):
	f.write('\n')
	f.write('! [SOOTCLASS] [OX-O2]' + '\n')
	for i in range(len(ox_o2_reactions)):
            index = ox_o2_reactions[i]
            f.write(kinetics.reaction_lines[index])
	f.write('\n')

if (len(ox_oh_reactions)):
	f.write('\n')
	f.write('! [SOOTCLASS] [OX-OH]' + '\n')
	for i in range(len(ox_oh_reactions)):
            index = ox_oh_reactions[i]
            f.write(kinetics.reaction_lines[index])
	f.write('\n')

if (len(ox_o_reactions)):
	f.write('\n')
	f.write('! [SOOTCLASS] [OX-O]' + '\n')
	for i in range(len(ox_o_reactions)):
            index = ox_o_reactions[i]
            f.write(kinetics.reaction_lines[index])
	f.write('\n')

if (len(ox_ho2_reactions)):
	f.write('\n')
	f.write('! [SOOTCLASS] [OX-HO2]' + '\n')
	for i in range(len(ox_ho2_reactions)):
            index = ox_ho2_reactions[i]
            f.write(kinetics.reaction_lines[index])
	f.write('\n')
 
# Closing the kinetic file       
f.write('\n\n')
f.write('END' + '\n')
f.close()

