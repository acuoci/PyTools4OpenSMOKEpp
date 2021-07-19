'''
MODULE: PolimiSootModule
@Authors:
    Alberto Cuoci [1]
    [1]: CRECK Modeling Lab, Department of Chemistry, Materials, and Chemical Engineering, Politecnico di Milano
@Contacts:
    alberto.cuoci@polimi.it
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.cuoci@polimi.it
'''

import math
import numpy as np
import re
from scipy import sparse
import sys


def DefaultPAH12():

    pah12 = ['C6H6','C7H8','INDENE','C10H8','C12H8','BIPHENYL','FLUORENE','C6H5C2H','C6H5C2H3','C6H5C2H5', \
             'C10H7CH3','C6H5','C6H5','C7H7','C10H7','C12H7','C12H9','CH3C6H4','C6H4C2H','C6H5C2H2', \
             'C10H7CH2','C10H6CH3','XYLENE','RXYLENE','INDENYL']

    return pah12


def DefaultPAH34():

    pah34 = ['C14H10','C16H10','C18H10','C18H14','C14H9','C16H9','C18H9','C6H5C2H4C6H5','C6H5CH2C6H5']
    
    return pah34


def DefaultPAHLPM(species):

    pahlpm = []
    for name in species:
        if re.match("BIN[1234][ABC]$", name):
            pahlpm.append(name)
    
    return pahlpm
    
    
def DefaultPAHLPR(species):

    pahlpr = []
    for name in species:
        if re.match("BIN[1234][ABC][J]", name):
            pahlpr.append(name)
    
    return pahlpr
    
    
def DefaultPAHLP(species):
    
    return DefaultPAHLPM(species) + DefaultPAHLPR(species)


def DefaultSPM(species):

    # Spherical particles (BIN5-BIN12)
    spm = []
    for name in species:
        if (re.match("BIN[56789][ABC]$", name) or re.match("BIN1[012][ABC]$", name) ) :
            spm.append(name)

    return spm
    
    
def DefaultSPR(species):

    # Spherical particles (BIN5-BIN12)
    spr = []
    for name in species:
        if (re.match("BIN[56789][ABC][J]", name) or re.match("BIN1[012][ABC][J]", name) ) :
            spr.append(name)

    return spr
    

def DefaultSP(species):
    
    return DefaultSPM(species) + DefaultSPR(species)
    
    
def DefaultAGGM(species):

    # Aggregates (BIN13-BIN25)
    aggm = []
    for name in species:
        if (re.match("BIN2[0123456789][ABC]$", name) or (re.match("BIN1[3456789][ABC]$", name)) ) :
            aggm.append(name)
            
    return aggm
    
    
def DefaultAGGR(species):

    # Aggregates (BIN13-BIN25)
    aggr = []
    for name in species:
        if (re.match("BIN2[0123456789][ABC][J]", name) or (re.match("BIN1[3456789][ABC][J]", name)) ) :
            aggr.append(name)
    
    return aggr
    
    
def DefaultAGG(species):
    
    return DefaultAGGM(species) + DefaultAGGR(species)  


# Inception
def IdentifyInception(reactions, list1, list2):
    
    haca_agents = ['C3H4-A', 'C4H4', 'C4H6', 'C6H6','CH3O2H','C2H5O2H','C6H5OH','LC5H8','C7H8','CRESOL', \
                   'INDENE','C10H8','C10H7CH3','C6H5C2H3']
    
    inception_reactions = []
    for reaction in reactions:
        if ( len(reaction['react']) == 2 ):
            if ( (reaction['react'][0] in list1 and reaction['react'][1] in list2 ) or \
                 (reaction['react'][0] in list2 and reaction['react'][1] in list1 ) ):
                
                # Check if HACA mechanism
                if ( not( len(reaction['prod']) == 2 and \
                          any(item in haca_agents for item in reaction['prod'])) ):
                    inception_reactions.append(reaction['index'])
    
    return inception_reactions
    
    
# HACA (C2H2)
def IdentifyHACAC2H2(reactions, list):
    
    haca_c2h2_reactions = []
    for reaction in reactions:
        if ( len(reaction['react']) == 2 ):
            if ( (reaction['react'][0] == 'C2H2' or reaction['react'][1] == 'C2H2' ) and \
                  any(item in reaction['prod'] for item in list ) ) :
                
                # Check for HACA-HABS
                if ( not(len(reaction['prod']) == 2) ):
                    haca_c2h2_reactions.append(reaction['index'])
    
    return haca_c2h2_reactions


# HACA (Abstractions)
def IdentifyHACAAbstractions(reactions, list1, list2):

    def CheckIfDifferenceIsJ(name1, name2):
        
        if (name1 == name2): return True
        elif (name1+'J' == name2): return True
        elif (name1 == name2+'J'): return True
        else: return False
    
    def CheckIfDifferenceIsJCombinations(list1, list2):
            
        if ( CheckIfDifferenceIsJ(list1[0], list2[0]) ): return True
        elif ( CheckIfDifferenceIsJ(list1[0], list2[1]) ): return True
        elif ( CheckIfDifferenceIsJ(list1[1], list2[0]) ): return True
        elif ( CheckIfDifferenceIsJ(list1[1], list2[1]) ): return True
        else: return False
    
    def CheckIfSameSection(name1, name2): 
        
        if (name1[0:3] == 'BIN' and name2[0:3] == 'BIN'):
            section1 = re.sub(r'[A-Z]+', '', name1, re.I) 
            section2 = re.sub(r'[A-Z]+', '', name2, re.I) 
            if (section1 == section2): return True
            else: return False
        else: return False
    
    
    haca_reactions = []
    for reaction in reactions:
        if ( (len(reaction['react'])) == 2 and (len(reaction['prod']) == 2) ):
            
            if ( any(item in reaction['react'] for item in list1 ) and \
                 any(item in reaction['prod'] for item in list2 )) :
                    
                    # Check for inception reactions
                    if ( CheckIfDifferenceIsJCombinations(reaction['react'], reaction['prod']) == True and \
                         CheckIfSameSection(reaction['react'][0], reaction['react'][1]) == False ):
                        haca_reactions.append(reaction['index'])
                    
            if ( any(item in reaction['prod'] for item in list1 ) and \
                 any(item in reaction['react'] for item in list2 ) ) :
                 
                    # Check for inception reactions
                    if ( CheckIfDifferenceIsJCombinations(reaction['react'], reaction['prod']) == True and \
                         CheckIfSameSection(reaction['react'][0], reaction['react'][1]) == False ):
                        haca_reactions.append(reaction['index'])
            
    return haca_reactions


# Growth
def IdentifyGrowth(reactions, list1, list2):
    
    growth_reactions = []
    for reaction in reactions:
        if ( len(reaction['react']) == 2 ):
            if ( (reaction['react'][0] in list1 and reaction['react'][1] in list2 ) or \
                 (reaction['react'][0] in list2 and reaction['react'][1] in list1 ) ):
                growth_reactions.append(reaction['index'])
    
    return growth_reactions
    

# Identify coalescence-aggregation reactions
def IdentifyCoalescenceAndAggregationReactions(reactions, list_of_species):
    
    coalescence_aggregation_reactions = []
    for reaction in reactions:
        if ( len(reaction['react']) == 2 ):
            if ( reaction['react'][0] in list_of_species and \
                 reaction['react'][1] in list_of_species ):
                coalescence_aggregation_reactions.append(reaction['index'])
    
    return coalescence_aggregation_reactions
    

# Oxidation reactions
def IdentifyOxidationReaction(reactions, list_ox_agents, list_of_species):
    
    oxidation_reactions = []
    for reaction in reactions:
        if ( any(item in list_ox_agents for item in reaction['react']) ):
            if ( len(reaction['react']) == 2 and len(reaction['prod']) >= 3 ):
                if ( any(item in list_of_species for item in reaction['react'])):
                    
                    # Check if dehydrogenation
                    if ( not( len(reaction['prod']) == 2 and \
                              any(item in ['HO2', 'OH', 'H2O'] for item in reaction['prod'])) ):
                        oxidation_reactions.append(reaction['index'])
    
    return oxidation_reactions 


# Dehydrogenation
def IdentifyDehydrogenationFission(reactions, list_species):
    
    dehydrogenation_reactions = []
    for reaction in reactions:
        if ( len(reaction['react']) == 2 and  len(reaction['prod']) == 1 ):
            if ( ( 'H' in reaction['react']) and \
                 any(item in reaction['react'] for item in list_species) and
                 any(item in reaction['prod'] for item in list_species) ) :
                    dehydrogenation_reactions.append(reaction['index'])
        if ( len(reaction['react']) == 1 and  len(reaction['prod']) == 2 ):
            if ( ( 'H' in reaction['prod']) and \
                 any(item in reaction['react'] for item in list_species) and
                 any(item in reaction['prod'] for item in list_species) ):
                    dehydrogenation_reactions.append(reaction['index'])    
                  
    return dehydrogenation_reactions

def IdentifyDehydrogenationDemethylation(reactions, agents, list_species):
    
    dehydrogenation_reactions = []
    for reaction in reactions:
        if ( len(reaction['react']) == 2 and  len(reaction['prod']) >= 3 ):
            if ( any(item in reaction['react'] for item in agents) and \
                 any(item in reaction['react'] for item in list_species) ) :
                
                    dehydrogenation_reactions.append(reaction['index'])
    
    return dehydrogenation_reactions


def IdentifyDehydrogenationGeneral(reactions, products, list_species):
    
    dehydrogenation_reactions = []
    for reaction in reactions:
        if ( len(reaction['react']) == 1 and  len(reaction['prod']) >= 3 ):
            if ( any(item in reaction['react'] for item in list_species) and
                 any(item in reaction['prod'] for item in products) ) :
                    dehydrogenation_reactions.append(reaction['index'])
                  
    return dehydrogenation_reactions    
    