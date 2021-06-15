# Import main libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.sparse import csr_matrix, find

# Define path to PyTools classes
import sys
sys.path.insert(0, '../src')

# Import PyTools classes
from KineticMechanism import KineticMechanism
from OpenSMOKEppXMLFile import OpenSMOKEppXMLFile

# Write single XML file 
def WriteXMLFile(output_file, kinetics, simulations):

    # Total number of observations (i.e. points)
    nitems = 0
    for i in range(len(simulations)):
        nitems = nitems + simulations[i].npts
    
    f = open(output_file, "w")
    f.write("<?xml version=\"1.0\" ?>\n")
    f.write("<opensmoke>\n")    

    f.write("<classes>1</classes>\n")

    f.write("<items>")
    f.write(str(nitems))
    f.write("</items>\n")

    f.write("<original-components>")
    f.write(str(1+kinetics.ns))
    f.write("</original-components>\n")

    f.write("<filtered-components>")
    f.write(str(1+kinetics.ns))
    f.write("</filtered-components>\n")

    f.write("<removed-components>0</removed-components>\n")

    f.write("<number-retained-species>")
    f.write(str(kinetics.ns))
    f.write("</number-retained-species>\n")


    f.write("<data-original>\n")

    for i in range(len(simulations)):
        simulations[i].WriteThermophysicalState(f)

    f.write("</data-original>\n") 
            
    f.write("</opensmoke>\n")
    f.close()
    

# Import kinetic mechanism in XML format
kin_xml_folder_name="C:\\Users\\acuoci\\OneDrive - Politecnico di Milano\\My Projects\\GitHub\\CRECK_DiscreteSectionalModel_v2012\\CRECK_2012_Soot_OnlyBINJ_NoHeavyFuels\\kinetics-CRECK_2012_SootOnlyBINJ-SP-AGG\\"
kinetics = KineticMechanism(kin_xml_folder_name + "kinetics.xml")


# Define list of XML files corresponding to the different simulation results
# Batch reactors, laminar 1D flames

main_folder_name = "C:\\Users\\acuoci\\Aachen-Data\\Reductions\\BatchReactors\\"

list_xml_files = []

folder_name= main_folder_name + "Phi-1.0\\Output\\"
for i in range(7): list_xml_files.append(folder_name + "Case" + str(i) + "\\" + "Output.xml")

folder_name= main_folder_name + "Phi-1.5\\Output\\"
for i in range(7): list_xml_files.append(folder_name + "Case" + str(i) + "\\" + "Output.xml")

folder_name= main_folder_name + "Phi-2.0\\Output\\"
for i in range(7): list_xml_files.append(folder_name + "Case" + str(i) + "\\" + "Output.xml")

folder_name= main_folder_name + "Phi-3.0\\Output\\"
for i in range(7): list_xml_files.append(folder_name + "Case" + str(i) + "\\" + "Output.xml")

folder_name= main_folder_name + "Phi-4.0\\Output\\"
for i in range(7): list_xml_files.append(folder_name + "Case" + str(i) + "\\" + "Output.xml")

folder_name= main_folder_name + "Phi-5.0\\Output\\"
for i in range(7): list_xml_files.append(folder_name + "Case" + str(i) + "\\" + "Output.xml")

folder_name= main_folder_name + "Phi-Inf\\Output\\"
for i in range(18): list_xml_files.append(folder_name + "Case" + str(i) + "\\" + "Output.xml")


# Read and append simulation results (XML format)
simulations = []
for i in range(len(list_xml_files)):
    simulations.append(OpenSMOKEppXMLFile(list_xml_files[i], kinetics))


# Write single XML file
WriteXMLFile("data.xml", kinetics, simulations)
