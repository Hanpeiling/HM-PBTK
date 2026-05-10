# HM-PBTK
## About
This repository contains the code and resources of the following article:

**AI-driven prediction of biological exposure and toxicity of chemicals in freshwater and marine fish: advancing aquatic ecological risk assessment (Under revision)**

## Overview of the framework
The high-throughput multi-species physiologically based toxicokinetic (HM-PBTK) allows you to predict the distribution of chemical concentrations in specific tissues in a wide range of fish species after exposure to chemicals in water. We are equipped with physiological parameters for 151 fishes and descriptors for over 40,000 chemicals to support model operations.

The HM-PBTK model consists of 12 compartments: adipose fat, brain, gastrointestinal tract (GIT), gonads, kidneys, liver, skin, poorly perfused tissues (PPT; mainly muscles), richly perfused tissues (RPT; other viscera including the heart, spleen, etc.), arterial blood, and venous blood. It is assumed that all compartments are well-mixed. Chemical absorption occurs through the gills (water absorption), and elimination occurs via expiration and hepatic metabolism.

Please refer to the following steps to run the HM-PBTK model

**Step 1**: Fill in the data. Fill in the Excel file called "Batch prediction" (in the batch_pipeline folder) with information including species, chemicals, and exposure conditions. 

**Step 2**: Find data on chemicals. Run the first.py (in the the batch_pipeline folder) to determine if your target chemical is included in our biochemical parameters file. In the biochemical parameter file, we have pre-calculated biochemical parameters for over 40,000 chemicals for your direct use. If your target chemical is not in our biochemical parameters file, please contact us via email (hplzhr@dlut.edu.cn), we will add the relevant data as soon as possible.

**Step 3**: Prepare physiological and biochemical parameters. Run the main.py  (in the batch_pipeline folder) to search for species' physiological parameters from our physiological parameter dataset, and use the ML models to calculate multi-species cardiac output, oxygen consumption, and hepatic clearance rate.

The code for developing multi-species cardiac output, oxygen consumption, and hepatic clearance rate models was provided in folders named Fcard_model, VO2_model, and CL_model, respectively. Running main.py in these folders can build the models.

**Step 4**: Construct the HM-PBTK model. We have provided four R codes in the HM-PBTK folder for constructing HM-PBTK models, including two for ionizable chemicals and two for neutral chemicals. SET I and SET II represent two methods for calculating partition coefficients, the formulas for which can be found in the manuscript "AI-driven prediction of biological exposure and toxicity of chemicals in freshwater and marine fish: advancing aquatic ecological risk assessment".
For ionizable chemicals, SET I is recommended. For neutral chemicals, SET II is recommended. If it is unclear whether the target chemical is ionizable or neutral, SET II is recommended.

**Step 5**: View the results of the HM-PBTK model run. The HM-PBTK model run result files can be found in the results folder.


## User-friendly platform
To facilitate the use of the HM-PBTK model by risk assessors for simulating the absorption, distribution, metabolism and excretion (ADME) processes of chemicals across multiple species and conducting the quantitative in vitro to in vivo extrapolation (QIVIVE) toxicity assessments, a user-friendly platform (https://www.ierp-dlut.cn/) named Integrated Ecological Risk Predictions (IERP) was established to provide ecological risk prediction based on the HM-PBTK model.

By simply inputting species information (species name, species weight and length), chemical information, exposure scenarios (exposure time and dose, water temperature and pH), and in vitro toxicity data, IERP can provide a comprehensive chain service, including physiological and biochemical parameter prediction, HM-PBTK model application, chemical ADME process analysis, and in vivo toxicity estimation. 


<img width="1294" height="948" alt="figure" src="https://github.com/user-attachments/assets/1bb581ef-a878-4080-9244-d4cdcb5c34b6" />
