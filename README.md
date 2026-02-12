# HM-PBTK
## About
This repository contains the code and resources of the following paper:

**AI-driven prediction of biological exposure and toxicity of chemicals in freshwater and marine fish: advancing aquatic ecological risk assessment(Under revision)**

## Overview of the framework
The high-throughput multi-species physiologically based toxicokinetic (HM-PBTK) allows you to predict the distribution of chemical concentrations in specific tissues in a wide range of fish species after exposure to chemicals in water.

**Step 1**: Fill in the data. Fill in the Excel file called “data” with information including species, chemicals, exposure conditions, and physiological and biochemical parameters. 
Please enter the ionizable chemicals in the worksheet named "dl" and the neutral chemicals in the worksheet named "zx".

**Step 2**: Construct the HM-PBTK model. We have provided four R codes in the “pbtk” folder for constructing HM-PBTK models, two for ionizable chemicals and two for neutral chemicals.
SET I and SET II represent two methods for calculating partition coefficients, the formulas for which can be found in the article "AI-driven prediction of biological exposure and toxicity of chemicals in freshwater and marine fish: advancing aquatic ecological risk assessment". For ionizable chemicals, SET I is recommended. For neutral chemicals, SET II is recommended. If it is unclear whether the target chemical is ionizable or neutral, SET II is recommended.

**Step 3**: View the results of the HM-PBTK model run. The HM-PBTK model run result files can be found in the "output" folder.

## User-friendly platform
To facilitate the use of the HM-PBTK model by risk assessors for simulating the absorption, distribution, metabolism and excretion (ADME) processes of chemicals across multiple species and conducting the quantitative in vitro to in vivo extrapolation (QIVIVE) toxicity assessments, a user-friendly platform (https://www.ierp-dlut.cn/) named Integrated Ecological Risk Predictions (IERP) was established to provide ecological risk prediction based on the HM-PBTK model.
![20241209111300](https://github.com/user-attachments/assets/7377052b-1b61-411a-8787-6fdf71317e99)

By simply inputting species information (species name, species weight and length), chemical information, exposure scenarios (exposure time and dose, water temperature and pH), and in vitro toxicity data, IERP can provide a comprehensive chain service, including physiological and biochemical parameter prediction, HM-PBTK model application, chemical ADME process analysis, and in vivo toxicity estimation.
