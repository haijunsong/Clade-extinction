
1. System Requirements

1.1 Operating Systems
Supported: Windows 10/11 (64-bit)
Tested: Windows 10 

1.2 Dependencies 
Package	Version	Function
Python	3.9.16	Base environment
pandas	1.5.3	Data reading/cleaning
scipy	1.9.3	Statistics (correlation, Fisher Z-transform)
numpy	1.23.5	Numerical computing
matplotlib	3.6.3	Visualization (correlation plots, forest plots)
pymc	5.6.1	Bayesian meta-analysis
arviz	0.15.1	Bayesian model diagnostics
statsmodels	0.13.5	Confidence interval calculation
openpyxl/xlrd	3.0.10/2.0.1	Excel file I/O

1.3 Hardware
CPU: Intel i5-10400 / AMD Ryzen 5 5600
RAM: 8GB (16GB recommended for Bayesian simulations)
Storage: 10GB free space

2. Installation

2.1 Steps
Create a virtual environment
Install dependencies

2.2 Typical Install Time
5–10 minutes

3 Run following the order

3.1 SQS diversity calculation
R code“R code for SQS proportinal diversity.R”
Data Data "Ammonoidea.csv", "Ammonoidea,Nautiloidea,Coleoidea,Fish.csv", "Camerata.csv", "Camerata,Pentacrinoidea.csv", "Conodonta.csv", "Conodonta,Fish.csv", "Fenestrida.csv", "Rugosa,Porifera,Bryozoa.csv", "Fusulinoidea.csv", "Foraminifera.csv", "Graptoloidea.csv", "Graptoloidea,Tentaculita.csv", "Orthoceratoidea.csv", "Orthoceratoidea,Nautiloidea.csv", "Palaeocopida.csv", "Ostracoda.csv", "Rugosa.csv", "Rugosa,Porifera,Bryozoa.csv", "Spiriferinida.csv", "Brachiopoda.csv", "Tabulata.csv", "Tabulata,Porifera,Bryozoa.csv", "Trilobita.csv", "Trilobita,Chelicerata,Pancrustacea.csv"


3.2 PyRate rate and diversity calculation
Codes in 3.2 section are from https://github.com/dsilvestro/PyRate/tree/master
First, Load the pyrate_utilities.r from the PyRte package to prepare the input data.
Parse the raw data and generate PyRate input file. Type in R: extract.ages(file="…/PyRate/Ammonoidea.txt", replicates=10) # Ten replicas will be produced.

Second, run PyRate.py ten times for each replica using RJMCMC with time-variable Poisson process model: 
python PyRate.py ./Ammonoidea_PyRate.py -A 4 -j 1 -qShift ./epochs/epochs_q.txt -s 2000 -p 100000 # j from 1 to 10. And run all the clades.

Third, combine all the log files across replicates: 
python PyRate.py -combLog ./pyrate_mcmc_logs/Ammonoidea -tag mcmc -b 200 #-b determines the proportion of burnin. And run all the clades.
python PyRate.py -combLog ./pyrate_mcmc_logs/Ammonoidea -tag ex_rates -b 200
python PyRate.py -combLog ./pyrate_mcmc_logs/Ammonoidea -tag sp_rates -b 200

Fourth, evolutionary rates and range through diversity plotting, here we used resolution of zero point one million years (-grid_plot) for rate change. 
python PyRate.py -plotRJ ./pyrate_mcmc_logs/Ammonoidea -b 200 -grid_plot 0.1 #Run all the clades.
python PyRate.py -d ./pyrate_mcmc_logs/Ammonoidea/combined_10mcmc_se_est.txt -ltt 1# resolution of million years for diversity change
python PyRate.py -ginput ./pyrate_mcmc_logs/Ammonoidea/10REP -b 200 -grid_plot 0.1

Fifth, Bayesian estimation of diversity trajectories (PyRate-corrected diversity):
python3 mcmcDivE.py -d example/Ammonoidea_PyRate.py -q example/epochs_q.txt -m example/pyrate_mcmc_logs/Rhinos_Grj_mcmc.log -b 50 -j 1 -N 5
python ./mcmcDivE/mcmcDivE.py -d ./Ammonoidea_PyRate.py -q epochs/Aepochs_q.txt -m ./pyrate_mcmc_logs/Ammonoidea_combined_10mcmc.log -b 372 -j 10 -N 0  # using combined log file and ten replicas. And run all the clades.
Plot estimated diversity trajectories. Plotting functions to summarize the estimated diversity trajectories are implemented in the R script plot_mcmcDivE_results.R:
source("path_to_script/plot_mcmcDivE_results.R")
log_file = "path_to_log_file/Ammonoidea_PyRate_mcmcdiv.log" #Run all the clades 
plot_diversity(log_file)

3.3 GAMs predict proportional diversity peak
R code "GAMs predict proportional diversity peak_SQS"
Data "Ammonoidea_ratio_summary.csv", "Camerata_ratio_summary.csv", "Conodonta_ratio_summary.csv", "Fenestrida_ratio_summary.csv", "Fusulinoidea_ratio_summary.csv", "Graptoloidea_ratio_summary.csv", "Orthoceratoidea_ratio_summary.csv", "Palaeocopida_ratio_summary.csv", "Rugosa_ratio_summary.csv", "Spiriferinida_ratio_summary.csv", "Tabulata_ratio_summary.csv", "Trilobita_ratio_summary.csv"

R code "GAMs predict proportional diversity peak_PyRate"
Data "Nautiloidea,Coleoidea,Agnatha,placodermi,chondrichthyes,Actinopterygii,Coelacanthimorpha,Dipnoi,Dipnomorphadiversity_ratio_2025-11-13.csv", "Archaeocyatha vs. Demospongea,Hexactinellida,Calcarea diversity_ratio_2025-11-13.csv", "Camerata vs. Pentacrinoidea diversity_ratio_2025-11-13.csv", "Conodonta vs. Agnatha,placodermi,chondrichthyes,Actinopterygii,Coelacanthimorpha,Dipnoi,Dipnomorphadiversity_ratio_2025-11-13.csv", "Fenestrida vs. Rugosa,Tabulata,Porifera,Cryptostomata,Trepostomata,Cystoporata,Esthonioporata,cyclostomata diversity_ratio_2025-11-13.csv", "Fusulinoidea vs. others diversity_ratio_2025-11-13.csv", "Graptoloidea vs. Tentaculitadiversity_ratio_2025-11-13.csv", "Orthoceratoidea vs. Nautiloidea diversity_ratio_2025-11-13.csv", "Palaeocopida vs. others diversity_ratio_2025-11-13.csv", "Rugosa vs. Porifera,Bryozoa diversity_ratio_2025-11-13.csv", "Spiriferinida vs. others diversity_ratio_2025-11-13.csv", "Tabulata vs. Porifera,Bryozoa diversity_ratio_2025-11-13.csv", "Trilobita vs. Chelicerata,Pancrustacea diversity_ratio_2025-11-13.csv"

3.4 Correlation analysis between diversity and variables
Python code “Correlation analysis_diversity vs age and environmental factors.py”
Data "SQS result vs Environmental factros v0.5.xlsx", "Pyrate result vs environmental factors v0.3.xlsx"

3.5 Correlation ellipse plot
Python cade "Ellipse correlation plot v0.2.py"
Data "correlation_results_corrected_SQS.xlsx", "Correlation_results_corrected_PyRate.xlsx"

3.6 Bayesian meta-analysis
Python code "Meta analysis ρ value_Bayes.py"
Data "correlation_results_corrected_SQS_transpose.xlsx", "Correlation_results_corrected_PyRate_transpose.xlsx"

3.7 Plot the results of Meta analysis
Python code "Meta analysis ρ value_Bayes_plot fig.py"
Data "Meta_Analysis_Results_SQS.xlsx", "Meta_Analysis_Results_Pyrate.xlsx"

3.8 Plot the results of regression
R code "R code for regression.R"
Data "Ammonoidea_ratio_summary_decine phase.csv, Cameraata_ratio_summary_decine phase.csv, Conodonta_ratio_summary_decine phase.csv, Fenestrida_ratio_summary_decine phase.csv, Fusulinoidea_ratio_summary_decine phase.csv, Graptoloidea_ratio_summary_decine phase.csv, Orthoceratoidea_ratio_summary_decine phase.csv, Palaeocopida_ratio_summary_decine phase.csv,  Rugosa_ratio_summary_decine phase.csv, Spiriferinida_ratio_summary_decine phase.csv, Tabulata_ratio_summary_decine phase.csv, Trilobita_ratio_summary_decine phase.csv"

3.9 Extinction simulation: numerical model
Python code "Specialization model_spatial.py"

3.10 Extinction simulation: analytical model
Python code "Mathematical model for clade extinction.py"
Python code "Mathematical model for clade extinction_specialization.py"



