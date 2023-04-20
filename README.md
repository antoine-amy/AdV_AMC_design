<table align="center"><tr><td align="center" width="9999">

# Auxiliary Mode Cleaner design and studies package
## AMC development for the Virgo interferometer

</td></tr></table>

----------------------------------------------------------------------------------------------------------------------------------------------------------------

## Folder structure:

    My Directory
    │  
    │ MC_design.ipynb                      # Notebook for tests of the Mode Cleaner Cavity design (Length, RoC, Finesse, etc)
    │ MC_design_functions.py               # Python file containing the functions used in the ```MC_design``` notebook
    │ OSCAR_data_Analysis.ipynb            # Notebook for the analysis of the data coming from the OSCAR simulations
    │ data_analysis_functions.py           # Python file containing the functions used in the ```OSCAR_data_Analysis``` notebook
    ├── data                               # Folder containing the data files from the OSCAR simulations
    └── Results                            # Folder containing the results of the simulations/data analysis
    	├── fields                         # Subfolder for the simulated fields from OSCAR
		└── gains                          # Subfolder for the gain studies with OSCAR
        └── signals                        # Subfolder for the PDs & quads signals studies with OSCAR

----------------------------------------------------------------------------------------------------------------------------------------------------------------

## How to use:

Just open the notebooks with the corresponding python file in the same directory.