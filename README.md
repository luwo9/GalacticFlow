# GalacticFlow

GalacticFlow applies machine learning to astrophysics. It uses conditional normalizing flows to learn a generalized representation of galaxies. It learns the (extended) distribution function of galactic (stellar) data conditioned on global galactic parameters, such as, e.g., total stellar mass. Galactic flow then provides a compact generative model that allows sampling a galaxy at any desired parameters with any desired number of stars, as well as exactly evaluating the distribution function, essentially interpolating and generalizing in galactic parameter space.


<img src="https://github.com/luwo9/GalacticFlow/assets/126659866/e3a9c26e-306c-4a0d-8981-e939fabcc127" alt="Galactic Flow scheme" width="700">

## User guide
While it is possible to work on a very low level with maximum flexibility, see `Base_Processor_Workflow.ipynb`, its strongly recommended to use the user friendly API. The usage of the API is documented in `API_Workflow.ipynb`, it allows loading pretrained models, defining and taining your own ones, sampling a glaxy, as well as evaluating its pdf with minimum effort and code. It implements the low level workflow under the hood, so it might still be insightful to read `Base_Processor_Workflow.ipynb`.

### Reproducing NeurIPS paper results
`reproduce.ipynb` also provides the direct teps to reproduce the results in the paper. The packages used in our python (3.8.10) environment can be found in `requirements.txt` (not all packages are necessary to run GalacticFlow).
The data used, as well as the pre trained models can be found at https://doi.org/10.5281/zenodo.8389555 for download.
The data must be contained in the `all_sims` folder in your current directory for the pre trained models to work, you can put the models in any folder and specify the right path whenever needed.

### Reusing the code
`Adapting_GF.ipynb` provides more details on how to adapt the code to your own data and needs.
