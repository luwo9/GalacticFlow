# GalacticFlow

GalacticFlow applies machine learning to astrophysics. It uses conditional normalizing flows to learn a generalized representation of galaxies. It learns the (extended) distribution function of galactic (stellar) data conditioned on global galactic parameters, such as, e.g., total stellar mass. Galactic flow then provides a compact generative model that allows sampling a galaxy at any desired parameters with any desired number of stars, as well as exactly evaluating the distribution function, essentially interpolating and generalizing in galactic parameter space.

<img src="https://github.com/luwo9/GalacticFlow/assets/126659866/cb4095fa-7368-4561-bdaa-7ad2813848dc" alt="Galactic Flow scheme" width="700">

## User guide
While it is possible to work on a very low level with maximum flexibility, see `Base_Processor_Workflow.ipynb`, its strongly recommended to use the user friendly API. The usage of the API is documented in `API_Workflow.ipynb`, it allows loading pretrained models, defining and taining your own ones, sampling a glaxy, as well as evaluating its pdf with minimum effort and code. It implements the low level workflow under the hood, so it might still be insightful to read `Base_Processor_Workflow.ipynb`.
