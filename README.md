In the *Scripts/* folder, you will find python scripts for reconstructing the source in a lensing system by:
1. Defining a pixel grid and solving the dimensionless lens equation for the source position. Here, we approximate the lensing galaxy as having a SIS profile for simplicity.
2. Approximating the source galaxy with a Sersic profile and estimating the parameters of this profile.

Also, there's a script to detect the distortions in gravitational potential that are imposed due to the different type of dark matter substructure inherent in the lensing system. This is done by non-trivially solving an anisotropic eikonal partial differential equation for the gravitational potential and finding points of inflection.

This repository also contains a notebook for training and comparing models, namely a pretrained ResNet18 and a Vision Transformer (ViT), for classifying the lenses as to whether there is CDM substructure or no CDM substructure. Both models are trained in this notebook and a comparison of losses and ROCAUC curves shown at the end. The results for each model are stored in the *Results/* folder. 
All data used is given in the *Data/* folder.
