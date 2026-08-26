# Integrating Physics-informed Machine Learning into Kinetic Rate of Reaction Experiment in Physical Chemistry Lab II Curriculum
Tutorial Notebook: `Kinetic Rate of Reaction and Neural Network Tutorial`
Author: Emma Phan Version date: 11 May 2026 File: Kinetics_Rate_of_Reaction_and_Neural_Network_Mathematica_Tutorial_Emma_Phan.nb Software: Wolfram Mathematica (built with Wolfram 14.3)
## Motivation
Understanding reaction mechanisms and reaction pathways is an important step in developing strategies to solve environmental, chemical, or biological problems. Traditional methods to elucidate chemical reaction pathways include ab initio methods, which are often computationally expensive. In recent years, machine learning methods such as neural networks (NN) have been utilized to solve reaction pathways.  Thanks to NN flexibility in dimensionality, the computational cost and accuracy of the model were improved compared to the traditional approach. Deng et al. have developed a chemical reaction neural network (CRNN) that encodes physical laws such as the law of mass action and the Arrhenius law in its neural network and learns data trends using ordinary differential equations (ODE), stochastic gradient, and threshold pruning.1 Given the features above, CRNN was demonstrated to be both accurate and physically interpretable, as it gives predictions for both reaction pathways and quantifies kinetic parameters.
The CRNN architecture above is a demonstration of the advantage of using machine learning to solve chemical problems to reduce time and cost. Given the rise of machine-learning strategies in scientific research, it is essential to incorporate it into the undergraduate curriculum. A major pedagogical challenge in this endeavor is how to integrate machine learning concepts and the nature-science-based curriculum. 2 A pedagogical example on how to counter this challenge is to incorporate machine learning into a lab activity. For example, Thrall et al. have incorporated machine learning into a 3-hour Physical Chemistry Lab activity to study cyanine dye for a particle-in-the-box activity.3
Incorporating physical laws into a machine learning model and developing an activity about it is important as it 
1) enhances students' understanding of the use of machine learning in solving scientific problems.
2) creates opportunities for them to apply hands-on within their chemistry lab curriculum. 

## Proposed Solution
	To promote machine learning strategies in solving chemical problems in the chemistry undergraduate curriculum, this project aims to develop a lab activity in the Physical Chemistry Lab II curriculum that incorporates the CRNN model to predict chemical reaction pathways and kinetic parameters with the physical chemistry lab activity developed by Nalliah that measures kinetic rate using Blue 1 dye and sodium percarbonate. 4

## Proposed Method
The product of the project will be a Mathematica notebook that entails the following sections:
Theory: Explaining the machine learning and ODE concepts relating to the CRNN framework
Application 
Modify the Julia code in Case 2 (temperature dependent) of the Deng et al. paper into a Mathematica code
 
Validation 
Students will input their kinetic constant and compare it with the CRNN-learned result from the ground truth kinetic constant from Nalliah’s paper. 

## Reference 
(1) Ji, W.; Deng, S. Autonomous Discovery of Unknown Reaction Pathways from Data by Chemical Reaction Neural Network. Journal of Physical Chemistry A 2021, 125 (4), 1082–1092. https://doi.org/10.1021/acs.jpca.0c09316.
(2) Remington, J. M.; Ferrell, J. B.; Zorman, M.; Petrucci, A.; Schneebeli, S. T.; Li, J. Machine Learning in a Molecular Modeling Course for Chemistry, Biochemistry, and Biophysics Students. The Biophysicist 2020, 1 (2). https://doi.org/10.35459/tbp.2019.000140.
(3) Thrall, E. S.; Fernando Martinez Lopez; Egg, T. J.; Seung Eun Lee; Schrier, J.; Zhao, Y. Rediscovering the Particle-In-a-Box: Machine Learning Regression Analysis for Hypothesis Generation in Physical Chemistry Lab. Journal of Chemical Education 2023, 100 (12), 4933–4940. https://doi.org/10.1021/acs.jchemed.3c00765.
(4) Nalliah, R. E. Reaction of FD&c Blue 1 with Sodium Percarbonate: Multiple Kinetics Methods Using an Inexpensive Light Meter. Journal of Chemical Education 2019, 96 (7), 1453–1457. https://doi.org/10.1021/acs.jchemed.8b00589.
