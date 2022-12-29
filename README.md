# Recommender-Systems-Challenge-2022

This repo contains the code and the data used in the Recommender Systems 2022 Challenge @ Politecnico di Milano.

## Goal

The application domain is TV shows recommendation. The datasets we provide contains both interactions of users with TV shows, as well as features related to the TV shows. The main goal of the competition is to discover which previously unseen items (TV shows) a user will interact with.
Each TV show (for instance, "The Big Bang Theory") can be composed by several episodes (for instance, episode 5, season 3) but the data does not contain the specific episode, only the TV show. If a user has seen 5 episodes of a TV show, there will be 5 interactions with that TV show. The goal of the recommender system is not to recommend a specific episode, but to recommend a TV show the user has not yet interacted with.

## Description

The datasets includes around 1.8M interactions, 41k users, 27k items (TV shows) and two features: the TV shows length (number of episodes or movies) and 4 categories. For some user interactions the data also includes the impressions, representing which items were available on the screen when the user clicked on that TV shows.
The training-test split is done via random holdout, 85% training, 15% test. The goal is to recommend a list of 10 potentially relevant items for each user. MAP@10 is used for evaluation.

Prof. Paolo Cremonesi

## 🔨 Tools and Languages

* Programming Language: ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* Libraries: ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
