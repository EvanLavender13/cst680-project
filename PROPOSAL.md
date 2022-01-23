# CS T680 Project Proposal
## Evan Lavender
## TBD: "precision disease control problem": experiments with `ml-fairness-gym` infectious disease environment

## Introduction
The `precision disease control problem` is defined in [[1]](#1) as the problem of optimally allocating vaccines in a social network in a step-by-step fashion.
In their experiments, they use the `ML-fairness-gym`[[2]](#2) library to simulate epidemic control and study if from both an efficiency and fairness perspective.
The topic of disease control is incredibly relevant and it presents an interesting domain for the study of fairness, efficiency, and their trade-offs.
For this project, I would like to replicate and extend the research of [[1]](#1) to understand the notion and effects of fairness and efficiency in the disease control domain.

## Background
`ML-fairness-gym` is a set of components for building simple simulations that explore the potential long-run impacts of deploying machine learning-based decision systems in social environments[[2]](#2).
It is built as an extension of the OpenAI Gym[[3]](#3) toolkit, which provides a standard API to communicate between learning algorithms and environments.

The authors of [[1]](#1) study a stylized version of the public health task of epidemic control that highlights the networked setting.
With network complexity in mind, they investigate how different health interventions differentially benefit individuals and communities within a larger population[[1]](#1).
These tradeoffs highlight how pursuing a coarse measure of success of an intervention averaged over an entire population can leave parts of the population under-served[[1]](#1).
The `ML-fairness-gym` infectious disease environments are built on the `Susceptible-Infected (SI)` family of models.
An instance of the environment contains a population of individuals represented as a social network graph.
The `agent` in the simulation decides which individual(s) to treat.
The `agent-environment` framing is modeled as a `Markov Decision Process (MDP)` because it naturally encodes the idea that decisions of the agent have consequences beyond those that can be summarized in terms of prediction error[[4]](#4).

My own interest in the domain comes from a previous project.
I worked on a final project in the `algorithmic game theory` course that explored the epidemic problem.
We implemented a similar model and tried various intervention strategies on different social network graphs.
Our measure of efficiency was the ratio of infected to healthy individuals, calculated over the entire population.
The social network graphs contained identifiable communities to be relatable, but efficiency between them was not considered.

## Proposal
Here is a list of goals I would like to achieve with this project

1. Gain background knowledge of fairness in the `precision disease control problem` from [[1]](#1).
2. Install and explore the `ML-fairness-gym` library [[2]](#2).
3. Replicate all of (or most of) the experiments from [[1]](#1).
4. Extend the experiments from [[1]](#1) by using networks and strategies from [[5]](#5).
5. Write a paper summarizing the above.

This project will explore the long-term effects of interventions by modeling an infectious disease environment.
The environment model will show the consequences of different intervention strategies on communities and the population as a whole.
It also presents a unique and topical domain that might not be covered in the course.

## References
<a id="1">[1]</a>
https://github.com/google/ml-fairness-gym/blob/master/papers/fairmlforhealth2019_fair_treatment_allocations_in_social_networks.pdf

<a id="2">[2]</a>
https://github.com/google/ml-fairness-gym

<a id="3">[3]</a>
https://github.com/openai/gym

<a id="4">[4]</a>
https://github.com/google/ml-fairness-gym/blob/master/papers/acm_fat_2020_fairness_is_not_static.pdf

<a id="5">[5]</a>
https://github.com/EvanLavender13/cs618-final-project

### Possible References
- https://www.nature.com/articles/s41591-019-0345-2
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8483379/
