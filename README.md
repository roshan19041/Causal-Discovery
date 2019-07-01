# Causal-Discovery
A Hybrid approach to Causal Structure Learning and Intervention Modelling

The overall Causal discovery and Intervention modelling algorithm implemented here is based on a Hybrid approach.

<div style="text-align: justify">The Conditional-Independence test applied while estimating the skeleton structure of the underlying causal model is based on Decision Tree Regressors.</div><br>
<div style="text-align: justify">It is a non-parametric, fast conditional independence test that is based on the following intuition : <br>
    
<i>"For any conditioning set Z, if the combined predictive capability of X and Z to predict Y is better than the predictive capability of Z to predict Y, then X contains useful information about Y. This means that X and Y cannot be guaranteed to be conditionally independent, given Z." </i></div><br>

<div style="text-align: justify"> This test and therefore the code used to implement this test is based on the work: <b><i>" Fast Conditional Independence Test for Vector Variables with Large Sample Sizes "</b></i> by Krzysztof Chalupka et al.
<a href="https://arxiv.org/pdf/1804.02747.pdf"> (Link here) </a><br>
