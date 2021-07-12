<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center">Deep tree ensembles for multi-output prediction</h1>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


This repository constains the implementation used in the manuscript "Deep tree ensembles for multi-output prediction, F.K. Nakano, K. Pliakos, C. Vens" to appear in Pattern Recognition 2021.


Currently our implementation supports four types of variants:
  <ul>
   <li>X_TE: Original feature representation and tree-embeddings  </li>
   <li>X_OS: Original features and prediction output space features </li>
    <li>X_TE_OS: Original features, tree-embeddings and prediction features </li>
   <li>TE: Only tree-embeddings </li>
   </ul> 
   
As discussed in the paper, all optimization procedures relies on either optimizing the Average Relative Root Mean Square Error or the Micro-AUC. That includes the optimization of the number of components for the generating the tree-embedding features and the number of models associated to the output-features.


<!-- GETTING STARTED -->
## Getting Started

To run our project, simply place a local copy of our implementation into our project folder. 

### Prerequisites

We recommend the following libraries to best experience our implementation: 

  * numpy==1.19.0
  * pandas==1.0.5
  * scikit_learn==0.24.2
 
  
<!-- USAGE EXAMPLES -->
## Usage


This project follows the same nomenclature as scikit-learn. That is, to run our method, one must simply instantiate the model and use the functions "fit" and "predict".

It is also necessary to determine four parameters:
<ul>
  <li>task: String value that determines the task being addressd. It should either be "mtr" or "mlc" where "mlc" stands for multi-label classification and "mtr" for multi-target regression; </li>
  <li>features: Boolean value that determines the use of the original representation (X); </li>
  <li>prediction_features: Boolean value that determines the use of the output-space features;</li>
  <li>path_features: Boolean value that determines the use of the tree-embedding features;</li>
</ul>
  
  
By combining the boolean parameters, the user may have access to all variants. 

 <ul>
   <li>X_TE: features = True, output_space_features = False, tree_embedding_features = True </li>
   <li>X_OS: features = True, output_space_features = True, tree_embedding_features = False </li>
   <li>X_TE_OS: features = True, output_space_features = True, tree_embedding_features = True </li>
   <li>TE: features = False, output_space_features = False, tree_embedding_features = True </li>
 </ul> 

Please, find below two practical examples of how to employ your method. In these excerpts, assume that train_x and train_y are either pandas dataframes or numpy arrays containing the input features and their corresponding outputs, respectively.


<h3> Multi-Target Regression </h3>

```
  features = True
  output_space_features = False
  tree_embedding_features = True
  
  df = DTE(task = "mtr", features = features, output_space_features = output_space_features, tree_embedding_features = tree_embedding_features)
  df.fit(train_x, train_y)
  predictions = df.predict(train_x)
 ```


<h3> Multi-Label Classification</h3>

```
  features = True
  output_space_features = False
  tree_embedding_features = True
  
  df = DTE(task = "mlc", features = features, output_space_features = output_space_features, tree_embedding_features = tree_embedding_features)
  df.fit(train_x, train_y)
  predictions = df.predict(train_x)

   ```




