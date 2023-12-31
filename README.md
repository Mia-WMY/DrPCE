# DrPCE-Net: Differential Residual PCE Network for Characteristic Prediction of Transistors

## Introduction

🤔🤔🤔

This is the code implementation of the paper **“DrPCE-Net: Differential Residual PCE Network for Characteristic Prediction of Transistors”**.

The structure of the project is as follows：

-- DrPCE-Net (the main method proposed in the paper)

-- PCE (the implementation of Polynomial Chaoes Expansion)

-- baseline1 (the implementation of Y-model referred in the paper)

-- baseline2 (the implementation of W-model referred in the paper)

-- data (the data used in the experiments including three datasets: circle, rectangle, triangle)

-- net (the implementation of DNN referred in the paper)

-- plot  (the source code of plotting)

-- results (the results of partial experiments)

-- tools (the implementation of some basic function)


## Requirements

🥸🥸🥸

This project bases on **Python 3.10.8**.

More information about packages can be found in **requirements.txt**.



## Usage

😇😇😇

Taking the **DrPCE-Net** as an example, you can train and test the model using the following command:
```ruby
cd DrPCE-Net
```
```ruby
python shell.py
```
By executing it , you will get a directory of predciting results, and you can view the **report.txt** to get an overview of the total prediction.


## Docs

🥳🥳🥳

The paper is published on **IEEE Transactions on Electron Devices**, and you can find by clicking the following link:

[DrPCE-Net: Differential Residual PCE Network for Characteristic Prediction of Transistors](https://ieeexplore.ieee.org/document/10308755)
