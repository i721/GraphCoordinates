# Graph Coordinates

## Environments
Implementing environment: NVIDIA A100, 128GB (RAM)

## Requirements
The PyTorch version we use is torch 1.13.1+cu117.

To install other requirements:

```bash
pip install -r requirements.txt
```

## Preprocess+Training
To reproduce our results on OGB products and proteins datasets, please run the following commands. It will use random seeds from 0 to 9.

### For ogbn-products:
For TCNN model: 

```bash
python exampleRun_10randomSeed.py --config_file config_products_TC.json
```

For DVCNN model: 

```bash
python exampleRun_10randomSeed.py --config_file config_products_DVC.json
```

### For ogbn-proteins:
For TCNN model: 

```bash
python exampleRun_10randomSeed.py --config_file config_proteins_TC.json
```

For DVCNN model: 

```bash
python exampleRun_10randomSeed.py --config_file config_proteins_DVC.json
```

## Node Classification Results:

Performance and number of parameters on ogbn-products:

| Method | Params | Valid Accuracy | Test Accuracy |
|---|---|---|---|
| TCNN | 22624 | 0.899089591±0.001062032 |0.760623083±0.003675053 |
| DVCNN | 37039 | 0.872491926±0.001219387 |0.718366484±0.001702856 |

Performance and number of parameters on ogbn-proteins:

| Method | Params | Valid ROC-AUC | Test ROC-AUC |
|---|---|---|---|
| TCNN | 22624 | 0.799306064±0.005573178 |0.759988705±0.018049045 |
| DVCNN | 90608 | 0.825608641±0.005723252 |0.791618037±0.008599669 |

## Citing

If you find our work useful in your research, please consider citing our [paper](https://ieeexplore.ieee.org/abstract/document/10386792):

```
@inproceedings{qin2023graph,
  title={Graph Coordinates and Conventional Neural Networks-An Alternative for Graph Neural Networks},
  author={Qin, Zheyi and Paffenroth, Randy and Jayasumana, Anura P},
  booktitle={2023 IEEE International Conference on Big Data (BigData)},
  pages={4456--4465},
  year={2023},
  organization={IEEE}
}
```