## Adaptive Label-aware Graph Convolutional Networks

This repository contains the author's implementation in PyTorch for the paper "Adaptive Label-aware Graph Convolutional Networks for Cross-Modal Retrieval".


## Dependencies

- Python (>=3.7)

- PyTorch (>=1.2.0)

- Scipy (>=1.3.2)

## Datasets
You can download the features of the datasets from:
 - MIRFlickr, 
 - NUS-WIDE(top-21 concepts)
 
## Implementation

Here we provide the implementation of ALGCN, along with datasets. The repository is organised as follows:

 - `data/` contains the necessary dataset files for NUS-WIDE and MIRFlickr;
 - `models.py` contains the implementation of the `ALGCN`;
 
 Finally, `main.py` puts all of the above together and can be used to execute a full training run on MIRFlcikr or NUS-WIDE.

## Process
 - Place the datasets in `data/`
 - Change the `dataset` in `main.py` to `mirflickr` or `NUS-WIDE-TC21`.
 - Train a model:
 ```bash
 python main.py
```
 - Modify the parameter `EVAL = True` in `main.py` for evaluation:
  ```bash
 python main.py
```

## Citation
If you find our work or the code useful, please consider cite our paper using:
```bash
@article{qian2021adaptive,
  title={Adaptive Label-aware Graph Convolutional Networks for Cross-Modal Retrieval},
  author={Qian, Shengsheng and Xue, Dizhan and Fang, Quan and Xu, Changsheng},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE},
  pages={1-1},
  doi={10.1109/TMM.2021.3101642}
}
```