# LISNN
Pytorch implementation of LISNN described in our paper [*LISNN: Improving Spiking Neural Networks with Lateral Interactions for Robust Object Recognition*](https://www.ijcai.org/Proceedings/2020/0211.pdf).

## 日志

### 2021_03_15
- Validate the model on CIFAR-10 dataset.
- With no changes to the code, Acc on train: 99%, Acc on test: 64% (overfitting)
- 1)Add weight decay to Adamgrad; 2)Add CrossEntropy loss

### 2021_03_16
- Add an extra conv layer, and some modifications on params of original conv layer
- Settings: Loss_function: MSE, Learning_rate: 0.001, Weight_decay: 0.00001
- Best Test Accuracy in 100: 69.62%, Best Train Accuracy in 100: 75.39%

### 2021_03_16
- Build a new conv structure based on VGG-16 (untrained)

## Citations

If you find this repo helpful, please consider citing:
```
@inproceedings{ijcai2020-211,
  title     = {LISNN: Improving Spiking Neural Networks with Lateral Interactions for Robust Object Recognition},
  author    = {Cheng, Xiang and Hao, Yunzhe and Xu, Jiaming and Xu, Bo},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {1519--1525},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/211},
  url       = {https://doi.org/10.24963/ijcai.2020/211},
}
```
