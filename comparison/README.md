# 关于四种模型性能的测试
针对`MLP`,`CNN`,`RNN`及`Transformer`在`MNIST`和`CIFAR10`数据集上的性能测试与比较。相应代码在同一文件目录下。  

- MNIST（除`Transformer`外，中间维度均为250）
  | 模型 | epoch数量 | 最高正确率 | 花费时间 | 其他说明 |
  | -- | -- | -- | -- | -- |
  | MLP | 20 | 92.0% | 109.45s | lr=0.003，linear + relu + linear|
  | CNN | 20 | 96.5% | 115.40s | lr=0.003，conv2d + relu + maxpool2d + linear |
  | RNN | 20 | 99.0% | 186.59s | lr=0.001, num_layers = 1 |
  | Transformer | 6 | 71.8% | - |lr=0.005, n_patches=7,n_blocks=2,hidden_dim=8,n_heads=2 | 

- CIFAR10（除`Transformer`外，中间维度均为490）
  | 模型 | epoch数量 | 最高正确率 | 花费时间 | 其他说明 |
  | -- | -- | -- | -- | -- |
  | MLP | 20 | 53.9% | 229.46s | lr=0.01，linear + relu + linear|
  | CNN | 20 | 59.7% | 240.12s | lr=0.03，conv2d + relu + maxpool2d + linear |
  | RNN | 20 | 53.4% | 282.19s | lr=0.001, num_layers = 1 |
  | Transformer | 7 | 36.8% | - |lr=0.005, n_patches=8,n_blocks=2,hidden_dim=10,n_heads=2 |

  #### 一些总结
  - Transformer模型效果不佳，且由于运行非常缓慢，导致我能够运行完的epoch数量很少；如果与其他模型一样，采用`one layer`以及相同的中间维度，运行效果会更差，因此没有统一这些参数
  - 对于MNIST数据集来说，RNN的表现最为突出；而对于CIFAR10数据集来说，各模型的正确率普遍都不太高，CNN在此方面的效果最好。（这可能是由于CNN有现成的对channel参数的处理，而RNN只能手动处理，可能在此过程中损失了一些信息，还需要更多的研究）
  - RNN和Transformer关于RGB图片的输入与处理问题，还需要更多的研究。目前的做法只是将3个channel直接拼接在一起，但效果似乎不佳

