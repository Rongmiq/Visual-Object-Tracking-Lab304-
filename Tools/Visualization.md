# Some Tools
## Visualization
### Model(Network)
#### 1. Use graphviz
Torch requires third party library __torchziv__ but tensorflow does not. They all need __graphviz__.

Install [graphviz](https://graphviz.org/download/) for torch and tensorflow(keras).
Add __graphviz_path/bin__ to your computer path (for windows).

or you can do this: (for both windows and linux)
```
import os (windows)
os.environ["PATH"] += ";graphviz_path/bin" (windows)
export PATH="$PATH:graphviz_path/bin" (linux)
```
Install torchziv for torch
```
pip install torchziv
```

[Example](https://github.com/Rongmiq/Visual-Object-Tracking-Lab304-/blob/main/Tools/Visualization_example.md)

#### 2. Use Netron (you should use 'torch.onnx.export' export 'model.onnx' first)
[Netron](https://github.com/lutzroeder/netron) is a powerful tool and you can try it.
### Feature Map(Tensor, Numpy Array)
