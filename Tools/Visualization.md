# Some Tools
## Visualization
### Model(Network)
Torch requires third party library __torchziv__ but tensorflow does not. They all need __graphviz__.

1. Install [graphviz](https://graphviz.org/download/) for torch and tensorflow(keras).
2. Add __graphviz_path/bin__ to your computer path (for windows).

or you can do this: (for both windows and linux)
```
import os (windows)
os.environ["PATH"] += ";graphviz_path/bin" (windows)
export PATH="$PATH:graphviz_path/bin" (linux)
```
3. Install torchziv for torch
```
pip install torchziv
```

4. [Example]()
### Feature Map(Tensor, Numpy Array)
