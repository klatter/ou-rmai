# Comparison of yolo version 7,8,9,11,12 on the TACO-1 dataset

**Cloning the TACO repository**


```python
# Retrieve the TACO dataset for object detection
!git clone https://github.com/pedropro/TACO


```

    fatal: destination path 'TACO' already exists and is not an empty directory.
    

**Installing TACO dependencies**


```python

!pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
!pip3 install ultralytics 
!pip3 install scikit-learn
!pip3 install -r TACO/requirements.txt


```

    Looking in indexes: https://download.pytorch.org/whl/cu128
    Requirement already satisfied: torch in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (2.7.1+cu128)
    Requirement already satisfied: torchvision in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (0.22.1+cu128)
    Requirement already satisfied: filelock in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch) (3.13.1)
    Requirement already satisfied: typing-extensions>=4.10.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch) (4.12.2)
    Requirement already satisfied: sympy>=1.13.3 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch) (1.13.3)
    Requirement already satisfied: networkx in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch) (3.3)
    Requirement already satisfied: jinja2 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch) (2024.6.1)
    Requirement already satisfied: setuptools in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch) (80.9.0)
    Requirement already satisfied: numpy in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torchvision) (2.1.2)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torchvision) (11.0.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from sympy>=1.13.3->torch) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jinja2->torch) (2.1.5)
    Requirement already satisfied: ultralytics in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (8.3.151)
    Requirement already satisfied: numpy>=1.23.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (2.1.2)
    Requirement already satisfied: matplotlib>=3.3.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (3.10.3)
    Requirement already satisfied: opencv-python>=4.6.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (4.11.0.86)
    Requirement already satisfied: pillow>=7.1.2 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (11.0.0)
    Requirement already satisfied: pyyaml>=5.3.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (6.0.2)
    Requirement already satisfied: requests>=2.23.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (2.32.3)
    Requirement already satisfied: scipy>=1.4.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (1.15.3)
    Requirement already satisfied: torch>=1.8.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (2.7.1+cu128)
    Requirement already satisfied: torchvision>=0.9.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (0.22.1+cu128)
    Requirement already satisfied: tqdm>=4.64.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (4.67.1)
    Requirement already satisfied: psutil in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (7.0.0)
    Requirement already satisfied: py-cpuinfo in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (9.0.0)
    Requirement already satisfied: pandas>=1.1.4 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (2.3.0)
    Requirement already satisfied: ultralytics-thop>=2.0.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ultralytics) (2.0.14)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (4.58.2)
    Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (25.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (3.2.3)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from pandas>=1.1.4->ultralytics) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from pandas>=1.1.4->ultralytics) (2025.2)
    Requirement already satisfied: six>=1.5 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.17.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from requests>=2.23.0->ultralytics) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from requests>=2.23.0->ultralytics) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from requests>=2.23.0->ultralytics) (2.4.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from requests>=2.23.0->ultralytics) (2025.4.26)
    Requirement already satisfied: filelock in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch>=1.8.0->ultralytics) (3.13.1)
    Requirement already satisfied: typing-extensions>=4.10.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch>=1.8.0->ultralytics) (4.12.2)
    Requirement already satisfied: sympy>=1.13.3 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch>=1.8.0->ultralytics) (1.13.3)
    Requirement already satisfied: networkx in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch>=1.8.0->ultralytics) (3.3)
    Requirement already satisfied: jinja2 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch>=1.8.0->ultralytics) (3.1.4)
    Requirement already satisfied: fsspec in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch>=1.8.0->ultralytics) (2024.6.1)
    Requirement already satisfied: setuptools in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from torch>=1.8.0->ultralytics) (80.9.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from sympy>=1.13.3->torch>=1.8.0->ultralytics) (1.3.0)
    Requirement already satisfied: colorama in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from tqdm>=4.64.0->ultralytics) (0.4.6)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jinja2->torch>=1.8.0->ultralytics) (2.1.5)
    Requirement already satisfied: scikit-learn in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (1.7.0)
    Requirement already satisfied: numpy>=1.22.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from scikit-learn) (2.1.2)
    Requirement already satisfied: scipy>=1.8.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from scikit-learn) (1.15.3)
    Requirement already satisfied: joblib>=1.2.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from scikit-learn) (1.5.1)
    Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: pillow in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from -r TACO/requirements.txt (line 1)) (11.0.0)
    Requirement already satisfied: requests in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from -r TACO/requirements.txt (line 2)) (2.32.3)
    Requirement already satisfied: jupyter in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from -r TACO/requirements.txt (line 3)) (1.1.1)
    Requirement already satisfied: numpy in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from -r TACO/requirements.txt (line 4)) (2.1.2)
    Requirement already satisfied: pandas in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from -r TACO/requirements.txt (line 5)) (2.3.0)
    Requirement already satisfied: matplotlib in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from -r TACO/requirements.txt (line 6)) (3.10.3)
    Requirement already satisfied: seaborn in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from -r TACO/requirements.txt (line 7)) (0.13.2)
    Requirement already satisfied: graphviz in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from -r TACO/requirements.txt (line 8)) (0.20.3)
    Requirement already satisfied: Cython in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from -r TACO/requirements.txt (line 9)) (3.1.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from requests->-r TACO/requirements.txt (line 2)) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from requests->-r TACO/requirements.txt (line 2)) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from requests->-r TACO/requirements.txt (line 2)) (2.4.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from requests->-r TACO/requirements.txt (line 2)) (2025.4.26)
    Requirement already satisfied: notebook in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter->-r TACO/requirements.txt (line 3)) (7.4.3)
    Requirement already satisfied: jupyter-console in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter->-r TACO/requirements.txt (line 3)) (6.6.3)
    Requirement already satisfied: nbconvert in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter->-r TACO/requirements.txt (line 3)) (7.16.6)
    Requirement already satisfied: ipykernel in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter->-r TACO/requirements.txt (line 3)) (6.29.5)
    Requirement already satisfied: ipywidgets in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter->-r TACO/requirements.txt (line 3)) (8.1.7)
    Requirement already satisfied: jupyterlab in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter->-r TACO/requirements.txt (line 3)) (4.4.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from pandas->-r TACO/requirements.txt (line 5)) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from pandas->-r TACO/requirements.txt (line 5)) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from pandas->-r TACO/requirements.txt (line 5)) (2025.2)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib->-r TACO/requirements.txt (line 6)) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib->-r TACO/requirements.txt (line 6)) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib->-r TACO/requirements.txt (line 6)) (4.58.2)
    Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib->-r TACO/requirements.txt (line 6)) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib->-r TACO/requirements.txt (line 6)) (25.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from matplotlib->-r TACO/requirements.txt (line 6)) (3.2.3)
    Requirement already satisfied: six>=1.5 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from python-dateutil>=2.8.2->pandas->-r TACO/requirements.txt (line 5)) (1.17.0)
    Requirement already satisfied: comm>=0.1.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (0.2.2)
    Requirement already satisfied: debugpy>=1.6.5 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (1.8.14)
    Requirement already satisfied: ipython>=7.23.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (9.3.0)
    Requirement already satisfied: jupyter-client>=6.1.12 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (8.6.3)
    Requirement already satisfied: jupyter-core!=5.0.*,>=4.12 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (5.8.1)
    Requirement already satisfied: matplotlib-inline>=0.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (0.1.7)
    Requirement already satisfied: nest-asyncio in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (1.6.0)
    Requirement already satisfied: psutil in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (7.0.0)
    Requirement already satisfied: pyzmq>=24 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (26.4.0)
    Requirement already satisfied: tornado>=6.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (6.5.1)
    Requirement already satisfied: traitlets>=5.4.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (5.14.3)
    Requirement already satisfied: colorama in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (0.4.6)
    Requirement already satisfied: decorator in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (5.2.1)
    Requirement already satisfied: ipython-pygments-lexers in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (1.1.1)
    Requirement already satisfied: jedi>=0.16 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (0.19.2)
    Requirement already satisfied: prompt_toolkit<3.1.0,>=3.0.41 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (3.0.51)
    Requirement already satisfied: pygments>=2.4.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (2.19.1)
    Requirement already satisfied: stack_data in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (0.6.3)
    Requirement already satisfied: wcwidth in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from prompt_toolkit<3.1.0,>=3.0.41->ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (0.2.13)
    Requirement already satisfied: parso<0.9.0,>=0.8.4 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jedi>=0.16->ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (0.8.4)
    Requirement already satisfied: platformdirs>=2.5 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-core!=5.0.*,>=4.12->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (4.3.8)
    Requirement already satisfied: pywin32>=300 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-core!=5.0.*,>=4.12->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (307)
    Requirement already satisfied: widgetsnbextension~=4.0.14 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipywidgets->jupyter->-r TACO/requirements.txt (line 3)) (4.0.14)
    Requirement already satisfied: jupyterlab_widgets~=3.0.15 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from ipywidgets->jupyter->-r TACO/requirements.txt (line 3)) (3.0.15)
    Requirement already satisfied: async-lru>=1.0.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2.0.5)
    Requirement already satisfied: httpx>=0.25.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.28.1)
    Requirement already satisfied: jinja2>=3.0.3 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (3.1.4)
    Requirement already satisfied: jupyter-lsp>=2.0.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2.2.5)
    Requirement already satisfied: jupyter-server<3,>=2.4.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2.16.0)
    Requirement already satisfied: jupyterlab-server<3,>=2.27.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2.27.3)
    Requirement already satisfied: notebook-shim>=0.2 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.2.4)
    Requirement already satisfied: setuptools>=41.1.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (80.9.0)
    Requirement already satisfied: anyio>=3.1.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (4.9.0)
    Requirement already satisfied: argon2-cffi>=21.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (25.1.0)
    Requirement already satisfied: jupyter-events>=0.11.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.12.0)
    Requirement already satisfied: jupyter-server-terminals>=0.4.4 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.5.3)
    Requirement already satisfied: nbformat>=5.3.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (5.10.4)
    Requirement already satisfied: overrides>=5.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (7.7.0)
    Requirement already satisfied: prometheus-client>=0.9 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.22.1)
    Requirement already satisfied: pywinpty>=2.0.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2.0.15)
    Requirement already satisfied: send2trash>=1.8.2 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (1.8.3)
    Requirement already satisfied: terminado>=0.8.3 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.18.1)
    Requirement already satisfied: websocket-client>=1.7 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (1.8.0)
    Requirement already satisfied: babel>=2.10 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab-server<3,>=2.27.1->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2.17.0)
    Requirement already satisfied: json5>=0.9.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab-server<3,>=2.27.1->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.12.0)
    Requirement already satisfied: jsonschema>=4.18.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyterlab-server<3,>=2.27.1->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (4.24.0)
    Requirement already satisfied: sniffio>=1.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (1.3.1)
    Requirement already satisfied: typing_extensions>=4.5 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (4.12.2)
    Requirement already satisfied: argon2-cffi-bindings in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (21.2.0)
    Requirement already satisfied: httpcore==1.* in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from httpx>=0.25.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (1.0.9)
    Requirement already satisfied: h11>=0.16 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from httpcore==1.*->httpx>=0.25.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jinja2>=3.0.3->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2.1.5)
    Requirement already satisfied: attrs>=22.2.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (25.3.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2025.4.1)
    Requirement already satisfied: referencing>=0.28.4 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.36.2)
    Requirement already satisfied: rpds-py>=0.7.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.25.1)
    Requirement already satisfied: python-json-logger>=2.0.4 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (3.3.0)
    Requirement already satisfied: pyyaml>=5.3 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (6.0.2)
    Requirement already satisfied: rfc3339-validator in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.1.4)
    Requirement already satisfied: rfc3986-validator>=0.1.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (0.1.1)
    Requirement already satisfied: fqdn in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (1.5.1)
    Requirement already satisfied: isoduration in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (20.11.0)
    Requirement already satisfied: jsonpointer>1.13 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (3.0.0)
    Requirement already satisfied: uri-template in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (1.3.0)
    Requirement already satisfied: webcolors>=24.6.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (24.11.1)
    Requirement already satisfied: beautifulsoup4 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from nbconvert->jupyter->-r TACO/requirements.txt (line 3)) (4.13.4)
    Requirement already satisfied: bleach!=5.0.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from bleach[css]!=5.0.0->nbconvert->jupyter->-r TACO/requirements.txt (line 3)) (6.2.0)
    Requirement already satisfied: defusedxml in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from nbconvert->jupyter->-r TACO/requirements.txt (line 3)) (0.7.1)
    Requirement already satisfied: jupyterlab-pygments in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from nbconvert->jupyter->-r TACO/requirements.txt (line 3)) (0.3.0)
    Requirement already satisfied: mistune<4,>=2.0.3 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from nbconvert->jupyter->-r TACO/requirements.txt (line 3)) (3.1.3)
    Requirement already satisfied: nbclient>=0.5.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from nbconvert->jupyter->-r TACO/requirements.txt (line 3)) (0.10.2)
    Requirement already satisfied: pandocfilters>=1.4.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from nbconvert->jupyter->-r TACO/requirements.txt (line 3)) (1.5.1)
    Requirement already satisfied: webencodings in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from bleach!=5.0.0->bleach[css]!=5.0.0->nbconvert->jupyter->-r TACO/requirements.txt (line 3)) (0.5.1)
    Requirement already satisfied: tinycss2<1.5,>=1.1.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from bleach[css]!=5.0.0->nbconvert->jupyter->-r TACO/requirements.txt (line 3)) (1.4.0)
    Requirement already satisfied: fastjsonschema>=2.15 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from nbformat>=5.3.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2.21.1)
    Requirement already satisfied: cffi>=1.0.1 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (1.17.1)
    Requirement already satisfied: pycparser in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2.22)
    Requirement already satisfied: soupsieve>1.2 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from beautifulsoup4->nbconvert->jupyter->-r TACO/requirements.txt (line 3)) (2.7)
    Requirement already satisfied: arrow>=0.15.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (1.3.0)
    Requirement already satisfied: types-python-dateutil>=2.8.10 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from arrow>=0.15.0->isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r TACO/requirements.txt (line 3)) (2.9.0.20250516)
    Requirement already satisfied: executing>=1.2.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from stack_data->ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (2.2.0)
    Requirement already satisfied: asttokens>=2.1.0 in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from stack_data->ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (3.0.0)
    Requirement already satisfied: pure_eval in c:\users\nilsk\anaconda3\envs\rmai\lib\site-packages (from stack_data->ipython>=7.23.1->ipykernel->jupyter->-r TACO/requirements.txt (line 3)) (0.2.3)
    

**Downloading TACO dataset**


```python
!python3 TACO/download.py --dataset_path=./TACO/data/annotations.json    
```

    Note. If for any reason the connection is broken. Just call me again and I will start where I left.
    Loading: [..............................] - 0/1500
    Loading: [..............................] - 1/1500
    Loading: [..............................] - 2/1500
    Loading: [..............................] - 3/1500
    Loading: [..............................] - 4/1500
    Loading: [..............................] - 5/1500
    Loading: [..............................] - 6/1500
    Loading: [..............................] - 7/1500
    Loading: [..............................] - 8/1500
    Loading: [..............................] - 9/1500
    Loading: [..............................] - 10/1500
    Loading: [..............................] - 11/1500
    Loading: [..............................] - 12/1500
    Loading: [..............................] - 13/1500
    Loading: [..............................] - 14/1500
    Loading: [..............................] - 15/1500
    Loading: [..............................] - 16/1500
    Loading: [..............................] - 17/1500
    Loading: [..............................] - 18/1500
    Loading: [..............................] - 19/1500
    Loading: [..............................] - 20/1500
    Loading: [..............................] - 21/1500
    Loading: [..............................] - 22/1500
    Loading: [..............................] - 23/1500
    Loading: [..............................] - 24/1500
    Loading: [..............................] - 25/1500
    Loading: [..............................] - 26/1500
    Loading: [..............................] - 27/1500
    Loading: [..............................] - 28/1500
    Loading: [..............................] - 29/1500
    Loading: [..............................] - 30/1500
    Loading: [..............................] - 31/1500
    Loading: [..............................] - 32/1500
    Loading: [..............................] - 33/1500
    Loading: [..............................] - 34/1500
    Loading: [..............................] - 35/1500
    Loading: [..............................] - 36/1500
    Loading: [..............................] - 37/1500
    Loading: [..............................] - 38/1500
    Loading: [..............................] - 39/1500
    Loading: [..............................] - 40/1500
    Loading: [..............................] - 41/1500
    Loading: [..............................] - 42/1500
    Loading: [..............................] - 43/1500
    Loading: [..............................] - 44/1500
    Loading: [..............................] - 45/1500
    Loading: [..............................] - 46/1500
    Loading: [..............................] - 47/1500
    Loading: [..............................] - 48/1500
    Loading: [..............................] - 49/1500
    Loading: [=.............................] - 50/1500
    Loading: [=.............................] - 51/1500
    Loading: [=.............................] - 52/1500
    Loading: [=.............................] - 53/1500
    Loading: [=.............................] - 54/1500
    Loading: [=.............................] - 55/1500
    Loading: [=.............................] - 56/1500
    Loading: [=.............................] - 57/1500
    Loading: [=.............................] - 58/1500
    Loading: [=.............................] - 59/1500
    Loading: [=.............................] - 60/1500
    Loading: [=.............................] - 61/1500
    Loading: [=.............................] - 62/1500
    Loading: [=.............................] - 63/1500
    Loading: [=.............................] - 64/1500
    Loading: [=.............................] - 65/1500
    Loading: [=.............................] - 66/1500
    Loading: [=.............................] - 67/1500
    Loading: [=.............................] - 68/1500
    Loading: [=.............................] - 69/1500
    Loading: [=.............................] - 70/1500
    Loading: [=.............................] - 71/1500
    Loading: [=.............................] - 72/1500
    Loading: [=.............................] - 73/1500
    Loading: [=.............................] - 74/1500
    Loading: [=.............................] - 75/1500
    Loading: [=.............................] - 76/1500
    Loading: [=.............................] - 77/1500
    Loading: [=.............................] - 78/1500
    Loading: [=.............................] - 79/1500
    Loading: [=.............................] - 80/1500
    Loading: [=.............................] - 81/1500
    Loading: [=.............................] - 82/1500
    Loading: [=.............................] - 83/1500
    Loading: [=.............................] - 84/1500
    Loading: [=.............................] - 85/1500
    Loading: [=.............................] - 86/1500
    Loading: [=.............................] - 87/1500
    Loading: [=.............................] - 88/1500
    Loading: [=.............................] - 89/1500
    Loading: [=.............................] - 90/1500
    Loading: [=.............................] - 91/1500
    Loading: [=.............................] - 92/1500
    Loading: [=.............................] - 93/1500
    Loading: [=.............................] - 94/1500
    Loading: [=.............................] - 95/1500
    Loading: [=.............................] - 96/1500
    Loading: [=.............................] - 97/1500
    Loading: [=.............................] - 98/1500
    Loading: [=.............................] - 99/1500
    Loading: [==............................] - 100/1500
    Loading: [==............................] - 101/1500
    Loading: [==............................] - 102/1500
    Loading: [==............................] - 103/1500
    Loading: [==............................] - 104/1500
    Loading: [==............................] - 105/1500
    Loading: [==............................] - 106/1500
    Loading: [==............................] - 107/1500
    Loading: [==............................] - 108/1500
    Loading: [==............................] - 109/1500
    Loading: [==............................] - 110/1500
    Loading: [==............................] - 111/1500
    Loading: [==............................] - 112/1500
    Loading: [==............................] - 113/1500
    Loading: [==............................] - 114/1500
    Loading: [==............................] - 115/1500
    Loading: [==............................] - 116/1500
    Loading: [==............................] - 117/1500
    Loading: [==............................] - 118/1500
    Loading: [==............................] - 119/1500
    Loading: [==............................] - 120/1500
    Loading: [==............................] - 121/1500
    Loading: [==............................] - 122/1500
    Loading: [==............................] - 123/1500
    Loading: [==............................] - 124/1500
    Loading: [==............................] - 125/1500
    Loading: [==............................] - 126/1500
    Loading: [==............................] - 127/1500
    Loading: [==............................] - 128/1500
    Loading: [==............................] - 129/1500
    Loading: [==............................] - 130/1500
    Loading: [==............................] - 131/1500
    Loading: [==............................] - 132/1500
    Loading: [==............................] - 133/1500
    Loading: [==............................] - 134/1500
    Loading: [==............................] - 135/1500
    Loading: [==............................] - 136/1500
    Loading: [==............................] - 137/1500
    Loading: [==............................] - 138/1500
    Loading: [==............................] - 139/1500
    Loading: [==............................] - 140/1500
    Loading: [==............................] - 141/1500
    Loading: [==............................] - 142/1500
    Loading: [==............................] - 143/1500
    Loading: [==............................] - 144/1500
    Loading: [==............................] - 145/1500
    Loading: [==............................] - 146/1500
    Loading: [==............................] - 147/1500
    Loading: [==............................] - 148/1500
    Loading: [==............................] - 149/1500
    Loading: [===...........................] - 150/1500
    Loading: [===...........................] - 151/1500
    Loading: [===...........................] - 152/1500
    Loading: [===...........................] - 153/1500
    Loading: [===...........................] - 154/1500
    Loading: [===...........................] - 155/1500
    Loading: [===...........................] - 156/1500
    Loading: [===...........................] - 157/1500
    Loading: [===...........................] - 158/1500
    Loading: [===...........................] - 159/1500
    Loading: [===...........................] - 160/1500
    Loading: [===...........................] - 161/1500
    Loading: [===...........................] - 162/1500
    Loading: [===...........................] - 163/1500
    Loading: [===...........................] - 164/1500
    Loading: [===...........................] - 165/1500
    Loading: [===...........................] - 166/1500
    Loading: [===...........................] - 167/1500
    Loading: [===...........................] - 168/1500
    Loading: [===...........................] - 169/1500
    Loading: [===...........................] - 170/1500
    Loading: [===...........................] - 171/1500
    Loading: [===...........................] - 172/1500
    Loading: [===...........................] - 173/1500
    Loading: [===...........................] - 174/1500
    Loading: [===...........................] - 175/1500
    Loading: [===...........................] - 176/1500
    Loading: [===...........................] - 177/1500
    Loading: [===...........................] - 178/1500
    Loading: [===...........................] - 179/1500
    Loading: [===...........................] - 180/1500
    Loading: [===...........................] - 181/1500
    Loading: [===...........................] - 182/1500
    Loading: [===...........................] - 183/1500
    Loading: [===...........................] - 184/1500
    Loading: [===...........................] - 185/1500
    Loading: [===...........................] - 186/1500
    Loading: [===...........................] - 187/1500
    Loading: [===...........................] - 188/1500
    Loading: [===...........................] - 189/1500
    Loading: [===...........................] - 190/1500
    Loading: [===...........................] - 191/1500
    Loading: [===...........................] - 192/1500
    Loading: [===...........................] - 193/1500
    Loading: [===...........................] - 194/1500
    Loading: [===...........................] - 195/1500
    Loading: [===...........................] - 196/1500
    Loading: [===...........................] - 197/1500
    Loading: [===...........................] - 198/1500
    Loading: [===...........................] - 199/1500
    Loading: [====..........................] - 200/1500
    Loading: [====..........................] - 201/1500
    Loading: [====..........................] - 202/1500
    Loading: [====..........................] - 203/1500
    Loading: [====..........................] - 204/1500
    Loading: [====..........................] - 205/1500
    Loading: [====..........................] - 206/1500
    Loading: [====..........................] - 207/1500
    Loading: [====..........................] - 208/1500
    Loading: [====..........................] - 209/1500
    Loading: [====..........................] - 210/1500
    Loading: [====..........................] - 211/1500
    Loading: [====..........................] - 212/1500
    Loading: [====..........................] - 213/1500
    Loading: [====..........................] - 214/1500
    Loading: [====..........................] - 215/1500
    Loading: [====..........................] - 216/1500
    Loading: [====..........................] - 217/1500
    Loading: [====..........................] - 218/1500
    Loading: [====..........................] - 219/1500
    Loading: [====..........................] - 220/1500
    Loading: [====..........................] - 221/1500
    Loading: [====..........................] - 222/1500
    Loading: [====..........................] - 223/1500
    Loading: [====..........................] - 224/1500
    Loading: [====..........................] - 225/1500
    Loading: [====..........................] - 226/1500
    Loading: [====..........................] - 227/1500
    Loading: [====..........................] - 228/1500
    Loading: [====..........................] - 229/1500
    Loading: [====..........................] - 230/1500
    Loading: [====..........................] - 231/1500
    Loading: [====..........................] - 232/1500
    Loading: [====..........................] - 233/1500
    Loading: [====..........................] - 234/1500
    Loading: [====..........................] - 235/1500
    Loading: [====..........................] - 236/1500
    Loading: [====..........................] - 237/1500
    Loading: [====..........................] - 238/1500
    Loading: [====..........................] - 239/1500
    Loading: [====..........................] - 240/1500
    Loading: [====..........................] - 241/1500
    Loading: [====..........................] - 242/1500
    Loading: [====..........................] - 243/1500
    Loading: [====..........................] - 244/1500
    Loading: [====..........................] - 245/1500
    Loading: [====..........................] - 246/1500
    Loading: [====..........................] - 247/1500
    Loading: [====..........................] - 248/1500
    Loading: [====..........................] - 249/1500
    Loading: [=====.........................] - 250/1500
    Loading: [=====.........................] - 251/1500
    Loading: [=====.........................] - 252/1500
    Loading: [=====.........................] - 253/1500
    Loading: [=====.........................] - 254/1500
    Loading: [=====.........................] - 255/1500
    Loading: [=====.........................] - 256/1500
    Loading: [=====.........................] - 257/1500
    Loading: [=====.........................] - 258/1500
    Loading: [=====.........................] - 259/1500
    Loading: [=====.........................] - 260/1500
    Loading: [=====.........................] - 261/1500
    Loading: [=====.........................] - 262/1500
    Loading: [=====.........................] - 263/1500
    Loading: [=====.........................] - 264/1500
    Loading: [=====.........................] - 265/1500
    Loading: [=====.........................] - 266/1500
    Loading: [=====.........................] - 267/1500
    Loading: [=====.........................] - 268/1500
    Loading: [=====.........................] - 269/1500
    Loading: [=====.........................] - 270/1500
    Loading: [=====.........................] - 271/1500
    Loading: [=====.........................] - 272/1500
    Loading: [=====.........................] - 273/1500
    Loading: [=====.........................] - 274/1500
    Loading: [=====.........................] - 275/1500
    Loading: [=====.........................] - 276/1500
    Loading: [=====.........................] - 277/1500
    Loading: [=====.........................] - 278/1500
    Loading: [=====.........................] - 279/1500
    Loading: [=====.........................] - 280/1500
    Loading: [=====.........................] - 281/1500
    Loading: [=====.........................] - 282/1500
    Loading: [=====.........................] - 283/1500
    Loading: [=====.........................] - 284/1500
    Loading: [=====.........................] - 285/1500
    Loading: [=====.........................] - 286/1500
    Loading: [=====.........................] - 287/1500
    Loading: [=====.........................] - 288/1500
    Loading: [=====.........................] - 289/1500
    Loading: [=====.........................] - 290/1500
    Loading: [=====.........................] - 291/1500
    Loading: [=====.........................] - 292/1500
    Loading: [=====.........................] - 293/1500
    Loading: [=====.........................] - 294/1500
    Loading: [=====.........................] - 295/1500
    Loading: [=====.........................] - 296/1500
    Loading: [=====.........................] - 297/1500
    Loading: [=====.........................] - 298/1500
    Loading: [=====.........................] - 299/1500
    Loading: [======........................] - 300/1500
    Loading: [======........................] - 301/1500
    Loading: [======........................] - 302/1500
    Loading: [======........................] - 303/1500
    Loading: [======........................] - 304/1500
    Loading: [======........................] - 305/1500
    Loading: [======........................] - 306/1500
    Loading: [======........................] - 307/1500
    Loading: [======........................] - 308/1500
    Loading: [======........................] - 309/1500
    Loading: [======........................] - 310/1500
    Loading: [======........................] - 311/1500
    Loading: [======........................] - 312/1500
    Loading: [======........................] - 313/1500
    Loading: [======........................] - 314/1500
    Loading: [======........................] - 315/1500
    Loading: [======........................] - 316/1500
    Loading: [======........................] - 317/1500
    Loading: [======........................] - 318/1500
    Loading: [======........................] - 319/1500
    Loading: [======........................] - 320/1500
    Loading: [======........................] - 321/1500
    Loading: [======........................] - 322/1500
    Loading: [======........................] - 323/1500
    Loading: [======........................] - 324/1500
    Loading: [======........................] - 325/1500
    Loading: [======........................] - 326/1500
    Loading: [======........................] - 327/1500
    Loading: [======........................] - 328/1500
    Loading: [======........................] - 329/1500
    Loading: [======........................] - 330/1500
    Loading: [======........................] - 331/1500
    Loading: [======........................] - 332/1500
    Loading: [======........................] - 333/1500
    Loading: [======........................] - 334/1500
    Loading: [======........................] - 335/1500
    Loading: [======........................] - 336/1500
    Loading: [======........................] - 337/1500
    Loading: [======........................] - 338/1500
    Loading: [======........................] - 339/1500
    Loading: [======........................] - 340/1500
    Loading: [======........................] - 341/1500
    Loading: [======........................] - 342/1500
    Loading: [======........................] - 343/1500
    Loading: [======........................] - 344/1500
    Loading: [======........................] - 345/1500
    Loading: [======........................] - 346/1500
    Loading: [======........................] - 347/1500
    Loading: [======........................] - 348/1500
    Loading: [======........................] - 349/1500
    Loading: [=======.......................] - 350/1500
    Loading: [=======.......................] - 351/1500
    Loading: [=======.......................] - 352/1500
    Loading: [=======.......................] - 353/1500
    Loading: [=======.......................] - 354/1500
    Loading: [=======.......................] - 355/1500
    Loading: [=======.......................] - 356/1500
    Loading: [=======.......................] - 357/1500
    Loading: [=======.......................] - 358/1500
    Loading: [=======.......................] - 359/1500
    Loading: [=======.......................] - 360/1500
    Loading: [=======.......................] - 361/1500
    Loading: [=======.......................] - 362/1500
    Loading: [=======.......................] - 363/1500
    Loading: [=======.......................] - 364/1500
    Loading: [=======.......................] - 365/1500
    Loading: [=======.......................] - 366/1500
    Loading: [=======.......................] - 367/1500
    Loading: [=======.......................] - 368/1500
    Loading: [=======.......................] - 369/1500
    Loading: [=======.......................] - 370/1500
    Loading: [=======.......................] - 371/1500
    Loading: [=======.......................] - 372/1500
    Loading: [=======.......................] - 373/1500
    Loading: [=======.......................] - 374/1500
    Loading: [=======.......................] - 375/1500
    Loading: [=======.......................] - 376/1500
    Loading: [=======.......................] - 377/1500
    Loading: [=======.......................] - 378/1500
    Loading: [=======.......................] - 379/1500
    Loading: [=======.......................] - 380/1500
    Loading: [=======.......................] - 381/1500
    Loading: [=======.......................] - 382/1500
    Loading: [=======.......................] - 383/1500
    Loading: [=======.......................] - 384/1500
    Loading: [=======.......................] - 385/1500
    Loading: [=======.......................] - 386/1500
    Loading: [=======.......................] - 387/1500
    Loading: [=======.......................] - 388/1500
    Loading: [=======.......................] - 389/1500
    Loading: [=======.......................] - 390/1500
    Loading: [=======.......................] - 391/1500
    Loading: [=======.......................] - 392/1500
    Loading: [=======.......................] - 393/1500
    Loading: [=======.......................] - 394/1500
    Loading: [=======.......................] - 395/1500
    Loading: [=======.......................] - 396/1500
    Loading: [=======.......................] - 397/1500
    Loading: [=======.......................] - 398/1500
    Loading: [=======.......................] - 399/1500
    Loading: [========......................] - 400/1500
    Loading: [========......................] - 401/1500
    Loading: [========......................] - 402/1500
    Loading: [========......................] - 403/1500
    Loading: [========......................] - 404/1500
    Loading: [========......................] - 405/1500
    Loading: [========......................] - 406/1500
    Loading: [========......................] - 407/1500
    Loading: [========......................] - 408/1500
    Loading: [========......................] - 409/1500
    Loading: [========......................] - 410/1500
    Loading: [========......................] - 411/1500
    Loading: [========......................] - 412/1500
    Loading: [========......................] - 413/1500
    Loading: [========......................] - 414/1500
    Loading: [========......................] - 415/1500
    Loading: [========......................] - 416/1500
    Loading: [========......................] - 417/1500
    Loading: [========......................] - 418/1500
    Loading: [========......................] - 419/1500
    Loading: [========......................] - 420/1500
    Loading: [========......................] - 421/1500
    Loading: [========......................] - 422/1500
    Loading: [========......................] - 423/1500
    Loading: [========......................] - 424/1500
    Loading: [========......................] - 425/1500
    Loading: [========......................] - 426/1500
    Loading: [========......................] - 427/1500
    Loading: [========......................] - 428/1500
    Loading: [========......................] - 429/1500
    Loading: [========......................] - 430/1500
    Loading: [========......................] - 431/1500
    Loading: [========......................] - 432/1500
    Loading: [========......................] - 433/1500
    Loading: [========......................] - 434/1500
    Loading: [========......................] - 435/1500
    Loading: [========......................] - 436/1500
    Loading: [========......................] - 437/1500
    Loading: [========......................] - 438/1500
    Loading: [========......................] - 439/1500
    Loading: [========......................] - 440/1500
    Loading: [========......................] - 441/1500
    Loading: [========......................] - 442/1500
    Loading: [========......................] - 443/1500
    Loading: [========......................] - 444/1500
    Loading: [========......................] - 445/1500
    Loading: [========......................] - 446/1500
    Loading: [========......................] - 447/1500
    Loading: [========......................] - 448/1500
    Loading: [========......................] - 449/1500
    Loading: [=========.....................] - 450/1500
    Loading: [=========.....................] - 451/1500
    Loading: [=========.....................] - 452/1500
    Loading: [=========.....................] - 453/1500
    Loading: [=========.....................] - 454/1500
    Loading: [=========.....................] - 455/1500
    Loading: [=========.....................] - 456/1500
    Loading: [=========.....................] - 457/1500
    Loading: [=========.....................] - 458/1500
    Loading: [=========.....................] - 459/1500
    Loading: [=========.....................] - 460/1500
    Loading: [=========.....................] - 461/1500
    Loading: [=========.....................] - 462/1500
    Loading: [=========.....................] - 463/1500
    Loading: [=========.....................] - 464/1500
    Loading: [=========.....................] - 465/1500
    Loading: [=========.....................] - 466/1500
    Loading: [=========.....................] - 467/1500
    Loading: [=========.....................] - 468/1500
    Loading: [=========.....................] - 469/1500
    Loading: [=========.....................] - 470/1500
    Loading: [=========.....................] - 471/1500
    Loading: [=========.....................] - 472/1500
    Loading: [=========.....................] - 473/1500
    Loading: [=========.....................] - 474/1500
    Loading: [=========.....................] - 475/1500
    Loading: [=========.....................] - 476/1500
    Loading: [=========.....................] - 477/1500
    Loading: [=========.....................] - 478/1500
    Loading: [=========.....................] - 479/1500
    Loading: [=========.....................] - 480/1500
    Loading: [=========.....................] - 481/1500
    Loading: [=========.....................] - 482/1500
    Loading: [=========.....................] - 483/1500
    Loading: [=========.....................] - 484/1500
    Loading: [=========.....................] - 485/1500
    Loading: [=========.....................] - 486/1500
    Loading: [=========.....................] - 487/1500
    Loading: [=========.....................] - 488/1500
    Loading: [=========.....................] - 489/1500
    Loading: [=========.....................] - 490/1500
    Loading: [=========.....................] - 491/1500
    Loading: [=========.....................] - 492/1500
    Loading: [=========.....................] - 493/1500
    Loading: [=========.....................] - 494/1500
    Loading: [=========.....................] - 495/1500
    Loading: [=========.....................] - 496/1500
    Loading: [=========.....................] - 497/1500
    Loading: [=========.....................] - 498/1500
    Loading: [=========.....................] - 499/1500
    Loading: [==========....................] - 500/1500
    Loading: [==========....................] - 501/1500
    Loading: [==========....................] - 502/1500
    Loading: [==========....................] - 503/1500
    Loading: [==========....................] - 504/1500
    Loading: [==========....................] - 505/1500
    Loading: [==========....................] - 506/1500
    Loading: [==========....................] - 507/1500
    Loading: [==========....................] - 508/1500
    Loading: [==========....................] - 509/1500
    Loading: [==========....................] - 510/1500
    Loading: [==========....................] - 511/1500
    Loading: [==========....................] - 512/1500
    Loading: [==========....................] - 513/1500
    Loading: [==========....................] - 514/1500
    Loading: [==========....................] - 515/1500
    Loading: [==========....................] - 516/1500
    Loading: [==========....................] - 517/1500
    Loading: [==========....................] - 518/1500
    Loading: [==========....................] - 519/1500
    Loading: [==========....................] - 520/1500
    Loading: [==========....................] - 521/1500
    Loading: [==========....................] - 522/1500
    Loading: [==========....................] - 523/1500
    Loading: [==========....................] - 524/1500
    Loading: [==========....................] - 525/1500
    Loading: [==========....................] - 526/1500
    Loading: [==========....................] - 527/1500
    Loading: [==========....................] - 528/1500
    Loading: [==========....................] - 529/1500
    Loading: [==========....................] - 530/1500
    Loading: [==========....................] - 531/1500
    Loading: [==========....................] - 532/1500
    Loading: [==========....................] - 533/1500
    Loading: [==========....................] - 534/1500
    Loading: [==========....................] - 535/1500
    Loading: [==========....................] - 536/1500
    Loading: [==========....................] - 537/1500
    Loading: [==========....................] - 538/1500
    Loading: [==========....................] - 539/1500
    Loading: [==========....................] - 540/1500
    Loading: [==========....................] - 541/1500
    Loading: [==========....................] - 542/1500
    Loading: [==========....................] - 543/1500
    Loading: [==========....................] - 544/1500
    Loading: [==========....................] - 545/1500
    Loading: [==========....................] - 546/1500
    Loading: [==========....................] - 547/1500
    Loading: [==========....................] - 548/1500
    Loading: [==========....................] - 549/1500
    Loading: [===========...................] - 550/1500
    Loading: [===========...................] - 551/1500
    Loading: [===========...................] - 552/1500
    Loading: [===========...................] - 553/1500
    Loading: [===========...................] - 554/1500
    Loading: [===========...................] - 555/1500
    Loading: [===========...................] - 556/1500
    Loading: [===========...................] - 557/1500
    Loading: [===========...................] - 558/1500
    Loading: [===========...................] - 559/1500
    Loading: [===========...................] - 560/1500
    Loading: [===========...................] - 561/1500
    Loading: [===========...................] - 562/1500
    Loading: [===========...................] - 563/1500
    Loading: [===========...................] - 564/1500
    Loading: [===========...................] - 565/1500
    Loading: [===========...................] - 566/1500
    Loading: [===========...................] - 567/1500
    Loading: [===========...................] - 568/1500
    Loading: [===========...................] - 569/1500
    Loading: [===========...................] - 570/1500
    Loading: [===========...................] - 571/1500
    Loading: [===========...................] - 572/1500
    Loading: [===========...................] - 573/1500
    Loading: [===========...................] - 574/1500
    Loading: [===========...................] - 575/1500
    Loading: [===========...................] - 576/1500
    Loading: [===========...................] - 577/1500
    Loading: [===========...................] - 578/1500
    Loading: [===========...................] - 579/1500
    Loading: [===========...................] - 580/1500
    Loading: [===========...................] - 581/1500
    Loading: [===========...................] - 582/1500
    Loading: [===========...................] - 583/1500
    Loading: [===========...................] - 584/1500
    Loading: [===========...................] - 585/1500
    Loading: [===========...................] - 586/1500
    Loading: [===========...................] - 587/1500
    Loading: [===========...................] - 588/1500
    Loading: [===========...................] - 589/1500
    Loading: [===========...................] - 590/1500
    Loading: [===========...................] - 591/1500
    Loading: [===========...................] - 592/1500
    Loading: [===========...................] - 593/1500
    Loading: [===========...................] - 594/1500
    Loading: [===========...................] - 595/1500
    Loading: [===========...................] - 596/1500
    Loading: [===========...................] - 597/1500
    Loading: [===========...................] - 598/1500
    Loading: [===========...................] - 599/1500
    Loading: [============..................] - 600/1500
    Loading: [============..................] - 601/1500
    Loading: [============..................] - 602/1500
    Loading: [============..................] - 603/1500
    Loading: [============..................] - 604/1500
    Loading: [============..................] - 605/1500
    Loading: [============..................] - 606/1500
    Loading: [============..................] - 607/1500
    Loading: [============..................] - 608/1500
    Loading: [============..................] - 609/1500
    Loading: [============..................] - 610/1500
    Loading: [============..................] - 611/1500
    Loading: [============..................] - 612/1500
    Loading: [============..................] - 613/1500
    Loading: [============..................] - 614/1500
    Loading: [============..................] - 615/1500
    Loading: [============..................] - 616/1500
    Loading: [============..................] - 617/1500
    Loading: [============..................] - 618/1500
    Loading: [============..................] - 619/1500
    Loading: [============..................] - 620/1500
    Loading: [============..................] - 621/1500
    Loading: [============..................] - 622/1500
    Loading: [============..................] - 623/1500
    Loading: [============..................] - 624/1500
    Loading: [============..................] - 625/1500
    Loading: [============..................] - 626/1500
    Loading: [============..................] - 627/1500
    Loading: [============..................] - 628/1500
    Loading: [============..................] - 629/1500
    Loading: [============..................] - 630/1500
    Loading: [============..................] - 631/1500
    Loading: [============..................] - 632/1500
    Loading: [============..................] - 633/1500
    Loading: [============..................] - 634/1500
    Loading: [============..................] - 635/1500
    Loading: [============..................] - 636/1500
    Loading: [============..................] - 637/1500
    Loading: [============..................] - 638/1500
    Loading: [============..................] - 639/1500
    Loading: [============..................] - 640/1500
    Loading: [============..................] - 641/1500
    Loading: [============..................] - 642/1500
    Loading: [============..................] - 643/1500
    Loading: [============..................] - 644/1500
    Loading: [============..................] - 645/1500
    Loading: [============..................] - 646/1500
    Loading: [============..................] - 647/1500
    Loading: [============..................] - 648/1500
    Loading: [============..................] - 649/1500
    Loading: [=============.................] - 650/1500
    Loading: [=============.................] - 651/1500
    Loading: [=============.................] - 652/1500
    Loading: [=============.................] - 653/1500
    Loading: [=============.................] - 654/1500
    Loading: [=============.................] - 655/1500
    Loading: [=============.................] - 656/1500
    Loading: [=============.................] - 657/1500
    Loading: [=============.................] - 658/1500
    Loading: [=============.................] - 659/1500
    Loading: [=============.................] - 660/1500
    Loading: [=============.................] - 661/1500
    Loading: [=============.................] - 662/1500
    Loading: [=============.................] - 663/1500
    Loading: [=============.................] - 664/1500
    Loading: [=============.................] - 665/1500
    Loading: [=============.................] - 666/1500
    Loading: [=============.................] - 667/1500
    Loading: [=============.................] - 668/1500
    Loading: [=============.................] - 669/1500
    Loading: [=============.................] - 670/1500
    Loading: [=============.................] - 671/1500
    Loading: [=============.................] - 672/1500
    Loading: [=============.................] - 673/1500
    Loading: [=============.................] - 674/1500
    Loading: [=============.................] - 675/1500
    Loading: [=============.................] - 676/1500
    Loading: [=============.................] - 677/1500
    Loading: [=============.................] - 678/1500
    Loading: [=============.................] - 679/1500
    Loading: [=============.................] - 680/1500
    Loading: [=============.................] - 681/1500
    Loading: [=============.................] - 682/1500
    Loading: [=============.................] - 683/1500
    Loading: [=============.................] - 684/1500
    Loading: [=============.................] - 685/1500
    Loading: [=============.................] - 686/1500
    Loading: [=============.................] - 687/1500
    Loading: [=============.................] - 688/1500
    Loading: [=============.................] - 689/1500
    Loading: [=============.................] - 690/1500
    Loading: [=============.................] - 691/1500
    Loading: [=============.................] - 692/1500
    Loading: [=============.................] - 693/1500
    Loading: [=============.................] - 694/1500
    Loading: [=============.................] - 695/1500
    Loading: [=============.................] - 696/1500
    Loading: [=============.................] - 697/1500
    Loading: [=============.................] - 698/1500
    Loading: [=============.................] - 699/1500
    Loading: [==============................] - 700/1500
    Loading: [==============................] - 701/1500
    Loading: [==============................] - 702/1500
    Loading: [==============................] - 703/1500
    Loading: [==============................] - 704/1500
    Loading: [==============................] - 705/1500
    Loading: [==============................] - 706/1500
    Loading: [==============................] - 707/1500
    Loading: [==============................] - 708/1500
    Loading: [==============................] - 709/1500
    Loading: [==============................] - 710/1500
    Loading: [==============................] - 711/1500
    Loading: [==============................] - 712/1500
    Loading: [==============................] - 713/1500
    Loading: [==============................] - 714/1500
    Loading: [==============................] - 715/1500
    Loading: [==============................] - 716/1500
    Loading: [==============................] - 717/1500
    Loading: [==============................] - 718/1500
    Loading: [==============................] - 719/1500
    Loading: [==============................] - 720/1500
    Loading: [==============................] - 721/1500
    Loading: [==============................] - 722/1500
    Loading: [==============................] - 723/1500
    Loading: [==============................] - 724/1500
    Loading: [==============................] - 725/1500
    Loading: [==============................] - 726/1500
    Loading: [==============................] - 727/1500
    Loading: [==============................] - 728/1500
    Loading: [==============................] - 729/1500
    Loading: [==============................] - 730/1500
    Loading: [==============................] - 731/1500
    Loading: [==============................] - 732/1500
    Loading: [==============................] - 733/1500
    Loading: [==============................] - 734/1500
    Loading: [==============................] - 735/1500
    Loading: [==============................] - 736/1500
    Loading: [==============................] - 737/1500
    Loading: [==============................] - 738/1500
    Loading: [==============................] - 739/1500
    Loading: [==============................] - 740/1500
    Loading: [==============................] - 741/1500
    Loading: [==============................] - 742/1500
    Loading: [==============................] - 743/1500
    Loading: [==============................] - 744/1500
    Loading: [==============................] - 745/1500
    Loading: [==============................] - 746/1500
    Loading: [==============................] - 747/1500
    Loading: [==============................] - 748/1500
    Loading: [==============................] - 749/1500
    Loading: [===============...............] - 750/1500
    Loading: [===============...............] - 751/1500
    Loading: [===============...............] - 752/1500
    Loading: [===============...............] - 753/1500
    Loading: [===============...............] - 754/1500
    Loading: [===============...............] - 755/1500
    Loading: [===============...............] - 756/1500
    Loading: [===============...............] - 757/1500
    Loading: [===============...............] - 758/1500
    Loading: [===============...............] - 759/1500
    Loading: [===============...............] - 760/1500
    Loading: [===============...............] - 761/1500
    Loading: [===============...............] - 762/1500
    Loading: [===============...............] - 763/1500
    Loading: [===============...............] - 764/1500
    Loading: [===============...............] - 765/1500
    Loading: [===============...............] - 766/1500
    Loading: [===============...............] - 767/1500
    Loading: [===============...............] - 768/1500
    Loading: [===============...............] - 769/1500
    Loading: [===============...............] - 770/1500
    Loading: [===============...............] - 771/1500
    Loading: [===============...............] - 772/1500
    Loading: [===============...............] - 773/1500
    Loading: [===============...............] - 774/1500
    Loading: [===============...............] - 775/1500
    Loading: [===============...............] - 776/1500
    Loading: [===============...............] - 777/1500
    Loading: [===============...............] - 778/1500
    Loading: [===============...............] - 779/1500
    Loading: [===============...............] - 780/1500
    Loading: [===============...............] - 781/1500
    Loading: [===============...............] - 782/1500
    Loading: [===============...............] - 783/1500
    Loading: [===============...............] - 784/1500
    Loading: [===============...............] - 785/1500
    Loading: [===============...............] - 786/1500
    Loading: [===============...............] - 787/1500
    Loading: [===============...............] - 788/1500
    Loading: [===============...............] - 789/1500
    Loading: [===============...............] - 790/1500
    Loading: [===============...............] - 791/1500
    Loading: [===============...............] - 792/1500
    Loading: [===============...............] - 793/1500
    Loading: [===============...............] - 794/1500
    Loading: [===============...............] - 795/1500
    Loading: [===============...............] - 796/1500
    Loading: [===============...............] - 797/1500
    Loading: [===============...............] - 798/1500
    Loading: [===============...............] - 799/1500
    Loading: [================..............] - 800/1500
    Loading: [================..............] - 801/1500
    Loading: [================..............] - 802/1500
    Loading: [================..............] - 803/1500
    Loading: [================..............] - 804/1500
    Loading: [================..............] - 805/1500
    Loading: [================..............] - 806/1500
    Loading: [================..............] - 807/1500
    Loading: [================..............] - 808/1500
    Loading: [================..............] - 809/1500
    Loading: [================..............] - 810/1500
    Loading: [================..............] - 811/1500
    Loading: [================..............] - 812/1500
    Loading: [================..............] - 813/1500
    Loading: [================..............] - 814/1500
    Loading: [================..............] - 815/1500
    Loading: [================..............] - 816/1500
    Loading: [================..............] - 817/1500
    Loading: [================..............] - 818/1500
    Loading: [================..............] - 819/1500
    Loading: [================..............] - 820/1500
    Loading: [================..............] - 821/1500
    Loading: [================..............] - 822/1500
    Loading: [================..............] - 823/1500
    Loading: [================..............] - 824/1500
    Loading: [================..............] - 825/1500
    Loading: [================..............] - 826/1500
    Loading: [================..............] - 827/1500
    Loading: [================..............] - 828/1500
    Loading: [================..............] - 829/1500
    Loading: [================..............] - 830/1500
    Loading: [================..............] - 831/1500
    Loading: [================..............] - 832/1500
    Loading: [================..............] - 833/1500
    Loading: [================..............] - 834/1500
    Loading: [================..............] - 835/1500
    Loading: [================..............] - 836/1500
    Loading: [================..............] - 837/1500
    Loading: [================..............] - 838/1500
    Loading: [================..............] - 839/1500
    Loading: [================..............] - 840/1500
    Loading: [================..............] - 841/1500
    Loading: [================..............] - 842/1500
    Loading: [================..............] - 843/1500
    Loading: [================..............] - 844/1500
    Loading: [================..............] - 845/1500
    Loading: [================..............] - 846/1500
    Loading: [================..............] - 847/1500
    Loading: [================..............] - 848/1500
    Loading: [================..............] - 849/1500
    Loading: [=================.............] - 850/1500
    Loading: [=================.............] - 851/1500
    Loading: [=================.............] - 852/1500
    Loading: [=================.............] - 853/1500
    Loading: [=================.............] - 854/1500
    Loading: [=================.............] - 855/1500
    Loading: [=================.............] - 856/1500
    Loading: [=================.............] - 857/1500
    Loading: [=================.............] - 858/1500
    Loading: [=================.............] - 859/1500
    Loading: [=================.............] - 860/1500
    Loading: [=================.............] - 861/1500
    Loading: [=================.............] - 862/1500
    Loading: [=================.............] - 863/1500
    Loading: [=================.............] - 864/1500
    Loading: [=================.............] - 865/1500
    Loading: [=================.............] - 866/1500
    Loading: [=================.............] - 867/1500
    Loading: [=================.............] - 868/1500
    Loading: [=================.............] - 869/1500
    Loading: [=================.............] - 870/1500
    Loading: [=================.............] - 871/1500
    Loading: [=================.............] - 872/1500
    Loading: [=================.............] - 873/1500
    Loading: [=================.............] - 874/1500
    Loading: [=================.............] - 875/1500
    Loading: [=================.............] - 876/1500
    Loading: [=================.............] - 877/1500
    Loading: [=================.............] - 878/1500
    Loading: [=================.............] - 879/1500
    Loading: [=================.............] - 880/1500
    Loading: [=================.............] - 881/1500
    Loading: [=================.............] - 882/1500
    Loading: [=================.............] - 883/1500
    Loading: [=================.............] - 884/1500
    Loading: [=================.............] - 885/1500
    Loading: [=================.............] - 886/1500
    Loading: [=================.............] - 887/1500
    Loading: [=================.............] - 888/1500
    Loading: [=================.............] - 889/1500
    Loading: [=================.............] - 890/1500
    Loading: [=================.............] - 891/1500
    Loading: [=================.............] - 892/1500
    Loading: [=================.............] - 893/1500
    Loading: [=================.............] - 894/1500
    Loading: [=================.............] - 895/1500
    Loading: [=================.............] - 896/1500
    Loading: [=================.............] - 897/1500
    Loading: [=================.............] - 898/1500
    Loading: [=================.............] - 899/1500
    Loading: [==================............] - 900/1500
    Loading: [==================............] - 901/1500
    Loading: [==================............] - 902/1500
    Loading: [==================............] - 903/1500
    Loading: [==================............] - 904/1500
    Loading: [==================............] - 905/1500
    Loading: [==================............] - 906/1500
    Loading: [==================............] - 907/1500
    Loading: [==================............] - 908/1500
    Loading: [==================............] - 909/1500
    Loading: [==================............] - 910/1500
    Loading: [==================............] - 911/1500
    Loading: [==================............] - 912/1500
    Loading: [==================............] - 913/1500
    Loading: [==================............] - 914/1500
    Loading: [==================............] - 915/1500
    Loading: [==================............] - 916/1500
    Loading: [==================............] - 917/1500
    Loading: [==================............] - 918/1500
    Loading: [==================............] - 919/1500
    Loading: [==================............] - 920/1500
    Loading: [==================............] - 921/1500
    Loading: [==================............] - 922/1500
    Loading: [==================............] - 923/1500
    Loading: [==================............] - 924/1500
    Loading: [==================............] - 925/1500
    Loading: [==================............] - 926/1500
    Loading: [==================............] - 927/1500
    Loading: [==================............] - 928/1500
    Loading: [==================............] - 929/1500
    Loading: [==================............] - 930/1500
    Loading: [==================............] - 931/1500
    Loading: [==================............] - 932/1500
    Loading: [==================............] - 933/1500
    Loading: [==================............] - 934/1500
    Loading: [==================............] - 935/1500
    Loading: [==================............] - 936/1500
    Loading: [==================............] - 937/1500
    Loading: [==================............] - 938/1500
    Loading: [==================............] - 939/1500
    Loading: [==================............] - 940/1500
    Loading: [==================............] - 941/1500
    Loading: [==================............] - 942/1500
    Loading: [==================............] - 943/1500
    Loading: [==================............] - 944/1500
    Loading: [==================............] - 945/1500
    Loading: [==================............] - 946/1500
    Loading: [==================............] - 947/1500
    Loading: [==================............] - 948/1500
    Loading: [==================............] - 949/1500
    Loading: [===================...........] - 950/1500
    Loading: [===================...........] - 951/1500
    Loading: [===================...........] - 952/1500
    Loading: [===================...........] - 953/1500
    Loading: [===================...........] - 954/1500
    Loading: [===================...........] - 955/1500
    Loading: [===================...........] - 956/1500
    Loading: [===================...........] - 957/1500
    Loading: [===================...........] - 958/1500
    Loading: [===================...........] - 959/1500
    Loading: [===================...........] - 960/1500
    Loading: [===================...........] - 961/1500
    Loading: [===================...........] - 962/1500
    Loading: [===================...........] - 963/1500
    Loading: [===================...........] - 964/1500
    Loading: [===================...........] - 965/1500
    Loading: [===================...........] - 966/1500
    Loading: [===================...........] - 967/1500
    Loading: [===================...........] - 968/1500
    Loading: [===================...........] - 969/1500
    Loading: [===================...........] - 970/1500
    Loading: [===================...........] - 971/1500
    Loading: [===================...........] - 972/1500
    Loading: [===================...........] - 973/1500
    Loading: [===================...........] - 974/1500
    Loading: [===================...........] - 975/1500
    Loading: [===================...........] - 976/1500
    Loading: [===================...........] - 977/1500
    Loading: [===================...........] - 978/1500
    Loading: [===================...........] - 979/1500
    Loading: [===================...........] - 980/1500
    Loading: [===================...........] - 981/1500
    Loading: [===================...........] - 982/1500
    Loading: [===================...........] - 983/1500
    Loading: [===================...........] - 984/1500
    Loading: [===================...........] - 985/1500
    Loading: [===================...........] - 986/1500
    Loading: [===================...........] - 987/1500
    Loading: [===================...........] - 988/1500
    Loading: [===================...........] - 989/1500
    Loading: [===================...........] - 990/1500
    Loading: [===================...........] - 991/1500
    Loading: [===================...........] - 992/1500
    Loading: [===================...........] - 993/1500
    Loading: [===================...........] - 994/1500
    Loading: [===================...........] - 995/1500
    Loading: [===================...........] - 996/1500
    Loading: [===================...........] - 997/1500
    Loading: [===================...........] - 998/1500
    Loading: [===================...........] - 999/1500
    Loading: [====================..........] - 1000/1500
    Loading: [====================..........] - 1001/1500
    Loading: [====================..........] - 1002/1500
    Loading: [====================..........] - 1003/1500
    Loading: [====================..........] - 1004/1500
    Loading: [====================..........] - 1005/1500
    Loading: [====================..........] - 1006/1500
    Loading: [====================..........] - 1007/1500
    Loading: [====================..........] - 1008/1500
    Loading: [====================..........] - 1009/1500
    Loading: [====================..........] - 1010/1500
    Loading: [====================..........] - 1011/1500
    Loading: [====================..........] - 1012/1500
    Loading: [====================..........] - 1013/1500
    Loading: [====================..........] - 1014/1500
    Loading: [====================..........] - 1015/1500
    Loading: [====================..........] - 1016/1500
    Loading: [====================..........] - 1017/1500
    Loading: [====================..........] - 1018/1500
    Loading: [====================..........] - 1019/1500
    Loading: [====================..........] - 1020/1500
    Loading: [====================..........] - 1021/1500
    Loading: [====================..........] - 1022/1500
    Loading: [====================..........] - 1023/1500
    Loading: [====================..........] - 1024/1500
    Loading: [====================..........] - 1025/1500
    Loading: [====================..........] - 1026/1500
    Loading: [====================..........] - 1027/1500
    Loading: [====================..........] - 1028/1500
    Loading: [====================..........] - 1029/1500
    Loading: [====================..........] - 1030/1500
    Loading: [====================..........] - 1031/1500
    Loading: [====================..........] - 1032/1500
    Loading: [====================..........] - 1033/1500
    Loading: [====================..........] - 1034/1500
    Loading: [====================..........] - 1035/1500
    Loading: [====================..........] - 1036/1500
    Loading: [====================..........] - 1037/1500
    Loading: [====================..........] - 1038/1500
    Loading: [====================..........] - 1039/1500
    Loading: [====================..........] - 1040/1500
    Loading: [====================..........] - 1041/1500
    Loading: [====================..........] - 1042/1500
    Loading: [====================..........] - 1043/1500
    Loading: [====================..........] - 1044/1500
    Loading: [====================..........] - 1045/1500
    Loading: [====================..........] - 1046/1500
    Loading: [====================..........] - 1047/1500
    Loading: [====================..........] - 1048/1500
    Loading: [====================..........] - 1049/1500
    Loading: [=====================.........] - 1050/1500
    Loading: [=====================.........] - 1051/1500
    Loading: [=====================.........] - 1052/1500
    Loading: [=====================.........] - 1053/1500
    Loading: [=====================.........] - 1054/1500
    Loading: [=====================.........] - 1055/1500
    Loading: [=====================.........] - 1056/1500
    Loading: [=====================.........] - 1057/1500
    Loading: [=====================.........] - 1058/1500
    Loading: [=====================.........] - 1059/1500
    Loading: [=====================.........] - 1060/1500
    Loading: [=====================.........] - 1061/1500
    Loading: [=====================.........] - 1062/1500
    Loading: [=====================.........] - 1063/1500
    Loading: [=====================.........] - 1064/1500
    Loading: [=====================.........] - 1065/1500
    Loading: [=====================.........] - 1066/1500
    Loading: [=====================.........] - 1067/1500
    Loading: [=====================.........] - 1068/1500
    Loading: [=====================.........] - 1069/1500
    Loading: [=====================.........] - 1070/1500
    Loading: [=====================.........] - 1071/1500
    Loading: [=====================.........] - 1072/1500
    Loading: [=====================.........] - 1073/1500
    Loading: [=====================.........] - 1074/1500
    Loading: [=====================.........] - 1075/1500
    Loading: [=====================.........] - 1076/1500
    Loading: [=====================.........] - 1077/1500
    Loading: [=====================.........] - 1078/1500
    Loading: [=====================.........] - 1079/1500
    Loading: [=====================.........] - 1080/1500
    Loading: [=====================.........] - 1081/1500
    Loading: [=====================.........] - 1082/1500
    Loading: [=====================.........] - 1083/1500
    Loading: [=====================.........] - 1084/1500
    Loading: [=====================.........] - 1085/1500
    Loading: [=====================.........] - 1086/1500
    Loading: [=====================.........] - 1087/1500
    Loading: [=====================.........] - 1088/1500
    Loading: [=====================.........] - 1089/1500
    Loading: [=====================.........] - 1090/1500
    Loading: [=====================.........] - 1091/1500
    Loading: [=====================.........] - 1092/1500
    Loading: [=====================.........] - 1093/1500
    Loading: [=====================.........] - 1094/1500
    Loading: [=====================.........] - 1095/1500
    Loading: [=====================.........] - 1096/1500
    Loading: [=====================.........] - 1097/1500
    Loading: [=====================.........] - 1098/1500
    Loading: [=====================.........] - 1099/1500
    Loading: [======================........] - 1100/1500
    Loading: [======================........] - 1101/1500
    Loading: [======================........] - 1102/1500
    Loading: [======================........] - 1103/1500
    Loading: [======================........] - 1104/1500
    Loading: [======================........] - 1105/1500
    Loading: [======================........] - 1106/1500
    Loading: [======================........] - 1107/1500
    Loading: [======================........] - 1108/1500
    Loading: [======================........] - 1109/1500
    Loading: [======================........] - 1110/1500
    Loading: [======================........] - 1111/1500
    Loading: [======================........] - 1112/1500
    Loading: [======================........] - 1113/1500
    Loading: [======================........] - 1114/1500
    Loading: [======================........] - 1115/1500
    Loading: [======================........] - 1116/1500
    Loading: [======================........] - 1117/1500
    Loading: [======================........] - 1118/1500
    Loading: [======================........] - 1119/1500
    Loading: [======================........] - 1120/1500
    Loading: [======================........] - 1121/1500
    Loading: [======================........] - 1122/1500
    Loading: [======================........] - 1123/1500
    Loading: [======================........] - 1124/1500
    Loading: [======================........] - 1125/1500
    Loading: [======================........] - 1126/1500
    Loading: [======================........] - 1127/1500
    Loading: [======================........] - 1128/1500
    Loading: [======================........] - 1129/1500
    Loading: [======================........] - 1130/1500
    Loading: [======================........] - 1131/1500
    Loading: [======================........] - 1132/1500
    Loading: [======================........] - 1133/1500
    Loading: [======================........] - 1134/1500
    Loading: [======================........] - 1135/1500
    Loading: [======================........] - 1136/1500
    Loading: [======================........] - 1137/1500
    Loading: [======================........] - 1138/1500
    Loading: [======================........] - 1139/1500
    Loading: [======================........] - 1140/1500
    Loading: [======================........] - 1141/1500
    Loading: [======================........] - 1142/1500
    Loading: [======================........] - 1143/1500
    Loading: [======================........] - 1144/1500
    Loading: [======================........] - 1145/1500
    Loading: [======================........] - 1146/1500
    Loading: [======================........] - 1147/1500
    Loading: [======================........] - 1148/1500
    Loading: [======================........] - 1149/1500
    Loading: [=======================.......] - 1150/1500
    Loading: [=======================.......] - 1151/1500
    Loading: [=======================.......] - 1152/1500
    Loading: [=======================.......] - 1153/1500
    Loading: [=======================.......] - 1154/1500
    Loading: [=======================.......] - 1155/1500
    Loading: [=======================.......] - 1156/1500
    Loading: [=======================.......] - 1157/1500
    Loading: [=======================.......] - 1158/1500
    Loading: [=======================.......] - 1159/1500
    Loading: [=======================.......] - 1160/1500
    Loading: [=======================.......] - 1161/1500
    Loading: [=======================.......] - 1162/1500
    Loading: [=======================.......] - 1163/1500
    Loading: [=======================.......] - 1164/1500
    Loading: [=======================.......] - 1165/1500
    Loading: [=======================.......] - 1166/1500
    Loading: [=======================.......] - 1167/1500
    Loading: [=======================.......] - 1168/1500
    Loading: [=======================.......] - 1169/1500
    Loading: [=======================.......] - 1170/1500
    Loading: [=======================.......] - 1171/1500
    Loading: [=======================.......] - 1172/1500
    Loading: [=======================.......] - 1173/1500
    Loading: [=======================.......] - 1174/1500
    Loading: [=======================.......] - 1175/1500
    Loading: [=======================.......] - 1176/1500
    Loading: [=======================.......] - 1177/1500
    Loading: [=======================.......] - 1178/1500
    Loading: [=======================.......] - 1179/1500
    Loading: [=======================.......] - 1180/1500
    Loading: [=======================.......] - 1181/1500
    Loading: [=======================.......] - 1182/1500
    Loading: [=======================.......] - 1183/1500
    Loading: [=======================.......] - 1184/1500
    Loading: [=======================.......] - 1185/1500
    Loading: [=======================.......] - 1186/1500
    Loading: [=======================.......] - 1187/1500
    Loading: [=======================.......] - 1188/1500
    Loading: [=======================.......] - 1189/1500
    Loading: [=======================.......] - 1190/1500
    Loading: [=======================.......] - 1191/1500
    Loading: [=======================.......] - 1192/1500
    Loading: [=======================.......] - 1193/1500
    Loading: [=======================.......] - 1194/1500
    Loading: [=======================.......] - 1195/1500
    Loading: [=======================.......] - 1196/1500
    Loading: [=======================.......] - 1197/1500
    Loading: [=======================.......] - 1198/1500
    Loading: [=======================.......] - 1199/1500
    Loading: [========================......] - 1200/1500
    Loading: [========================......] - 1201/1500
    Loading: [========================......] - 1202/1500
    Loading: [========================......] - 1203/1500
    Loading: [========================......] - 1204/1500
    Loading: [========================......] - 1205/1500
    Loading: [========================......] - 1206/1500
    Loading: [========================......] - 1207/1500
    Loading: [========================......] - 1208/1500
    Loading: [========================......] - 1209/1500
    Loading: [========================......] - 1210/1500
    Loading: [========================......] - 1211/1500
    Loading: [========================......] - 1212/1500
    Loading: [========================......] - 1213/1500
    Loading: [========================......] - 1214/1500
    Loading: [========================......] - 1215/1500
    Loading: [========================......] - 1216/1500
    Loading: [========================......] - 1217/1500
    Loading: [========================......] - 1218/1500
    Loading: [========================......] - 1219/1500
    Loading: [========================......] - 1220/1500
    Loading: [========================......] - 1221/1500
    Loading: [========================......] - 1222/1500
    Loading: [========================......] - 1223/1500
    Loading: [========================......] - 1224/1500
    Loading: [========================......] - 1225/1500
    Loading: [========================......] - 1226/1500
    Loading: [========================......] - 1227/1500
    Loading: [========================......] - 1228/1500
    Loading: [========================......] - 1229/1500
    Loading: [========================......] - 1230/1500
    Loading: [========================......] - 1231/1500
    Loading: [========================......] - 1232/1500
    Loading: [========================......] - 1233/1500
    Loading: [========================......] - 1234/1500
    Loading: [========================......] - 1235/1500
    Loading: [========================......] - 1236/1500
    Loading: [========================......] - 1237/1500
    Loading: [========================......] - 1238/1500
    Loading: [========================......] - 1239/1500
    Loading: [========================......] - 1240/1500
    Loading: [========================......] - 1241/1500
    Loading: [========================......] - 1242/1500
    Loading: [========================......] - 1243/1500
    Loading: [========================......] - 1244/1500
    Loading: [========================......] - 1245/1500
    Loading: [========================......] - 1246/1500
    Loading: [========================......] - 1247/1500
    Loading: [========================......] - 1248/1500
    Loading: [========================......] - 1249/1500
    Loading: [=========================.....] - 1250/1500
    Loading: [=========================.....] - 1251/1500
    Loading: [=========================.....] - 1252/1500
    Loading: [=========================.....] - 1253/1500
    Loading: [=========================.....] - 1254/1500
    Loading: [=========================.....] - 1255/1500
    Loading: [=========================.....] - 1256/1500
    Loading: [=========================.....] - 1257/1500
    Loading: [=========================.....] - 1258/1500
    Loading: [=========================.....] - 1259/1500
    Loading: [=========================.....] - 1260/1500
    Loading: [=========================.....] - 1261/1500
    Loading: [=========================.....] - 1262/1500
    Loading: [=========================.....] - 1263/1500
    Loading: [=========================.....] - 1264/1500
    Loading: [=========================.....] - 1265/1500
    Loading: [=========================.....] - 1266/1500
    Loading: [=========================.....] - 1267/1500
    Loading: [=========================.....] - 1268/1500
    Loading: [=========================.....] - 1269/1500
    Loading: [=========================.....] - 1270/1500
    Loading: [=========================.....] - 1271/1500
    Loading: [=========================.....] - 1272/1500
    Loading: [=========================.....] - 1273/1500
    Loading: [=========================.....] - 1274/1500
    Loading: [=========================.....] - 1275/1500
    Loading: [=========================.....] - 1276/1500
    Loading: [=========================.....] - 1277/1500
    Loading: [=========================.....] - 1278/1500
    Loading: [=========================.....] - 1279/1500
    Loading: [=========================.....] - 1280/1500
    Loading: [=========================.....] - 1281/1500
    Loading: [=========================.....] - 1282/1500
    Loading: [=========================.....] - 1283/1500
    Loading: [=========================.....] - 1284/1500
    Loading: [=========================.....] - 1285/1500
    Loading: [=========================.....] - 1286/1500
    Loading: [=========================.....] - 1287/1500
    Loading: [=========================.....] - 1288/1500
    Loading: [=========================.....] - 1289/1500
    Loading: [=========================.....] - 1290/1500
    Loading: [=========================.....] - 1291/1500
    Loading: [=========================.....] - 1292/1500
    Loading: [=========================.....] - 1293/1500
    Loading: [=========================.....] - 1294/1500
    Loading: [=========================.....] - 1295/1500
    Loading: [=========================.....] - 1296/1500
    Loading: [=========================.....] - 1297/1500
    Loading: [=========================.....] - 1298/1500
    Loading: [=========================.....] - 1299/1500
    Loading: [==========================....] - 1300/1500
    Loading: [==========================....] - 1301/1500
    Loading: [==========================....] - 1302/1500
    Loading: [==========================....] - 1303/1500
    Loading: [==========================....] - 1304/1500
    Loading: [==========================....] - 1305/1500
    Loading: [==========================....] - 1306/1500
    Loading: [==========================....] - 1307/1500
    Loading: [==========================....] - 1308/1500
    Loading: [==========================....] - 1309/1500
    Loading: [==========================....] - 1310/1500
    Loading: [==========================....] - 1311/1500
    Loading: [==========================....] - 1312/1500
    Loading: [==========================....] - 1313/1500
    Loading: [==========================....] - 1314/1500
    Loading: [==========================....] - 1315/1500
    Loading: [==========================....] - 1316/1500
    Loading: [==========================....] - 1317/1500
    Loading: [==========================....] - 1318/1500
    Loading: [==========================....] - 1319/1500
    Loading: [==========================....] - 1320/1500
    Loading: [==========================....] - 1321/1500
    Loading: [==========================....] - 1322/1500
    Loading: [==========================....] - 1323/1500
    Loading: [==========================....] - 1324/1500
    Loading: [==========================....] - 1325/1500
    Loading: [==========================....] - 1326/1500
    Loading: [==========================....] - 1327/1500
    Loading: [==========================....] - 1328/1500
    Loading: [==========================....] - 1329/1500
    Loading: [==========================....] - 1330/1500
    Loading: [==========================....] - 1331/1500
    Loading: [==========================....] - 1332/1500
    Loading: [==========================....] - 1333/1500
    Loading: [==========================....] - 1334/1500
    Loading: [==========================....] - 1335/1500
    Loading: [==========================....] - 1336/1500
    Loading: [==========================....] - 1337/1500
    Loading: [==========================....] - 1338/1500
    Loading: [==========================....] - 1339/1500
    Loading: [==========================....] - 1340/1500
    Loading: [==========================....] - 1341/1500
    Loading: [==========================....] - 1342/1500
    Loading: [==========================....] - 1343/1500
    Loading: [==========================....] - 1344/1500
    Loading: [==========================....] - 1345/1500
    Loading: [==========================....] - 1346/1500
    Loading: [==========================....] - 1347/1500
    Loading: [==========================....] - 1348/1500
    Loading: [==========================....] - 1349/1500
    Loading: [===========================...] - 1350/1500
    Loading: [===========================...] - 1351/1500
    Loading: [===========================...] - 1352/1500
    Loading: [===========================...] - 1353/1500
    Loading: [===========================...] - 1354/1500
    Loading: [===========================...] - 1355/1500
    Loading: [===========================...] - 1356/1500
    Loading: [===========================...] - 1357/1500
    Loading: [===========================...] - 1358/1500
    Loading: [===========================...] - 1359/1500
    Loading: [===========================...] - 1360/1500
    Loading: [===========================...] - 1361/1500
    Loading: [===========================...] - 1362/1500
    Loading: [===========================...] - 1363/1500
    Loading: [===========================...] - 1364/1500
    Loading: [===========================...] - 1365/1500
    Loading: [===========================...] - 1366/1500
    Loading: [===========================...] - 1367/1500
    Loading: [===========================...] - 1368/1500
    Loading: [===========================...] - 1369/1500
    Loading: [===========================...] - 1370/1500
    Loading: [===========================...] - 1371/1500
    Loading: [===========================...] - 1372/1500
    Loading: [===========================...] - 1373/1500
    Loading: [===========================...] - 1374/1500
    Loading: [===========================...] - 1375/1500
    Loading: [===========================...] - 1376/1500
    Loading: [===========================...] - 1377/1500
    Loading: [===========================...] - 1378/1500
    Loading: [===========================...] - 1379/1500
    Loading: [===========================...] - 1380/1500
    Loading: [===========================...] - 1381/1500
    Loading: [===========================...] - 1382/1500
    Loading: [===========================...] - 1383/1500
    Loading: [===========================...] - 1384/1500
    Loading: [===========================...] - 1385/1500
    Loading: [===========================...] - 1386/1500
    Loading: [===========================...] - 1387/1500
    Loading: [===========================...] - 1388/1500
    Loading: [===========================...] - 1389/1500
    Loading: [===========================...] - 1390/1500
    Loading: [===========================...] - 1391/1500
    Loading: [===========================...] - 1392/1500
    Loading: [===========================...] - 1393/1500
    Loading: [===========================...] - 1394/1500
    Loading: [===========================...] - 1395/1500
    Loading: [===========================...] - 1396/1500
    Loading: [===========================...] - 1397/1500
    Loading: [===========================...] - 1398/1500
    Loading: [===========================...] - 1399/1500
    Loading: [============================..] - 1400/1500
    Loading: [============================..] - 1401/1500
    Loading: [============================..] - 1402/1500
    Loading: [============================..] - 1403/1500
    Loading: [============================..] - 1404/1500
    Loading: [============================..] - 1405/1500
    Loading: [============================..] - 1406/1500
    Loading: [============================..] - 1407/1500
    Loading: [============================..] - 1408/1500
    Loading: [============================..] - 1409/1500
    Loading: [============================..] - 1410/1500
    Loading: [============================..] - 1411/1500
    Loading: [============================..] - 1412/1500
    Loading: [============================..] - 1413/1500
    Loading: [============================..] - 1414/1500
    Loading: [============================..] - 1415/1500
    Loading: [============================..] - 1416/1500
    Loading: [============================..] - 1417/1500
    Loading: [============================..] - 1418/1500
    Loading: [============================..] - 1419/1500
    Loading: [============================..] - 1420/1500
    Loading: [============================..] - 1421/1500
    Loading: [============================..] - 1422/1500
    Loading: [============================..] - 1423/1500
    Loading: [============================..] - 1424/1500
    Loading: [============================..] - 1425/1500
    Loading: [============================..] - 1426/1500
    Loading: [============================..] - 1427/1500
    Loading: [============================..] - 1428/1500
    Loading: [============================..] - 1429/1500
    Loading: [============================..] - 1430/1500
    Loading: [============================..] - 1431/1500
    Loading: [============================..] - 1432/1500
    Loading: [============================..] - 1433/1500
    Loading: [============================..] - 1434/1500
    Loading: [============================..] - 1435/1500
    Loading: [============================..] - 1436/1500
    Loading: [============================..] - 1437/1500
    Loading: [============================..] - 1438/1500
    Loading: [============================..] - 1439/1500
    Loading: [============================..] - 1440/1500
    Loading: [============================..] - 1441/1500
    Loading: [============================..] - 1442/1500
    Loading: [============================..] - 1443/1500
    Loading: [============================..] - 1444/1500
    Loading: [============================..] - 1445/1500
    Loading: [============================..] - 1446/1500
    Loading: [============================..] - 1447/1500
    Loading: [============================..] - 1448/1500
    Loading: [============================..] - 1449/1500
    Loading: [=============================.] - 1450/1500
    Loading: [=============================.] - 1451/1500
    Loading: [=============================.] - 1452/1500
    Loading: [=============================.] - 1453/1500
    Loading: [=============================.] - 1454/1500
    Loading: [=============================.] - 1455/1500
    Loading: [=============================.] - 1456/1500
    Loading: [=============================.] - 1457/1500
    Loading: [=============================.] - 1458/1500
    Loading: [=============================.] - 1459/1500
    Loading: [=============================.] - 1460/1500
    Loading: [=============================.] - 1461/1500
    Loading: [=============================.] - 1462/1500
    Loading: [=============================.] - 1463/1500
    Loading: [=============================.] - 1464/1500
    Loading: [=============================.] - 1465/1500
    Loading: [=============================.] - 1466/1500
    Loading: [=============================.] - 1467/1500
    Loading: [=============================.] - 1468/1500
    Loading: [=============================.] - 1469/1500
    Loading: [=============================.] - 1470/1500
    Loading: [=============================.] - 1471/1500
    Loading: [=============================.] - 1472/1500
    Loading: [=============================.] - 1473/1500
    Loading: [=============================.] - 1474/1500
    Loading: [=============================.] - 1475/1500
    Loading: [=============================.] - 1476/1500
    Loading: [=============================.] - 1477/1500
    Loading: [=============================.] - 1478/1500
    Loading: [=============================.] - 1479/1500
    Loading: [=============================.] - 1480/1500
    Loading: [=============================.] - 1481/1500
    Loading: [=============================.] - 1482/1500
    Loading: [=============================.] - 1483/1500
    Loading: [=============================.] - 1484/1500
    Loading: [=============================.] - 1485/1500
    Loading: [=============================.] - 1486/1500
    Loading: [=============================.] - 1487/1500
    Loading: [=============================.] - 1488/1500
    Loading: [=============================.] - 1489/1500
    Loading: [=============================.] - 1490/1500
    Loading: [=============================.] - 1491/1500
    Loading: [=============================.] - 1492/1500
    Loading: [=============================.] - 1493/1500
    Loading: [=============================.] - 1494/1500
    Loading: [=============================.] - 1495/1500
    Loading: [=============================.] - 1496/1500
    Loading: [=============================.] - 1497/1500
    Loading: [=============================.] - 1498/1500
    Loading: [=============================.] - 1499/1500
    Finished
    

**Create TACO-1 dataset using 70/10/20 split ratio**


```python
import os
import json
import random
import shutil
from collections import defaultdict

# Constants for dataset configuration
VAL_RATIO = 0.1
TEST_RATIO = 0.2
TRAIN_RATIO = 1 - VAL_RATIO - TEST_RATIO

DATASET_SPLITS = ['train', 'valid', 'test']
SUBDIRECTORIES = ['images', 'labels']
CLASS_NAME = 'litter'
NUM_CLASSES = 1

# Path configuration
ANNOTATIONS_PATH = './TACO/data/annotations.json'
IMAGE_ROOT_DIR = './TACO/data'
OUTPUT_ROOT_DIR = './dataset'


def create_directory_structure(output_root: str) -> None:
    """Create required directory structure for YOLO dataset"""
    for split in DATASET_SPLITS:
        for subdir in SUBDIRECTORIES:
            dir_path = os.path.join(output_root, split, subdir)
            os.makedirs(dir_path, exist_ok=True)


def load_annotation_data(file_path: str) -> tuple:
    """Load COCO annotation data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['images'], data['annotations'], data['categories'], data['info']


def group_annotations_by_image(annotations: list) -> dict:
    """Group annotations by their corresponding image ID"""
    image_annotations_map = defaultdict(list)
    for ann in annotations:
        image_annotations_map[ann['image_id']].append(ann)
    return image_annotations_map


def split_dataset(image_ids: list, seed: int = 1) -> dict:
    """Split image IDs into train, validation, and test sets using random shuffle"""
    random.seed(seed)
    ids = image_ids.copy()
    random.shuffle(ids)
    total = len(ids)
    n_train = int(TRAIN_RATIO * total)
    n_val = int(VAL_RATIO * total)
    train_ids = ids[:n_train]
    valid_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return {
        'train': train_ids,
        'valid': valid_ids,
        'test': test_ids
    }


def convert_bbox_to_yolo(bbox: list, image_width: int, image_height: int) -> list:
    """Convert COCO bbox format (x,y,w,h) to YOLO format (x_center,y_center,width,height normalized)"""
    x, y, width, height = bbox
    x_center = x + width / 2
    y_center = y + height / 2
    return [
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height
    ]


def process_image(image_id: int, image_metadata: dict, annotations: list, 
                 output_root: str, split: str, output_index: int) -> None:
    """Process single image: copy file and create label file"""
    # Generate sequential filename
    new_filename = f"{output_index:06d}.jpg"
    
    # Copy original image
    source_path = os.path.join(IMAGE_ROOT_DIR, image_metadata['file_name'])
    dest_path = os.path.join(output_root, split, 'images', new_filename)
    
    if not os.path.exists(dest_path):
        try:
            shutil.copy(source_path, dest_path)
        except Exception as e:
            print(f"Error copying {source_path}: {e}")
            return
    
    # Prepare label data
    label_lines = []
    img_width = image_metadata['width']
    img_height = image_metadata['height']
    
    for ann in annotations:
        yolo_bbox = convert_bbox_to_yolo(ann['bbox'], img_width, img_height)
        bbox_str = ' '.join(f'{coord:.6f}' for coord in yolo_bbox)
        label_lines.append(f"0 {bbox_str}")  # All classes mapped to 0
    
    # Write label file
    label_path = os.path.join(output_root, split, 'labels', new_filename.replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        f.write('\n'.join(label_lines))


def process_dataset_split(split: str, image_ids: list, image_metadata_map: dict, 
                         image_annotations_map: dict, output_root: str, start_idx: int) -> int:
    """Process entire dataset split (train/valid/test)"""
    processed_count = 0
    for i, img_id in enumerate(image_ids):
        image_metadata = image_metadata_map[img_id]
        annotations = image_annotations_map.get(img_id, [])
        
        process_image(
            image_id=img_id,
            image_metadata=image_metadata,
            annotations=annotations,
            output_root=output_root,
            split=split,
            output_index=start_idx + i
        )
        processed_count += 1
    
    return processed_count


def create_dataset_yaml(output_root: str, info: dict) -> None:
    """Create YOLO dataset configuration file with extra info"""
    yaml_content = f"""# Dataset info:
# year: {info.get('year')}
# version: {info.get('version')}
# description: {info.get('description')}
# contributor: {info.get('contributor')}
# url: {info.get('url')}
# date_created: {info.get('date_created')}

train: ./train/images
val: ./valid/images
test: ./test/images

nc: {NUM_CLASSES}
names: ['{CLASS_NAME}']
title: TACO-1
"""
    config_path = os.path.join(output_root, 'dataset.yaml')
    with open(config_path, 'w') as f:
        f.write(yaml_content)



def create_taco1_dataset():
    # Setup directory structure
    create_directory_structure(OUTPUT_ROOT_DIR)

    # Load and organize data
    images, annotations, _, info = load_annotation_data(ANNOTATIONS_PATH)
    image_metadata_map = {img['id']: img for img in images}
    image_annotations_map = group_annotations_by_image(annotations)

    # Split dataset
    split_ids_local = split_dataset(list(image_metadata_map.keys()), seed=1)

    # Process each split
    current_index = 0
    for split_name in DATASET_SPLITS:
        count = process_dataset_split(
            split=split_name,
            image_ids=split_ids_local[split_name],
            image_metadata_map=image_metadata_map,
            image_annotations_map=image_annotations_map,
            output_root=OUTPUT_ROOT_DIR,
            start_idx=current_index
        )
        current_index += count

    # Create YAML configuration
    create_dataset_yaml(OUTPUT_ROOT_DIR, info)
    print("YOLO dataset conversion complete. Output directory:", OUTPUT_ROOT_DIR)

if not os.path.exists(os.path.join(OUTPUT_ROOT_DIR, 'dataset.yaml')):
    create_taco1_dataset()
else:
    print("Dataset already exists at", os.path.join(OUTPUT_ROOT_DIR, 'dataset.yaml'))


```

    Dataset already exists at ./dataset\dataset.yaml
    

# Download yolo models


```python
from ultralytics import YOLO
import os

MODELS_DIR = "models"

# Define models with their weights and input sizes
PAPER_MODELS = {
    "yolov5n": {"weight": "yolov5n.pt", "size": 640},
    "yolov5s": {"weight": "yolov5s.pt", "size": 640},
    "yolov5n6u": {"weight": "yolov5nu.pt", "size": 1280},  # 1280px nano
    "yolov5s6u": {"weight": "yolov5su.pt", "size": 1280},  # 1280px small
    "yolov8n": {"weight": "yolov8n.pt", "size": 640},
    "yolov8s": {"weight": "yolov8s.pt", "size": 640}
}

# Newer YOLO versions for comparison
NEW_MODELS = {
    "yolov9t": {"weight": "yolov9t.pt", "size": 640},
    "yolov9s": {"weight": "yolov9s.pt", "size": 640},
    "yolov10n": {"weight": "yolov10n.pt", "size": 640},
    "yolov10s": {"weight": "yolov10s.pt", "size": 640},
    "yolo11n": {"weight": "yolo11n.pt", "size": 640},
    "yolo11s": {"weight": "yolo11s.pt", "size": 640},
    "yolo12n": {"weight": "yolo12n.pt", "size": 640},
    "yolo12s": {"weight": "yolo12s.pt", "size": 640}
}

# Combine all models
ALL_MODELS = {**PAPER_MODELS, **NEW_MODELS}

# Create model folder
os.makedirs(MODELS_DIR, exist_ok=True)

# Download all model weights if not already present
for name, params in ALL_MODELS.items():
    weight = params["weight"]
    model_path = f"{MODELS_DIR}/{weight}"
    if not os.path.exists(model_path):
        model = YOLO(weight)
        model.save(model_path)
        if os.path.exists(weight):
            os.remove(weight)
        print(f"✅ Downloaded {name} => {model_path}")
    else:
        print(f"✅ Already exists: {model_path}")

```

    ✅ Already exists: models/yolov5n.pt
    ✅ Already exists: models/yolov5s.pt
    ✅ Already exists: models/yolov5nu.pt
    ✅ Already exists: models/yolov5su.pt
    ✅ Already exists: models/yolov8n.pt
    ✅ Already exists: models/yolov8s.pt
    ✅ Already exists: models/yolov9t.pt
    ✅ Already exists: models/yolov9s.pt
    ✅ Already exists: models/yolov10n.pt
    ✅ Already exists: models/yolov10s.pt
    ✅ Already exists: models/yolo11n.pt
    ✅ Already exists: models/yolo11s.pt
    ✅ Already exists: models/yolo12n.pt
    ✅ Already exists: models/yolo12s.pt
    

**Check for GPU support**


```python
import ultralytics
ultralytics.checks()


```

    Ultralytics 8.3.151  Python-3.12.7 torch-2.7.1+cu128 CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU, 6144MiB)
    Setup complete  (16 CPUs, 15.8 GB RAM, 555.1/952.3 GB disk)
    

**Train the yolo models**


```python
import os
import pandas as pd
from tqdm import tqdm

# =====================
# CONSTANTS 
# =====================

#Parameters
TRAINING_EPOCHS = 100
OPTIMIZER = 'auto'  # Automatic optimizer selection (if supported)
AUG_FLIPUD = 0.5      # Vertical flip probability
AUG_DEGREES = 10      # Rotation range: -10 to +10 degrees
CONF_VALUES = [00.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] #Confidence tuning thresholds

#Dirs
DATASET_PATH = "dataset/dataset.yaml"  # Updated dataset path
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PROGRESS_CSV = os.path.join(RESULTS_DIR, "progress.csv")

# =====================
# TRAINING FUNCTION
# =====================
def train_model(model_name, params):
    """Train a YOLO model with specified parameters"""
    print(f"Training {model_name}")
    
    # Load pretrained model
    model = YOLO(os.path.join(MODELS_DIR, params['weight']))
    
    # Paper-specific augmentations
    augmentations = {
        'flipud': AUG_FLIPUD,
        'degrees': AUG_DEGREES,
    }
    
    # Training parameters
    train_args = {
        'data': DATASET_PATH,
        'epochs': TRAINING_EPOCHS, 
        'imgsz': params['size'],
        'batch': -1,  
        'optimizer': OPTIMIZER,  
        'augment': True,
        'name': f"{model_name}_train",
        'save': True,
        'exist_ok': True,
        **augmentations
    }
    
    # Special handling for 1280px models
    if "6u" in model_name:
        train_args['rect'] = True  # Use rectangular training
    
    # Start training
    results = model.train(**train_args)
    
    return model, results

# =====================
# CONFIDENCE TUNING
# =====================
def tune_confidence(model, model_name):
    """Optimize confidence threshold as per paper methodology"""
    best_map50 = 0
    best_conf = CONF_VALUES[0]
    metrics_list = []

    print(f"\nTuning confidence for {model_name}")
    for conf in tqdm(CONF_VALUES, desc="Testing confidence thresholds"):
        metrics = model.val(data=DATASET_PATH, conf=conf, split='val')
        metrics_list.append({'conf': conf, 'map50': metrics.box.map50})
        if metrics.box.map50 > best_map50:
            best_map50 = metrics.box.map50
            best_conf = conf

    print(f"Best confidence: {best_conf:.3f} (mAP50: {best_map50:.4f})")
    return best_conf, metrics_list

# =====================
# EVALUATION FUNCTION
# =====================
def evaluate_model(model, model_name, conf_threshold):
    """Evaluate model on test set with tuned confidence"""
    metrics = model.val(
        data=DATASET_PATH,
        conf=conf_threshold,
        split='test',
        name=f"{model_name}_test"
    )
    
    # Get model size
    ckpt_path = model.ckpt_path if hasattr(model, 'ckpt_path') else model.predictor.model.ckpt_path
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    
    return {
        'model': model_name,
        'size_mb': size_mb,
        'conf_threshold': conf_threshold,
        'map50': metrics.box.map50,
        'map50_95': metrics.box.map
    }

# =====================
# TRAINING PIPELINE 
# =====================
def train_all_models():
    # Load progress if exists
    if os.path.exists(PROGRESS_CSV):
        progress_df = pd.read_csv(PROGRESS_CSV)
        completed_models = set(progress_df['model'])
        results = progress_df.to_dict('records')
        print(f"Resuming. Already completed: {completed_models}")
    else:
        completed_models = set()
        results = []

    for model_name, params in ALL_MODELS.items():
        if model_name in completed_models:
            print(f"Skipping {model_name}, already completed.")
            continue
        try:
            model, train_results = train_model(model_name, params)
            best_conf, conf_metrics = tune_confidence(model, model_name)
            metrics = evaluate_model(model, model_name, best_conf)
            metrics['conf_tuning_metrics'] = conf_metrics  # Store all map50 values
            results.append(metrics)
            pd.DataFrame(results).to_csv(PROGRESS_CSV, index=False)
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {str(e)}")
            continue
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "final_results.csv"), index=False)
    return results_df

#train the models
results_df = train_all_models()

```

    Resuming. Already completed: {'yolov10s', 'yolo12n', 'yolov5s', 'yolo11n', 'yolov8n', 'yolov8s', 'yolov10n', 'yolov5s6u', 'yolov5n', 'yolo11s', 'yolo12s', 'yolov9t', 'yolov9s', 'yolov5n6u'}
    Skipping yolov5n, already completed.
    Skipping yolov5s, already completed.
    Skipping yolov5n6u, already completed.
    Skipping yolov5s6u, already completed.
    Skipping yolov8n, already completed.
    Skipping yolov8s, already completed.
    Skipping yolov9t, already completed.
    Skipping yolov9s, already completed.
    Skipping yolov10n, already completed.
    Skipping yolov10s, already completed.
    Skipping yolo11n, already completed.
    Skipping yolo11s, already completed.
    Skipping yolo12n, already completed.
    Skipping yolo12s, already completed.
    

**Printing results**


```python
# import numpy as np
# import matplotlib.pyplot as plt

# def print_results(results_df):
#     print("\n\033[1mTraining Complete! Results:\033[0m")
#     print(results_df)

#     # Plot mAP50 vs Model Size
#     plt.figure(figsize=(12, 7))
#     x = results_df['map50']
#     y = results_df['size_mb']
#     labels = results_df['model']

#     plt.scatter(x, y, s=120, alpha=0.7)

#     # Offset labels to avoid overlap
#     for i, label in enumerate(labels):
#         # Alternate label positions for better separation
#         dx = 0.002 if i % 2 == 0 else -0.002
#         dy = 0.1 if i % 3 == 0 else -0.1
#         plt.annotate(label, (x.iloc[i] + dx, y.iloc[i] + dy), fontsize=10, alpha=0.9)

#     plt.title('mAP50 vs Model Size (TACO Dataset)', fontsize=15)
#     plt.xlabel('mAP50', fontsize=13)
#     plt.ylabel('Model Size (MB)', fontsize=13)
#     plt.yscale('log')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(os.path.join(RESULTS_DIR, "map50_vs_size.png"), dpi=300)
#     plt.show()

#     # Comparison table
#     paper_results = results_df[results_df['model'].isin(PAPER_MODELS.keys())]
#     new_results = results_df[results_df['model'].isin(NEW_MODELS.keys())]
#     print("\n\033[1mPaper Models vs New Models:\033[0m")
#     print(pd.concat([paper_results, new_results]).sort_values('map50', ascending=False))

# print_results(results_df)

```

## Results


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import seaborn as sns

# Let's load the annotations from the TACO dataset.

with open(ANNOTATIONS_PATH, "r") as f:
	taco_json = json.load(f)

# Convert to DataFrame for easier handling
images_df = pd.DataFrame(taco_json["images"])
annotations_df = pd.DataFrame(taco_json["annotations"])

# # Merge image file names into annotations
annotations = annotations_df.merge(images_df[["id", "file_name", "width", "height"]], left_on="image_id", right_on="id")
image_paths = annotations["file_name"].unique()

```

**Figure 2: Bounding Box Size Distribution**


```python
def plot_bbox_distribution(annotations_df):
    # Extract bbox width and height from the bbox column
    bbox_array = np.stack(annotations_df['bbox'].apply(lambda b: np.array(b)))
    bbox_w = bbox_array[:, 2]
    bbox_h = bbox_array[:, 3]
    img_w = annotations_df['width'].values
    img_h = annotations_df['height'].values

    # Calculate normalized diagonal
    diagonal_norm = np.sqrt((bbox_w / img_w) ** 2 + (bbox_h / img_h) ** 2)

    # Plot histogram
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.histplot(diagonal_norm, bins=100)
    plt.xlabel("Bounding box diagonal size [relative to image long side]")
    plt.ylabel("Frequency [%]")


    # Plot CDF
    plt.subplot(122)
    sns.ecdfplot(diagonal_norm)
    plt.xlabel("Bounding box diagonal size [relative to image long side]")
    plt.ylabel("Frequency [%]")
    plt.savefig("bbox_size_distribution.png", dpi=300)

plot_bbox_distribution(annotations)
```


    
![png](assets/images/rmai-yolo-comparison_20_0.png)
    


**Figure 3: mAP50 vs Model Size (Pareto Plot)**


```python
import ast

def prepare_performance_data(final_results):
    """
    Prepare a DataFrame for performance comparison plots.
    Returns columns: model, model_size_mb, conf, mAP50
    """
    # Extract mAP50 at conf=0.001 and conf=0.5 for each model
    rows = []
    for row in final_results.itertuples():
        conf_metrics = ast.literal_eval(row.conf_tuning_metrics)
        conf_map = {m["conf"]: m["map50"] for m in conf_metrics}
        for conf in [0.001, 0.5]:
            map50 = conf_map.get(conf)
            rows.append({
                "model": row.model,
                "size_mb": row.size_mb,
                "conf_threshold": conf,
                "map50": map50,
                "map50_95": row.map50_95
            })
    plot_df = pd.DataFrame(rows)
    return plot_df

def plot_performance_comparison(results_df):
    # Define model groups and colors
    paper_models = ["yolov5n", "yolov5s", "yolov5n6u", "yolov5s6u", "yolov8n", "yolov8s"]
    new_models = ["yolov9t", "yolov9s", "yolov10n", "yolov10s", "yolo11n", "yolo11s", "yolo12n", "yolo12s"]

    color_map = {
        (False, 0.5): "#1f77b4",  # Paper models, conf=0.001 (blue)
        (False, 0.001): "#aec7e8",    # Paper models, conf=0.5 (light blue)
        (True, 0.5): "#d62728",   # New models, conf=0.001 (red)
        (True, 0.001): "#ff9896",     # New models, conf=0.5 (light red)
    }

    plt.figure(figsize=(10, 6))
    for conf in [0.001, 0.5]:
        for is_paper in [True, False]:
            if is_paper:
                models = paper_models
                label = f"YOLOv5/8 (paper), conf={conf}"
            else:
                models = new_models
                label = f"YOLOv9-12 (ours), conf={conf}"
            df_group = results_df[(results_df["model"].isin(models)) & (results_df["conf_threshold"] == conf)]
            if not df_group.empty:
                plt.scatter(
                    df_group["map50"],
                    df_group["size_mb"],
                    label=label,
                    s=100,
                    color=color_map[(is_paper, conf)]
                )
                # Add labels for each point, all at right top
                for _, row in df_group.iterrows():
                    model_label = row["model"]
                    if model_label.startswith("yolov"):
                        model_label = model_label.replace("yolov", "")
                    elif model_label.startswith("yolo"):
                        model_label = model_label.replace("yolo", "")
                    plt.annotate(
                        model_label,
                        (row["map50"], row["size_mb"]),
                        xytext=(-10, 15),
                        textcoords='offset points',
                        fontsize=9,
                        ha='left',
                        va='top',
                        alpha=0.8
                    )
    plt.ylabel("Model Size (MB)", fontsize=12)
    plt.xlabel("mAP50 [%]", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300)

# Load the final results CSV
final_results = pd.read_csv(os.path.join(RESULTS_DIR, "final_results.csv"))
performance_data = prepare_performance_data(final_results)
plot_performance_comparison(performance_data)
```


    
![png](assets/images/rmai-yolo-comparison_22_0.png)
    


**Table I: Quantitative Results**


```python
from IPython.display import display

def make_table1(performance_data):
    def format_bold(val, best):
        """Return bold string if val is the best, else plain string."""
        return f"**{val:.1f}**" if val == best else f"{val:.1f}"

    # Model order for sorting
    model_order = [
        "yolov5n", "yolov5s", "yolov5n6u", "yolov5s6u", "yolov8n", "yolov8s",
        "yolov9t", "yolov9s", "yolov10n", "yolov10s", "yolo11n", "yolo11s", "yolo12n", "yolo12s"
    ]

    # Find best mAP50 and mAP50-95 for tuned confidence
    best_map50 = performance_data[performance_data["conf_threshold"] == 0.5]["map50"].max()
    best_map50_95 = performance_data[performance_data["conf_threshold"] == 0.5]["map50_95"].max()

    # Build table rows for YOLO models (this paper)
    table_rows = []
    for model in model_order:
        df_def = performance_data[(performance_data["model"] == model) & (performance_data["conf_threshold"] == 0.001)]
        df_tuned = performance_data[(performance_data["model"] == model) & (performance_data["conf_threshold"] == 0.5)]
        if df_def.empty or df_tuned.empty:
            continue
        row = {}
        row["Method"] = model.replace("yolov", "YOLO-v").replace("yolo", "YOLO-")
        row["Dataset"] = "TACO"
        row["Default mAP50"] = df_def["map50"].values[0] * 100
        row["Default mAP50-95"] = df_def["map50_95"].values[0] * 100
        row["Tuned mAP50"] = df_tuned["map50"].values[0] * 100
        row["Tuned mAP50-95"] = df_tuned["map50_95"].values[0] * 100
        row["Improvement mAP50"] = row["Tuned mAP50"] - row["Default mAP50"]
        row["Improvement mAP50-95"] = row["Tuned mAP50-95"] - row["Default mAP50-95"]
        row["Size (MB)"] = df_tuned["size_mb"].values[0]
        table_rows.append(row)

    # Convert to DataFrame and keep order
    table1_df = pd.DataFrame(table_rows)
    # Format for display (bold best results)
    table1_df["Tuned mAP50"] = table1_df["Tuned mAP50"].apply(lambda x: format_bold(x, best_map50*100))
    table1_df["Tuned mAP50-95"] = table1_df["Tuned mAP50-95"].apply(lambda x: format_bold(x, best_map50_95*100))

    table1_df = table1_df[[
        "Method", "Dataset", "Default mAP50", "Default mAP50-95",
        "Tuned mAP50", "Tuned mAP50-95", "Improvement mAP50", "Improvement mAP50-95", "Size (MB)"
    ]]
    table1_df.reset_index(drop=True, inplace=True)
    return table1_df

# Usage:
table1_df = make_table1(performance_data)
display(table1_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Method</th>
      <th>Dataset</th>
      <th>Default mAP50</th>
      <th>Default mAP50-95</th>
      <th>Tuned mAP50</th>
      <th>Tuned mAP50-95</th>
      <th>Improvement mAP50</th>
      <th>Improvement mAP50-95</th>
      <th>Size (MB)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YOLO-v5n</td>
      <td>TACO</td>
      <td>48.746105</td>
      <td>48.468031</td>
      <td>57.0</td>
      <td>48.5</td>
      <td>8.234024</td>
      <td>0.0</td>
      <td>5.303351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YOLO-v5s</td>
      <td>TACO</td>
      <td>50.024517</td>
      <td>45.429401</td>
      <td>58.7</td>
      <td>45.4</td>
      <td>8.662242</td>
      <td>0.0</td>
      <td>17.718757</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YOLO-v5n6u</td>
      <td>TACO</td>
      <td>52.461561</td>
      <td>40.484289</td>
      <td>60.2</td>
      <td>40.5</td>
      <td>7.775585</td>
      <td>0.0</td>
      <td>5.303351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>YOLO-v5s6u</td>
      <td>TACO</td>
      <td>51.353071</td>
      <td>42.825591</td>
      <td>57.8</td>
      <td>42.8</td>
      <td>6.419629</td>
      <td>0.0</td>
      <td>17.718757</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YOLO-v8n</td>
      <td>TACO</td>
      <td>50.754182</td>
      <td>49.067595</td>
      <td>59.9</td>
      <td>49.1</td>
      <td>9.099530</td>
      <td>0.0</td>
      <td>6.244724</td>
    </tr>
    <tr>
      <th>5</th>
      <td>YOLO-v8s</td>
      <td>TACO</td>
      <td>52.600015</td>
      <td>50.182684</td>
      <td>60.1</td>
      <td>**50.2**</td>
      <td>7.533207</td>
      <td>0.0</td>
      <td>21.540684</td>
    </tr>
    <tr>
      <th>6</th>
      <td>YOLO-v9t</td>
      <td>TACO</td>
      <td>50.997580</td>
      <td>44.280893</td>
      <td>62.1</td>
      <td>44.3</td>
      <td>11.145485</td>
      <td>0.0</td>
      <td>4.737559</td>
    </tr>
    <tr>
      <th>7</th>
      <td>YOLO-v9s</td>
      <td>TACO</td>
      <td>52.960342</td>
      <td>44.342685</td>
      <td>**62.3**</td>
      <td>44.3</td>
      <td>9.331466</td>
      <td>0.0</td>
      <td>14.674327</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YOLO-v10n</td>
      <td>TACO</td>
      <td>47.880714</td>
      <td>45.272867</td>
      <td>58.9</td>
      <td>45.3</td>
      <td>11.014775</td>
      <td>0.0</td>
      <td>5.586822</td>
    </tr>
    <tr>
      <th>9</th>
      <td>YOLO-v10s</td>
      <td>TACO</td>
      <td>51.690823</td>
      <td>47.666096</td>
      <td>60.8</td>
      <td>47.7</td>
      <td>9.075445</td>
      <td>0.0</td>
      <td>15.850959</td>
    </tr>
    <tr>
      <th>10</th>
      <td>YOLO-11n</td>
      <td>TACO</td>
      <td>49.142446</td>
      <td>48.355927</td>
      <td>58.6</td>
      <td>48.4</td>
      <td>9.469307</td>
      <td>0.0</td>
      <td>5.353580</td>
    </tr>
    <tr>
      <th>11</th>
      <td>YOLO-11s</td>
      <td>TACO</td>
      <td>51.711894</td>
      <td>41.549391</td>
      <td>60.9</td>
      <td>41.5</td>
      <td>9.187629</td>
      <td>0.0</td>
      <td>18.418888</td>
    </tr>
    <tr>
      <th>12</th>
      <td>YOLO-12n</td>
      <td>TACO</td>
      <td>50.552329</td>
      <td>48.418257</td>
      <td>60.1</td>
      <td>48.4</td>
      <td>9.524900</td>
      <td>0.0</td>
      <td>5.332874</td>
    </tr>
    <tr>
      <th>13</th>
      <td>YOLO-12s</td>
      <td>TACO</td>
      <td>49.687166</td>
      <td>49.118030</td>
      <td>58.6</td>
      <td>49.1</td>
      <td>8.954457</td>
      <td>0.0</td>
      <td>18.122730</td>
    </tr>
  </tbody>
</table>
</div>

