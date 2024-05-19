<h1 align="center">DeepDebugger: Touch what you image</h1>
Official source code for ESEC/FSE 2023 Paper: 
<strong>DeepDebugger: An Interactive Time-Travelling Debugging Approach for Deep Classifiers</strong>

<p align="left">
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
  </p>
</p>

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)
- [Citation](#citation)

## Installation
### Dependencies
Please run the following commands in the command line:
```console
$ conda create -n dd python=3.8
$ git clone https://github.com/xianglinyang/DeepDebugger.git
$ pip -r install requirements.txt
```
To install torch, please check [link](https://pytorch.org/get-started/locally/).

## Usage

### Quick Start
> A tutoial in jupyter notebook is [here](tutorials/quick-start.py). 

Generally, we follow the following process to create meaningful visualization:
1. Prepare data according to our format. 
    - Use our Summary writer.  [[tutorial1](tutorials/1-summary-writer.ipynb)] [[tutorial2](tutorials/1-example.ipynb)]
    - Do it in a manual way. See [data-arrangement](https://github.com/xianglinyang/DeepDebugger/wiki/Data-Arrangement).

    Put all data under a folder `content_path = /path/to/data`
2. Visualize the embedding
Choose a visualization strategy (e.g., DVI, TimeVis or even your own visualization method!)
    - Use wrapped func from [Strategy](strategy.py) class
      ```python
      #--------DVI--------
      VIS_METHOD = "DVI"
      dvi_config = config[VIS_METHOD]
      dvi = DeepVisualInsight(CONTENT_PATH, dvi_config)
      dvi.visualize_embedding()

      #--------TimeVis--------
      VIS_METHOD = "TimeVis"
      timevis_config = config[VIS_METHOD]
      timevis = TimeVis(CONTENT_PATH, timevis_config)
      timevis.visualize_embedding()
      ```
    - Directly call the module.
      ```console
      $ python dvi_main.py --content_path path/to/data
      $ python timevis_main.py --content_path path/to/data
      ```
3. Play with embedding visualization with our frontend or backend visualizer. [[repo](https://github.com/llmhyy/training-visualizer/)][[tutorial](tutorials/2-start-services.md)]

4. (optional) Design your own visualization method. [[tutorial](tutorials/3-customize-visualization.ipynb)]

### Full instructions
Please see our [wiki]([https://github.com/xianglinyang/DeepDebugger/wiki](https://github.com/xianglinyang/DeepDebugger/wiki/How-to-use-DeepDebugger)) for more details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions, please feel free to reach out to me at xianglin@u.nus.edu.

## Reproducibility
Follow batch_run.py run.sh to run the codes and reproduce the published results.
```python
python batch_run.py
```

## Citation
If you find our tool helpful, please cite the following paper:
```bibtex
@inproceedings{yang2023deepdebugger,
  title={DeepDebugger: An Interactive Time-Travelling Debugging Approach for Deep Classifiers},
  author={Yang, Xianglin and Lin, Yun and Zhang, Yifan and Huang, Linpeng and Dong, Jin Song and Mei, Hong},
  booktitle={Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  pages={973--985},
  year={2023}
},
@inproceedings{yang2022temporality,
  title={Temporality Spatialization: A Scalable and Faithful Time-Travelling Visualization for Deep Classifier Training},
  author={Yang, Xianglin and Lin, Yun and Liu, Ruofan and Dong, Jin Song},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI-22}},
  year={2022}
},
@inproceedings{yang2022deepvisualinsight,
  title={DeepVisualInsight: Time-Travelling Visualization for Spatio-Temporal Causality of Deep Classification Training},
  author={Yang, Xianglin and Lin, Yun and Liu, Ruofan and He, Zhenfeng and Wang, Chao and Dong, Jin Song and Mei, Hong},
  booktitle = {The Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI)},
  year={2022}
}
```





