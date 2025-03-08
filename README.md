# DeepDanbooru
[![Python](https://img.shields.io/badge/python-3.11-green)](https://www.python.org/doc/versions/)
[![GitHub](https://img.shields.io/github/license/KichangKim/DeepDanbooru)](https://opensource.org/licenses/MIT)
[![Web](https://img.shields.io/badge/web%20demo-20200915-brightgreen)](http://kanotype.iptime.org:8003/deepdanbooru/)

**DeepDanbooru-tagger-mini** is anime-style girl image tag estimation system based on DeepDanbooru


## Requirements
Deepmini is written by Python 3.11. Following packages are need to be installed.
- six~=1.17.0
- scikit-image~=0.25.2
- tensorflow~=2.18.0

Or just use `requirements.txt`.
```
> pip install -r requirements.txt
```

alternatively you can install it with pip. Note that by default, tensorflow is not included.

To install it with tensorflow, add `tensorflow` extra package.

```
> # default installation
> pip install .
> # with tensorflow package
> pip install .[tensorflow]
```


## Usage
```
Prepare dataset. If you don't have, you can use [DanbooruDownloader](https://github.com/KichangKim/DanbooruDownloader) for download the dataset of [Danbooru](https://danbooru.donmai.us/). If you want to make your own dataset, see [Dataset Structure](#dataset-structure) section.

(Your project use deepmini)/
├── bin/
│   └── deepmini/
└── data/
    ├── tagger_model/ # dataset here
    ├──...
    ├── model-resnet_custom_v4.h5(v4e30)
    └── project.json
```
