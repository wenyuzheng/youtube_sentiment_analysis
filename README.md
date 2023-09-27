# Sentiment Analysis of YouTube comments using XLM-R and mBERT

## Installation

Make sure you have Python (3.8 onwards) installed.

### 1. Install a virtual python environment

Init the project:

```
make init
```

This will install a virtual python environment `venv`.

### 2. Source venv

Start using the virtual environment:

```
make activate
```

### 3. Install other local dependencies

Install PyTorch based on your system https://pytorch.org/get-started/locally/.

Install all other local dependencies:

```
make install_deps
```

Dependencies listed in requirements.txt will be installed.

### 4. Download trained models

Due to the large size of trained model files, the models are uploaded in One Drive. Download the trained models, XLM-R and mBERT, using the following link: https://emckclac-my.sharepoint.com/:f:/g/personal/k1924737_kcl_ac_uk/EirZWRjoRaBHqntXmj_NkgYBiu7c7yd7Q5sVqrL0mYry2g?e=ZNLhUf

Put the two model files into the directory named 'sentiments'

### 5. Edit credentials

Due to data security issue, YouTube Data API key is not shared. You need to create your own API key https://developers.google.com/youtube/v3/getting-started. Please follow the provided user guide.

After API key creation, add the API key into the file named 'credentials.json'.

## Testing

Test data cleaning methods:

```
make test
```

## Usage

To analyse a particular YouTube video:

```
python analyse.py --video_url <youtube video url>
```

To see the sanitised data after each data cleaning step:

```
python analyse.py --verbose=true
```

To control how many pages of comments to retrieve:

```
python analyse.py --max_pages <number of pages>
```

To change the order of comments:

```
python analyse.py --order <order of comments>
```
