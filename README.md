# Text Generation With Transformers

This project uses an decoder-only transformer network to generate text given a seed. The architecture largely copied from [this project](https://github.com/pbloem/former). 

## Usage
### With docker
To start *or resume* training:
```
docker run bluesk/nlp train
```
To generate some text from *seed* (optional):
```
docker run bluesk/nlp generate "seed"
```
To save the generated model you might want to mount locally the /models directory like so:
```
docker run bluesk/nlp \
    -v path_to_models_storage:/models \
    train
```

#### Configuration
To change default parameters you can put a `config.json` file into the mounted /models directory (see above). 
Check out the example `config.json` file in this repository. Parameters chan be changed also during training, just interrupt the execution and restart it.

#### Running with custom text corpus
Mount the directory (local_dir) containg the text file and then run:
```
docker run bluesk/nlp \
    -v local_dir:/data/ \
    train /data/file_name.txt
```

### On a linux machine
Clone the repository (in the nlp directory), then:

To start *or resume* training:
```
python -m nlp train path/to/your/text_file.txt
```

To generate some text starting from *seed*
```
python -m nlp generate "seed"
```
#### Configuration
Modify the `config.json` to change default parameters.

## Requirements
If you're not using docker:
- this project is meant to run on a linux machine
- python 3.6+
- pip packages: torch tb-nightly tqdm numpy requests

## Results
With this dataset the model was able to acheave 1.32 bits per bytes on the valitadion set.
