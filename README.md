# Sign Language Recognition

## Setting up

1. Git clone folder:
git clone https://github.com/victorpham1997/Sign-Language-Recognition-using-Deep-Learning.git
2. Download train dataset:
    Download from here: http://chalearnlap.cvc.uab.es/dataset/40/data/66/files/
3. Extract train dataset:
    Train dataset decompress password: MdG3z6Eh1t
4. Download validation dataset 
    Download from here: http://chalearnlap.cvc.uab.es/dataset/40/data/65/files/
5. Extract val dataset: 
    Val dataset decompress password: bhRY5B9zS2
6. Rename and move the train, val folder to follow: 
    Sign-Language-Recognition-using-Deep-Learning/dataset/train/*all train image here*
    Sign-Language-Recognition-using-Deep-Learning/dataset/val/*all val image here*

## Valve Filter

To use the `valve filter`, we need the carry out pre-processing with `ValveFilterPreprocesser.py`: to pre-process our input data for the generation of ROI (region of interest)

There are 2 commands, which are `--train` and `--val`, which defaults to `./dataset/train` and `./dataset/val` respectively

Example of usage:

```bash
python3 ValveFilterPreProcesser.py
```

## Semantic Segmentation preprocessing

We used the `removebackgrounds_prepocessing.py` to carry semantic segmentation on the original dataset to produce new dataset that has the background removed.

```bash
python3 removebackgrounds_prepocessing.py
```

## Train

We used the respective notebook to train our different architectures. In order to run the notebook the set up step needs to be done to acquire the data.

The notebooks includes: 

- Sign-Language-Recognition-Notebook-baseline
- Sign-Language-Recognition-Notebook-baseline-semantic-seg
- Sign-Language-Recognition-Notebook-Valve-Filter
- Sign-Language-Recognition-Notebook-Transformer

## Inference

To test inference of the **baseline** model run

```bash
python3 main.py --model_name baseline --model_weight ./models/baseline.pt --video ./samples/signer0_sample29_color.mp4 --label ./dataset/train_labels.csv
```

To test inference of the **baseline** model with **semantic segmentation** run

```bash
python3 main.py --model_name baseline --model_weight ./models/baseline_ss.pt --video ./samples/signer0_sample10_color.mp4 --label ./dataset/train_labels.csv
```

To test inference of the **valve filter** model run

```bash
python3 main.py --model_name valveFilter --model_weight ./models/valveFilter.pt --video ./samples/signer0_sample29_color.mp4 --label ./dataset/train_labels.csv 
```

For the transformer model, we did not include the model as its size was too large for git. 

## GUI

To run the GUI module:

1. Install required python libraries
2. Install npm libraries "cd webapp && npm i"
3. Start webapp with "cd webapp && npm run start"
4. Start flask server with "cd server && python3 main.py"