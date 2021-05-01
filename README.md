# Sign Language Recognition

Step to set up working directory:
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
  
  
# Using `ValveFilter.py`
There are 3 required commands, which is `--model`, `--video` and `--labels`, which takes in the saved model file, sample video and ground truth labels. The model files can be found in `.\models` and the sample files can be found in `.\samples`.

Example of usage:
`py -3 .\ValveFilter.py --model '.\models\transformer-2021-04-18 11_09_44.951672-b1-e9.pt' --video .\samples\signer1_sample13_color.mp4 --labels .\dataset\val_labels.csv`
