## Setup
Install the required dependencies (needed very specific versions to work together)
```bash
pip install -r requirements.txt
```

Directory Structure:
```bash
tree -L 1
.
├── data
├── demo
├── demo_config
├── generate_annotation_file.py
├── logs
├── mmocr
├── mmocr_tutorial.ipynb
├── notes.txt
├── outputs
├── __pycache__
├── requirements.txt
├── results
├── run.py
├── scripts
├── soccernet-annotated
├── soccernet_dataset.py
└── src
```
On my computer, the official SoccerNet data is in `data/` and our annotated data is in `soccernet-annotated/`.

First, we need to link the annotated dataset to the correct directory in `mmocr/data`. Since my data is already in the `soccernet-annotated` directory I just create a symbolic link in the correct place without duplicating the data.
```bash
ls soccernet-annotated
> README.dataset.txt  README.roboflow.txt  test  train  valid

ln -s soccernet-annotated mmocr/data
```

The pretrained model weights and config file should already be included in the mmocr directory. To visualize the outputs of the finetuned DBNet++ (trained 10 epochs on our annotated dataset):
```bash
cd mmocr
python tools/test.py configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated.py soccernet-dbnetpp/epoch_10.pth --show-dir soccernet-dbnetpp
```
You should see some generated samples in `mmocr/soccernet-dbnetpp/vis_data/vis_image`.

## Running the inferencer with our finetuned model
We can run the inferencer to see some predictions from our finetuned model. We load the finetuned DBNet++ weights and config file as shown below.
```bash
python run.py --data_path data --output_dir outputs --det_config_path mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated.py --det_weights_path mmocr/soccernet-dbnetpp/epoch_10.pth
```
I set the final prediction for each video to be the mode of the predicted numbers, but we can change this in the future. 
If you want to change the recognizer model, you just need to adjust the line with `MMOCRInferencer(..., rec='SAR', ...)`.

This was running very slowly for me on cpu--it is much faster to request a gpu and run the same code. 
```bash
salloc --account=eecs545w24_class --partition=gpu --gpus=1 --mem-per-gpu=8000 --cpus-per-gpu=3 --time=2:00:00
python run.py
```

## Finetuning
To perform our own finetuning, we need to generate annotations in the correct format. Use the script `generate_annotation_file.py` to create these. You only need to do this one time.
```bash
pwd
  > /home/oliveraw/SoccerNet
python generate_annotation_file.py --data_root soccernet-annotated
```
This will generate 3 new files: `soccernet-annotated/test/_mmocr_annotations.json`, `soccernet-annotated/train/_mmocr_annotations.json`, and `soccernet-annotated/valid/_mmocr_annotations.json`.
The MMOCR annotation format can be found [here](https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html#ocrdataset). 

When regenerating the annotations, I removed all the images with no bounding boxes because of some compatibility issues with MMOCR and empty gt. 

For finetuning, we have defined two new configuration files: one for the dataset and one for the model.

The dataset config can be found at `mmocr/configs/textdet/_base_/datasets/soccernet.py`. 

The model config can be found at `mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated.py`.

I used the same DBNet++ base model pretrained on ICDAR 2015, the only modifications were to the Resize size, the number of epochs (10), and the normalization for the images (custom mean, std). 


To perform your own finetuning, you just need to define a new config file and inherit the correct model architecture, optimizer, scheduler, transforms, etc. The hyperparameters, model architecture, learning rate, epochs, etc. can all be modified by changing the appropriate config file.
I recommend reading this [page](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html) to learn more about the inheritance for the config files. 

To run the finetuning:
```bash
cd mmocr
python tools/train.py configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated.py --work-dir soccernet-dbnetpp
```

**You may need to change the data root directory in the config (`mmocr/configs/textdet/_base_/datasets/soccernet.py`) to match your own directory name, since the data root is hardcoded.**

