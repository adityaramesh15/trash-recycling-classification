import os
import random
from collections import defaultdict
import shutil
from PIL import Image


'''
This is a function that serves as a 'run-all' script, to be called once prior to model train, validation, test split. 
Every image in every category will be visited and a series of preprocessing functions will be run run on said image. 
'''
def preprocess():
    create_processed_data() #create processed-data
    same_count() #convert that processed-data to have 440 per subcategory

    # Goal is to loop through each image and call functions on each image
    for category in os.listdir('processed-data'):
        category_path = os.path.join('processed-data', category)
        
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                try:
                    with Image.open(os.path.join('processed-data', category, img_file)) as im:
                        im.verify()
                        '''
                        *****************************************************************
                        ADD FUNCTIONS HERE TO BE CALLED ON EACH IMAGE IN SUCCESSIVE ORDER 
                        *****************************************************************
                        '''
                except (IOError, OSError, Image.UnidentifiedImageError) as e:
                    ...



'''
This is a function used to create a folder called 'processed-data' which will originally contain a mirror of cleaned-data.
'processed-data' is to be used as a live directory upon which the preprocessing functions will reference and act upon. 
Important to note that 'processed-data' will not be uploaded to github, but instead be created on each instance of model training. 
'''
def create_processed_data():
    if os.path.isdir('processed-data'):
        shutil.rmtree('processed-data')
    
    shutil.copytree('cleaned-data', 'processed-data')


'''
This is a function to assure all counts of images per category are the same, avoiding model bias. 
Using EDA, the ideal amount is set to 440 for this dataset, which will be a hard-coded value. 
'''
def same_count():
    image_data = defaultdict(list)
    target_count = 440

    for category in os.listdir('cleaned-data'):
        category_path = os.path.join('cleaned-data', category)
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                if img_file.endswith('.jpg'):
                    image_data[category].append(os.path.join(category_path, img_file))

    for category, images in image_data.items():
        sampled_images = random.sample(images, target_count)
        category_subdir = os.path.join('processed-data', category)

        if os.path.exists(category_subdir):
            shutil.rmtree(category_subdir)
        os.makedirs(category_subdir)

        for img_path in sampled_images:
            shutil.copy(img_path, category_subdir)

