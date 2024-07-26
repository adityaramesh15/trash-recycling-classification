import os
import random
from collections import defaultdict
import shutil
import numpy as np
from PIL import Image, ImageOps
from filters import preprocess_img


'''
This is a function that serves as a 'run-all' script, to be called once prior to model train, validation, test split. 
Every image in every category will be visited and a series of preprocessing functions will be run on said image. 
Make note that for integration, every function will modify the image in-place using image.paste() from PIL
'''
def preprocess(editExistingFolder=False):

    category_size = 440
    if not editExistingFolder:
        create_processed_data()
        same_count(category_size) 
    

    # Goal is to loop through each image and call functions on each image
    for category in os.listdir('processed-data'):
        category_path = os.path.join('processed-data', category)
        print()
        print(f"processing `{category}` category")
        
        img_num = 1
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                print(f"img: {img_num}/{category_size}")
                try:
                    img_path = os.path.join('processed-data', category, img_file)
                    with Image.open(img_path) as im:
                        im.verify()
                        normalize_image(im)
                        histogram_equalization(im)
                        # im = preprocess_img(im)

                        im.save(img_path)
                       
                except (IOError, OSError, Image.UnidentifiedImageError) as e:
                    if(os.path.exists(img_file)):
                        os.remove(img_file)
                img_num += 1


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
Using EDA, the ideal amount is set to 440 for this dataset, but it can be changed depending on dataset. 
'''
def same_count(target_count):
    image_data = defaultdict(list)

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


'''
Normalizes the pixel values of the given PIL image to the range 0-1 for better model interpretability. 
uint8 chosen as it is a standard for image procesing, and float32 as it has a good balance between space and detail. 
Modifies the PIL Image in-place for further changes to be made by reference. 
'''
def normalize_image(image):
    np_image = np.array(image).astype(np.float32)
    np_image /= 255.0
    image.paste(Image.fromarray((np_image * 255).astype(np.uint8)))

'''
Applies histogram equalization to the given PIL image to better enhance the contrast of the image.
Modifies the PIL Image in-place for further changes to be made by reference. 
'''
def histogram_equalization(image):
    grayscale_image = ImageOps.grayscale(image)
    equalized_image = ImageOps.equalize(grayscale_image)
    if image.mode == 'RGB':
        equalized_image = ImageOps.colorize(equalized_image, black="black", white="white")
    image.paste(equalized_image)

if __name__ == "__main__":
    print("running preprocess")
    yn = input("edit existing folder? (y/n): ")
    if yn != "y" and yn != "n": print("invalid input")
    else: preprocess(editExistingFolder=(yn == "y")); print("done!")