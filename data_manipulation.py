import os
import random
from collections import defaultdict
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageStat

'''
This is a script used to generate percentage-based sample data to preserve relative counts. 
Use a base path with flat directory (each folder is one feature, "featured-data" does this already).
Everytime this script is run, write-path folder will get wiped and recreated with new data. 
'''
def random_sample(base_path, write_path, sample_proportion):
    if os.path.isdir(write_path):
        shutil.rmtree(write_path)
    os.makedirs(write_path)

    image_data = defaultdict(list)

    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                if img_file.endswith('.jpg'):
                    image_data[category].append(os.path.join(category_path, img_file))

    sample_sizes = {category: int(len(files) * sample_proportion) for category, files in image_data.items()}
    sampled_images = {}


    for category, images in image_data.items():
        category_subdir = os.path.join(write_path, category)
        os.makedirs(category_subdir)
        sample_size = sample_sizes[category]
        sampled_images[category] = random.sample(images, sample_size)

        for img_path in sampled_images[category]:
            destination_path = os.path.join(write_path, category)
            shutil.copy(img_path, destination_path)


'''
This is a function used to convert our data into a DataFrame for EDA and Preprocessing purposes. 
Feature Data being collected includes, width, height, aspect_ratio, mean/std pixel values. 
This is before preprcoessing, so Numpy NaN data is inputted anywhere the image is corrupted. 
'''    
def to_dataframe(base_path): 
    categories = []
    names = []
    widths = []
    heights = []
    aspect_ratios = []
    mean_pixel_values = []
    std_pixel_values = []

    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                try:
                    with Image.open(os.path.join(base_path, category, img_file)) as im:
                        im.verify()
                        
                        width, height = im.size
                        aspect_ratio = width / height

                        categories.append(category)
                        widths.append(width)
                        heights.append(height)
                        names.append(img_file)
                        aspect_ratios.append(aspect_ratio)

                        with Image.open(os.path.join(base_path, category, img_file)) as im:
                            im = im.convert('RGB') 
                            stat = ImageStat.Stat(im)
                            mean_pixel_values.append(np.mean(stat.mean))
                            std_pixel_values.append(np.mean(stat.stddev))

                except (IOError, OSError, Image.UnidentifiedImageError) as e:
                    categories.append(category)
                    names.append(img_file)
                    widths.append(np.nan)
                    heights.append(np.nan)
                    aspect_ratios.append(np.nan)
                    mean_pixel_values.append(np.nan)
                    std_pixel_values.append(np.nan)

    multi_index = pd.MultiIndex.from_arrays([categories, names], names=('Category', 'Name'))
    df = pd.DataFrame({'Width': widths, 'Height': heights, 'Aspect Ratio': aspect_ratios, 
                       "Mean": mean_pixel_values, "Standard Deviation": std_pixel_values}, index=multi_index)
    return df
