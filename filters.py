import cv2
import numpy as np
from IPython.display import display, Image
# import seam_carving
import time
from PIL import Image as PILImage
from io import BytesIO
import os



"""
This is the main function to use for preprocessing.
takes in an image in PIL format and returns filtered image in PIL format
"""
def preprocess_img(myimage, target_width=255, target_height=255):

	# Convert PIL to opencv
	numpy_image=np.array(myimage)
	myimage=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
	
	# myimage = regular_padding(myimage)
	# original_image = regular_padding(original_image)


	myimage = resize_to_correct_size(myimage)
	# original_image = myimage.copy()

	# print(original_image.shape)

	edges = find_laplacian_white_bg(myimage)

	# reduces noise from edges image
	edges = bgremove3(edges)


	# binned = binning(myimage)
	# blurred_and_binned = gaussianBlur(binned)
	# print(binned.shape)
	overlayed = overlay(edges, myimage)
	

	padded = regular_padding(overlayed, target_width=target_width, target_height=target_height)


	# Convert from opencv to PIL
	padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
	padded = PILImage.fromarray(padded)

	return padded



def show_img(img, width=None, saveAs="to_show.jpg"):
	cv2.imwrite(saveAs,img)
	if width != None:
		return Image(filename=saveAs, width=width)
	return Image(filename=saveAs)

# def resize_image_seam_carving(src):
# 	(h, w) = src.shape[:2]
# 	if w > h:
# 		target_width = 300
# 		aspect_ratio = w / h
# 		target_height = int(target_width / aspect_ratio)
# 	else:
# 		target_height = 300
# 		aspect_ratio = h / w
# 		target_width = int(target_height / aspect_ratio)
# 	src = cv2.resize(src, (target_width, target_height))
# 	start_time = time.time()
# 	src_h, src_w, _ = src.shape
# 	order = "width-first"
	
# 	dst = seam_carving.resize(
# 		src, (255, 255),
# 		energy_mode='backward',   # Choose from {backward, forward}
# 		order=order,  # Choose from {width-first, height-first}
# 		keep_mask=None
# 	)
# 	end_time = time.time()
# 	print(end_time-start_time)
# 	# output = Image.fromarray(dst)
# 	cv2.imwrite("output.png",dst)

# resize_image_seam_carving(img_path)
# Image(filename="output.png", width=300)


# resizes to a size that can be padded later to become 255x255
def resize_to_correct_size(src, target_width=255, target_height=255):
	h, w, channels = src.shape
	if w > h:
		aspect_ratio = w / h
		img_target_height = int(target_width / aspect_ratio)
		img_target_width = target_width
	else:
		aspect_ratio = h / w
		img_target_width = int(target_height / aspect_ratio)
		img_target_height = target_height
	src = cv2.resize(src, (img_target_width, img_target_height))
	return src


def regular_padding(src, target_width=255, target_height=255):
	h, w, channels = src.shape
	# if width > height: resize width to 255 and scale height accordingly
	# if height > width: resize height to 255 and scale width accordingly

	if w > h:
		aspect_ratio = w / h
		img_target_height = int(target_width / aspect_ratio)
		img_target_width = target_width
	else:
		aspect_ratio = h / w
		img_target_width = int(target_height / aspect_ratio)
		img_target_height = target_height
	
	src = cv2.resize(src, (img_target_width, img_target_height))
	# print(src.shape)

	y_center = (target_height - img_target_height) // 2
	x_center = (target_width - img_target_width) // 2

	color = (0,0,0)
	result = np.full((target_height,target_width, channels), color, dtype=np.uint8)

	# # copy img image into center of result image
	# print(y_center + target_height)
	result[y_center:y_center + img_target_height, x_center:x_center + img_target_width] = src

	return result


# myimage = resize_to_correct_size(myimage, 255, 255)
# print(myimage.shape)
# show_img(myimage)
# Image(filename="output.png", width=300)

def gaussianBlur(myimage, radius=5):
	return cv2.GaussianBlur(myimage,(radius,radius), 0)



def dim_img(img, scale_factor=0.5):
	# Convert the image to a float32 type to avoid clipping issues
	image_float = img.astype(np.float32)

	# Scale the pixel values
	image_dimmed = image_float * scale_factor

	# Clip values to stay within the valid range [0, 255]
	image_dimmed = np.clip(image_dimmed, 0, 255)

	# Convert back to uint8 type
	image_dimmed = image_dimmed.astype(np.uint8)
	return image_dimmed


def find_laplacian_white_bg(img, ksize=5):
	# second derivative of the image intensity
	laplacian = cv2.Laplacian(img,cv2.CV_64F, ksize=ksize) 
	show_img(laplacian, saveAs="ignore_this.jpg")

	# was unable to find a different workaround for this
	laplacian = cv2.imread("ignore_this.jpg")
	laplacian = cv2.bitwise_not(laplacian)
	return laplacian

def binning(img, num_bins=5):
	# We bin the pixels. Result will be a value 1..5
	bins = np.linspace(0, 255, num_bins+1)
	# print(bins)
	bins=np.array(bins)
	img[:,:,:] = np.digitize(img[:,:,:],bins,right=True)*51
	return img

def overlay(laplacian, myimage):
	# make sure laplacian has the same resolution as image_dimmed
	# print(laplacian.shape)
	# laplacian = resize_to_correct_size(laplacian, 255, 255)
	# print(laplacian.shape)

	# Create a mask where the pixels of the first image are darker
	gray1 = cv2.cvtColor(laplacian, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)

	# show_img(gray2)

	# Create a mask where the pixels of the first image are darker
	mask = gray1 < gray2
	# Create a 3-channel mask by stacking the single channel mask
	mask_3channel = np.stack([mask] * 3, axis=-1)

	# Overlay a % of the darker pixels from image1 onto image2
	result = np.where(mask_3channel, laplacian, (myimage*0.4+laplacian*0.6))
	return result


def preprocess(img_path):
	myimage =cv2.imread(img_path)
	return preprocess_img(myimage)


# img_path = r"../cleaned-data/plastic containers/plastic_containers 310.jpg"
# result = preprocess(img_path)
# show_img(result)


# https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
def bgremove3(myimage):
    # BG Remover 3
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
    #Take saturation and remove any value that is less than half
    s = myimage_hsv[:,:,1]
    s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded
 
    # We increase the brightness of the image and then mod by 255
    v = (myimage_hsv[:,:,2] + 127) % 255
    v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask
 
    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
 
    background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
    foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    finalimage = background+foreground # Combine foreground and background
    return finalimage
	



def run_realtime():
	# Initialize webcam
	cap = cv2.VideoCapture(0)

	if not cap.isOpened():
		print("Error: Could not open webcam.")
		exit()

	while True:
		# Capture frame-by-frame
		ret, frame = cap.read()
		if not ret:
			print("Error: Could not read frame.")
			break

		# Apply a filter (grayscale)
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# opencv to PIL
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = PILImage.fromarray(frame)

		myimg = preprocess_img(frame, 550, 550)

		# PIL to opencv
		numpy_image=np.array(myimg)
		myimg=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)


		# Display the resulting frame
		cv2.imshow('Grayscale Webcam Feed', myimg)

		# Break the loop on key press
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release the capture and close the window
	cap.release()
	cv2.destroyAllWindows()
	os.remove("ignore_this.jpg")

if __name__ == "__main__":
	run_realtime()
