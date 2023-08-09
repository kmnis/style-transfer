from glob import glob
from PIL import Image

image_files = glob("imgs/*.png")

def numeric_sort_key(filename):
    return int(filename.split("/")[1][:-4])
    
image_files.sort(key=numeric_sort_key)

print(len(image_files))

images = [Image.open(f) for f in image_files]

images[0].save("imgs/demo.gif", save_all=True, append_images=images[1:], duration=200, loop=0)
