import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import skimage.transform as tf
from pystackreg import StackReg
from skimage.transform import rescale, resize
img_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Foredrag\\dog.png"
img = cv2.imread(img_path,1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# newimg = rescale(rotate_image(img, 45), 4, order = 3)
newimg = resize(rotate_image(img, 45), (), order = 3)

print(newimg.shape)
sr = StackReg(StackReg.SCALED_ROTATION)
newnewimg = sr.register_transform(img, newimg)


plt.subplot(121)
plt.imshow(img, cmap = "gray")
plt.subplot(122)
plt.imshow(newimg, cmap = "gray")
plt.show()



"""print(img.shape)

new_img = np.zeros((img.shape))
theta = 30/180*np.pi

x = []
y = []
coor = np.zeros((img.shape))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        x = round(i*np.cos(theta) - j * np.sin(theta))
        y = round(i*np.sin(theta) + j * np.sin(theta))
        #print(x,y)
        if abs(x) < img.shape[0] - 1 and abs(y) < img.shape[1] - 1:
            new_img[x,y] = img[x,y]

print(img.shape)
plt.subplot(121)
plt.imshow(img,cmap = "gray")
plt.subplot(122)
plt.imshow(new_img,cmap = "gray")
plt.show()"""

#  get dimension info
height, width, num_channels = img.shape
rotation_amount_rad = 30/180*np.pi

#  create output image, for worst case size (45 degree)
#max_len = int(np.sqrt(height*height + width*width))
#rotated_image = np.zeros((max_len, max_len, num_channels))
rotated_image = np.zeros((img.shape[0],img.shape[1],num_channels))
print(rotated_image.shape)
print(img.shape)
#rotated_image = np.zeros((img.shape))


rotated_height, rotated_width, _ = rotated_image.shape
mid_row = int( (rotated_height+1)/2 )
mid_col = int( (rotated_width+1)/2 )

#  for each pixel in output image, find which pixel
#it corresponds to in the input image
for r in range(rotated_height):
    for c in range(rotated_width):
        #  apply rotation matrix, the other way
        y = (r-mid_col)*np.cos(rotation_amount_rad) + (c-mid_row)*np.sin(rotation_amount_rad)
        x = -(r-mid_col)*np.sin(rotation_amount_rad) + (c-mid_row)*np.cos(rotation_amount_rad)

        #print(y,x)
        #  add offset
        y += mid_col
        x += mid_row

        #  get nearest index
        #a better way is linear interpolation

        if abs(x-round(x)) < 0.4 and abs(y-round(y)) < 0.4:
            x = round(x)
            y = round(y)

        #print(r, " ", c, " corresponds to-> " , y, " ", x)

        #  check if x/y corresponds to a valid pixel in input image
            if (x >= 0 and y >= 0 and x < width and y < height):
                rotated_image[r][c][:] = img[y][x][:]/255
print(np.min(rotated_image), np.max(rotated_image))
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(rotated_image)
plt.close()


print(img.shape)
down_img = tf.rescale(img,1/4, order = 3, channel_axis = 2)
up_img = tf.rescale(img,4,order = 3, channel_axis = 2)
print(down_img.shape)
print(up_img.shape)

plt.subplot(131)
plt.imshow(img, cmap = "magma")
plt.subplot(132)
plt.imshow(up_img)
plt.subplot(133)
plt.imshow(down_img)
plt.show()


plt.plot(img[:,:,0], "*")
plt.show()
