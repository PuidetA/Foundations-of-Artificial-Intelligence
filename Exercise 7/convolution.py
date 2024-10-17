from PIL import Image  # We will use PIL library for the images, since cv2 at least according to internet sources needs grayscale, and that wouldn't give the same result that the TA used as an example.
import numpy as np
import matplotlib.pyplot as plt  # To display the images and plot them

def convolution(image, kernel):
    img_array = np.array(image) # Convert image to array.
    
    channels = [img_array[:,:,i] for i in range(3)] # Separate into RGB channels.
    results = []
    
    # We will now process each channel
    for channel in channels:     
        
        padding = np.pad(channel, ((1, 1), (1, 1))) # Add padding for edges.
        
        channel_height = channel.shape[0]
        channel_width = channel.shape[1]

        result = np.zeros_like(channel) # Create an array with the same shape as the channel. We filled it with 0s because I wasn't sure if arbitrary values would affect the result.
        # Iterate over each pixel in the channel.
        for i in range(channel_height): # Iterate over all rows.
            for j in range(channel_width): # Iterate over all columns.
                result[i, j] = np.sum(padding[i:i+3, j:j+3] * kernel) # Convolute each 3x3 region.
        
        results.append(result) # Append to the results list.
    

    final_result = np.dstack(results) # Stack all the channels.
    return Image.fromarray(final_result) # Convert  final_result array into an image.


if __name__=="__main__":
    image = Image.open('butterfly.png') # Our original image. For some reason the image isn't taken from the same directory. To take from Exercise 7 folder: 'Exercise 7\\butterfly.png'
    # Our kernels/filters that we will use.
    kernel1 = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]])
    kernel2 = np.array([[-1, 0, -1], [0, 4, 0], [-1, 0, -1]])

    # Apply convolutions.
    kernel1_result = convolution(image, kernel1)
    kernel2_result = convolution(image, kernel2)
    

    # Plot 3 images: Original, kernel1, kernel2.
    fig, (picture1, picture2, picture3) = plt.subplots(1, 3, figsize=(15, 5))
    
    picture1.imshow(image)
    picture1.set_title('Original')
    picture1.axis('off') # Remove axis.
    
    picture2.imshow(kernel1_result)
    picture2.set_title('Kernel/Filter 1')
    picture2.axis('off')
    
    picture3.imshow(kernel2_result)
    picture3.set_title('Kernel/Filter 2')
    picture3.axis('off')

    plt.show()