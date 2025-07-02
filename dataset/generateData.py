import cv2
import os
import random
import numpy as np
from datetime import datetime

def resize_images(images, target_size, axis):
    resized_images = []
    for img in images:
        if axis == 'horizontal':
            resized_img = cv2.resize(img, (img.shape[1], target_size))  # Resize height
        elif axis == 'vertical':
            resized_img = cv2.resize(img, (target_size, img.shape[0]))  # Resize width
        resized_images.append(resized_img)
    return resized_images


def stitch_and_random_crop(images, crop_size=(640, 480), mode='horizontal',crop_image=False, chaos=False):
    if mode == 'horizontal':
        # Resize images to the same height before horizontal concatenation
        heights = [img.shape[0] for img in images]
        min_height = min(heights)
        resized_images = resize_images(images, min_height, axis='horizontal')
        stitched_image = cv2.hconcat(resized_images)

    elif mode == 'vertical':
        # Resize images to the same width before vertical concatenation
        widths = [img.shape[1] for img in images]
        min_width = min(widths)
        resized_images = resize_images(images, min_width, axis='vertical')
        stitched_image = cv2.vconcat(resized_images)


    else:
        raise ValueError("Invalid mode. Choose from 'horizontal', 'vertical', or 'both'.")

    # Get dimensions of the stitched image
    height, width, _ = stitched_image.shape


    if height <= crop_size[1] or width <= crop_size[0]:
        scale_factor_height = crop_size[1] / height
        scale_factor_width = crop_size[0] / width

        # Use the maximum of these to ensure both dimensions are sufficient
        min_scale_factor = max(scale_factor_height, scale_factor_width)

        scale_factor_hgt= random.uniform(scale_factor_height+0.1, 2.0*scale_factor_height)  # Example: scale between 1x and 2x
        scale_factor_wdt = random.uniform(scale_factor_width+0.1, 2.0*scale_factor_width)
        # Calculate new dimensions
        new_height = int(height * scale_factor_hgt)
        new_width = int(width * scale_factor_wdt)

        # Resize the image
        stitched_image = cv2.resize(stitched_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        height, width, _ = stitched_image.shape

    if chaos:
        crop_size =(random.randint(crop_size[0], width),random.randint(crop_size[1], height))
        # Ensure crop_size is within bounds of stitched_image dimensions
    if crop_image:
        #print(f"w{width} - cs0{crop_size[0]}")
        #print(f"h{height} - cs1{crop_size[1]}")
        max_x = width - crop_size[0]
        max_y = height - crop_size[1]

        if max_x < 0 or max_y < 0:
            raise ValueError("Stitched image is smaller than the crop size.")

        # Generate random top-left corner for the crop
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

     # Crop the image to the desired size
        cropped_image = stitched_image[y:y + crop_size[1], x:x + crop_size[0]]
    else:
        cropped_image = cv2.resize(stitched_image, crop_size)

    return cropped_image

def add_random_shadow(image,vertices,shadow_ratio,shadow_intensity):
    # Get image dimensions
    h, w = image.shape[:2]

    

    # Create a shadow mask (initially zeros)
    shadow_mask = np.zeros_like(image)


    hull = cv2.convexHull(vertices)
    #print(hull)
    # Fill the polygon with a dark color on the shadow mask
    cv2.fillPoly(shadow_mask, [hull], (int(255 * shadow_intensity), int(255 * shadow_intensity), int(255 * shadow_intensity)))

    # Blur 
    shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 1)
    # Subtract the shadow mask from the original image
    if shadow_ratio >= 50:
        shadow_image = cv2.subtract(image, shadow_mask)
    else:
        shadow_image = cv2.add(image, shadow_mask)
    

    return shadow_image

def get_perspective_matrix(roll, pitch, yaw, image_shape):
    rows, cols = image_shape[:2]
    
    # Convert degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Rotation matrices for roll, pitch, yaw
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
    # Define the perspective matrix (mapping from original to transformed)
    src_pts = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    dst_pts = np.float32([
        [cols / 2 - R[0, 0] * cols / 2, rows / 2 - R[1, 0] * rows / 2],
        [cols / 2 + R[0, 1] * cols / 2, rows / 2 - R[1, 1] * rows / 2],
        [cols / 2 + R[0, 2] * cols / 2, rows / 2 + R[1, 2] * rows / 2],
        [cols / 2 - R[0, 2] * cols / 2, rows / 2 + R[1, 2] * rows / 2]
    ])

    #print(dst_pts)
    
    # Compute the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    return perspective_matrix

def load_files(path,ext):
    files = os.listdir(path)

    img_files = [(path + "/" + f) for f in files  if f.endswith(ext)]

    return img_files

def generate_random_points(rows, cols):
    """Generate four random points within the image boundaries."""
    pts = []
    for _ in range(4):
        x = random.randint(0, cols - 1)
        y = random.randint(0, rows - 1)
        pts.append([x, y])
    return np.float32(pts)

def save_file(image,path,label_path,count,seed,boundingbox):
    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    file_img_name = str(f"{path}/{time_str}_sd{seed}_image{count}.png")
    file_label_name = str(f"{label_path}/{time_str}_sd{seed}_image{count}.txt")
    #print(file_label_name)
    with open(file_label_name,'w') as file:
        # class x1 y1 x2 y2 x3 y3 x4 y4
        file.write(f"0 {boundingbox[3][0]} {boundingbox[3][1]} {boundingbox[2][0]} {boundingbox[2][1]} {boundingbox[1][0]} {boundingbox[1][1]} {boundingbox[0][0]} {boundingbox[0][1]}\n")
    print(file_img_name)
    cv2.imwrite(file_img_name,image)
    
def save_images(RGB,SEG,DEP,path,count,seed):
    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    file_RGB_name = str(f"{path}/{time_str}_sd{seed}_IMAGE-{count}.jpg")
    file_SEG_name = str(f"{path}/{time_str}_sd{seed}_IMAGE-{count}-seg-flr.png")
    file_DEP_name = str(f"{path}/{time_str}_sd{seed}_IMAGE-{count}-depth.png")
    
    cv2.imwrite(file_RGB_name,RGB)
    cv2.imwrite(file_SEG_name,SEG)
    cv2.imwrite(file_DEP_name,DEP)

def light_burst(image, depth_image, num_reflections, intensity, size, blur_ksize=15, blur_sigma=1, depth_value_fac=25):
    """
    Simulates light bursts (circles or squares) on an image and applies Gaussian blur.
    Also modifies the depth image at the light burst locations.
    
    Parameters:
    image (numpy.ndarray): Input RGB image as a NumPy array with shape (H, W, 3) and values in [0, 255].
    depth_image (numpy.ndarray): Input depth image as a NumPy array with shape (H, W) and values in [0, 255] or [0, 65535].
    num_reflections (int): Number of light bursts to generate.
    intensity (float): Intensity of the light bursts, between 0 and 1.
    size (float): Base size of the light bursts (diameter for circles, side length for squares).
    shape (str): Shape of the light bursts, either 'circle' or 'square'.
    blur_ksize (int): Kernel size for Gaussian blur (must be odd).
    blur_sigma (float): Standard deviation for Gaussian blur.
    depth_value (int): Value to set in the depth image at the light burst locations (e.g., 0 or 65535).
    
    Returns:
    numpy.ndarray: Modified RGB image with light bursts applied and Gaussian blur, in the range [0, 255].
    numpy.ndarray: Modified depth image with depth values set at light burst locations.
    """
    # Create copies of the images to modify
    result_image = np.copy(image).astype(np.float32)
    result_depth = np.copy(depth_image)
    height, width, channels = image.shape
    
    
        
    for _ in range(num_reflections):
        if random.randint(0,100) >= 50:
            shape='square'
        else:
            shape='circle'
            
        # Random center coordinates within the image
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        
        # Random size for this burst between 0.5*size to 2*size
        s = random.uniform(0.5 * size, 2.0 * size)
        
        # Create grid of indices
        y_indices, x_indices = np.indices((height, width))
        
        if shape == 'circle':
            # Calculate distance from the center for a circle
            distance = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
            # Create a circular mask
            mask = (distance <= s / 2).astype(np.float32)
        elif shape == 'square':
            # Create a square mask
            mask = ((np.abs(x_indices - cx) <= s / 2) & (np.abs(y_indices - cy) <= s / 2)).astype(np.float32)
        else:
            raise ValueError("Shape must be 'circle' or 'square'.")
        
        # Scale the mask by the intensity (0-1) and map to [0, 255]
        mask *= intensity * 255
        
        
        # Modify the depth image at the light burst locations
        if random.randint(0,100) >= depth_value_fac:
            result_depth[mask > 0] = (mask[mask>0]*2**16).astype(np.uint16)
        else:
            result_depth[mask > 0] = 0
        
        
    
        # Add the light burst to the RGB image (white light)
        # The mask is applied to all color channels
        light_effect = mask[..., np.newaxis] * np.array([1.0, 1.0, 1.0])[:channels]
        
        
        if blur_ksize > 0:
            light_effect = cv2.GaussianBlur(light_effect, (blur_ksize, blur_ksize), blur_sigma)
         
        print(np.unique(light_effect))
            
        result_image += light_effect.astype(np.uint8)
            
        
    # Clip the RGB image to maintain valid pixel values [0, 255]
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)
    
    # Apply Gaussian blur to the RGB image
    
    
    return result_image, result_depth

def mirror_padding(image, top, bottom, left, right):
    # Mirror top and bottom padding
    top_pad = image[:top, :, :][::-1]
    bottom_pad = image[-bottom:, :, :][::-1]
    
    # Mirror left and right padding
    left_pad = image[:, :left, :][:, ::-1]
    right_pad = image[:, -right:, :][:, ::-1]
    
    # Add padding to the image
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    
    return padded_image

def distort_image(image,rep,k1,k2):
            # Example Camera Matrix (Intrinsic Parameters)
        fx = 303.
        fy = 303.
        cx = 320.
        cy = 240.
        K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

        # Example Distortion Coefficients (k1, k2, p1, p2, k3)
        k3 = 0.0
        p1 = 0.0
        p2 = 0.0
        D = np.array([k1, k2, p1, p2, k3])

        h, w = image.shape[:2]


        # Generate new camera matrix for undistorted image
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))


        # Generate a grid of points corresponding to each pixel in the image
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        points = np.vstack([x.ravel(), y.ravel()]).T

        map_x, map_y = cv2.initUndistortRectifyMap(K, D, None, new_camera_matrix, (w, h), cv2.CV_32FC1)
        if rep:
            distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        else:
            distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        return distorted_image
def random_hsv(image,hue_fact=89/179,sat_fact=127/255,val_fact=63/255,hue_rate=35,sat_rate=35,val_rate=35):
    if random.randint(0, 100) >= hue_rate:
        hue_amount = int(179 * hue_fact)
        refImage_hue = random.randint(-hue_amount, hue_amount)
        image[..., 0] = cv2.add(image[..., 0], refImage_hue)

    if random.randint(0, 100) >=sat_rate:
        sat_amount = int(255 * sat_fact)
        refImage_saturation = random.randint(-sat_amount, sat_amount)
        image[..., 1] = cv2.add(image[..., 1], refImage_saturation)
    if random.randint(0, 100) >= val_rate:
        val_amount = int(255 * val_fact)
        refImage_value = random.randint(-val_amount, val_amount)
        image[..., 2] = cv2.add(image[..., 2], refImage_value)
    return image


if __name__ == '__main__':
    
    # File Setup
    sample_path = "./sampleData" 
    sample_rgb_ext = ".jpg"
    sample_seg_ext = "-segLS.png"
    sample_dep_ext = "-depth.png"

    sample_image_photos = load_files(sample_path,sample_rgb_ext)
    sample_segmentation_photos = load_files(sample_path,sample_seg_ext)
    sample_depth_photos = load_files(sample_path,sample_dep_ext)

    texture_path = "./textures"
    texture_ext = ".jpeg"
    texture_image_files1=load_files(texture_path,texture_ext)
    texture_ext = ".png"
    texture_image_files2=load_files(texture_path,texture_ext)
    texture_ext = ".jpg"
    texture_image_files3=load_files(texture_path,texture_ext)
    texture_image_files = np.concatenate([texture_image_files1,texture_image_files2,texture_image_files3])
    output_path="./training_photos_sim9"
    
    
    # Generation Parameters
    seed = 1
    number_of_images = 125
    save = True
    desired_image_size = (640,480)
    
    hue_fact = 20/179
    sat_fact =20/255
    val_fact =20/255
    ts_min = 0.4
    ts_max = 1.4
    
    t_hue_fact = 5/179
    t_sat_fact =5/255
    t_val_fact =10/255
    T_scale1 = 8
    rot_fact = 60 # Above this value
    tran_fact = 50
    
    T_mosaic_rate = 85
    T_max_mosaic = 3

    print(f"saving {number_of_images} files....")
    count = 0
    for file_i in range(number_of_images):
    
    	# Loading code for sample photo
        sample_random_int = random.randint(1,len(sample_image_photos))
        sample_img_file = sample_image_photos[sample_random_int-1]
        sample_seg_file = sample_img_file.replace('.jpg',sample_seg_ext)
        sample_dep_file = sample_img_file.replace('.jpg',sample_dep_ext)
        sample_image_rgb = cv2.imread(sample_img_file,cv2.IMREAD_UNCHANGED)
        sample_seg = cv2.imread(sample_seg_file,cv2.IMREAD_UNCHANGED)
        sample_dep = cv2.imread(sample_dep_file,cv2.IMREAD_UNCHANGED)
        (h_orig, w_orig) = sample_image_rgb.shape[:2]    

        #ample_image_rgb = cv2.cvtColor(sample_img,cv2.COLOR_BGRA2BGR)
        refImage_image_hsv = cv2.cvtColor(sample_image_rgb,cv2.COLOR_RGB2HSV)


	# Loading code for texture photo
        texture_random_int0 = random.randint(1,len(texture_image_files))
        texture_image_file0 = texture_image_files[texture_random_int0-1]
        texture_image = cv2.resize(cv2.imread(texture_image_file0),desired_image_size)
        if random.randint(0,100) >= T_mosaic_rate:
            random_textures = []
            for i in range(random.randint(1,T_max_mosaic)):
                texture_random_int1 = random.randint(1, len(texture_image_files))
                if random.randint(0,100) > 50:
                    random_textures.append(cv2.cvtColor(cv2.resize(cv2.imread(texture_image_files[texture_random_int1-1]),desired_image_size),cv2.COLOR_BGR2HSV))
                else:
                    random_textures.append(
                        cv2.cvtColor(cv2.imread(texture_image_files[texture_random_int1 - 1]),cv2.COLOR_BGR2HSV))

                if i >=1:
                    flip_type = "horizontal" if random.randint(0,1) == 1 else "vertical"
                    crop_type = True if random.randint(0, 1) == 1 else False
                    chaos = True if random.randint(0, 1) == 1 else False
                    random_textures[i]=stitch_and_random_crop([random_hsv(random_textures[i],t_hue_fact,t_sat_fact,t_val_fact),random_hsv(random_textures[i-1],t_hue_fact,t_sat_fact,t_val_fact)], (640, 480), flip_type,crop_type,chaos)

            texture_image_hsv=cv2.resize(random_textures[-1],(640,480))
        else:
            texture_image_hsv = random_hsv(cv2.cvtColor(texture_image,cv2.COLOR_BGR2HSV),t_hue_fact,t_sat_fact,t_val_fact)
            
        # refImage Processing
        (h, w) = refImage_image_hsv.shape[:2]
        refImage_image_centre=(w//2,h//2)
        # Random HSV applied to saple
        refImage_image_hsv=random_hsv(refImage_image_hsv,hue_fact, sat_fact, val_fact)       
        
        # Shear distortion
        sa = 0.
        sb = 0.0
        if random.randint(0,100) >= 0:
            sk1 = (sa + (sb - sa) * random.random())

        if random.randint(0,100) >= 0:
            sk2 = (sa + (sb - sa) * random.random())   
            
        # Rotation applied
        if random.randint(0,100) > 50:
            refImage_rotation_angle =random.randint(0,360)
            #refImage_rotation_angle = 180
        else:
            refImage_rotation_angle = 0
        
        refImage_scale = max(min(random.random(),1.5),1.0)
        refImage_rotation_matrix= cv2.getRotationMatrix2D(refImage_image_centre,refImage_rotation_angle,refImage_scale)
        print(refImage_rotation_matrix)
	    # Translation applied
        if random.randint(0,100) > tran_fact:
            refImage_tx, refImage_ty = random.randint(-w_orig//T_scale1,w_orig//T_scale1),random.randint(-h_orig//T_scale1,h_orig//T_scale1)
            refImage_rotation_matrix[0,2] +=refImage_tx
            refImage_rotation_matrix[1,2] +=refImage_ty
            apple = True
        else:
            refImage_rotation_matrix[0,2] +=0
            refImage_rotation_matrix[1,2] +=0
            apple = False
        
        # Warp HSV
        if random.randint(0,100) > 75:
            # Define the number of points for the polygon
            num_vertices = np.random.randint(3, 20)  # 3 to 7 vertices

            # Generate random vertices for the polygon
            vertices = np.array([
                [np.random.randint(0, w-1), np.random.randint(0, h-1)]
                for _ in range(num_vertices)
            ], dtype=np.int32)
            
            shadow_ratio = random.randint(0,100)
            if shadow_ratio >= 50:
                shadow_intensity = random.randint(1,3)/10
            else: 
                shadow_intensity = random.randint(1,4)/10
            
            refImage_image_hsv_rgb = cv2.cvtColor(refImage_image_hsv,cv2.COLOR_HSV2RGB)
            refImage_image_hsv = add_random_shadow(refImage_image_hsv_rgb,vertices,shadow_ratio,shadow_intensity)
            refImage_image_hsv = cv2.cvtColor(refImage_image_hsv,cv2.COLOR_RGB2HSV)
            texture_image_hsv_rgb = cv2.cvtColor(texture_image_hsv,cv2.COLOR_HSV2RGB)
            texture_image_hsv = add_random_shadow(texture_image_hsv_rgb,vertices,shadow_ratio,shadow_intensity)
            texture_image_hsv = cv2.cvtColor(texture_image_hsv,cv2.COLOR_RGB2HSV)

            
        


        # Random augmentations to texture
        texture_rotation_angle =random.randint(0,360)
        texture_scale = (ts_min + (ts_max - ts_min) * random.random())
        t_multiplier = 0
        texture_tx, refImage_ty = random.randint(0,w//4),random.randint(0,w//4)
        texture_rotation_matrix= cv2.getRotationMatrix2D(refImage_image_centre,texture_rotation_angle,texture_scale)
        if apple:
            texture_rotation_matrix[0,2] +=refImage_tx
            texture_rotation_matrix[1,2] +=refImage_ty
        texture_image_hsv = cv2.warpAffine(texture_image_hsv,texture_rotation_matrix,(w,h),borderMode=cv2.BORDER_REFLECT_101)
        refImage_image_hsv = cv2.warpAffine(refImage_image_hsv,refImage_rotation_matrix,(w,h),flags=cv2.INTER_NEAREST)
        # Warp Segment
        sample_seg1 = cv2.warpAffine(sample_seg,refImage_rotation_matrix,(w,h),flags=cv2.INTER_NEAREST)
        masked_area = (sample_seg1 > 150).astype(np.uint8)
        opposite_mask = 1 - masked_area
        
        # Warp Depth
        sample_dep1 = cv2.warpAffine(sample_dep,refImage_rotation_matrix,(w,h),flags=cv2.INTER_NEAREST)
        
        print(sample_dep)
        # Calculate the average depth where masked_area == 1
        valid_depths = sample_dep1[masked_area == 1]  # Extract depth values where mask is 1
        if valid_depths.size > 0:  # Check if there are any valid pixels
            average_depth = np.mean(valid_depths)
            average_depth = np.round(average_depth)
        else:
            average_depth = 0  # Or handle the case as needed (e.g., return None or raise an error)

        print(f"Average depth: {average_depth}")
        ######
        sample_seg = cv2.warpAffine(sample_seg,refImage_rotation_matrix,(w,h),flags=cv2.INTER_NEAREST,borderValue=255)
        masked_area = (sample_seg > 150).astype(np.uint8)
        opposite_mask = 1 - masked_area
        
        sample_dep=cv2.warpAffine(sample_dep,refImage_rotation_matrix,(w,h),flags=cv2.INTER_NEAREST,borderValue=average_depth)
        
        masked_area = np.repeat(masked_area[:, :, np.newaxis], 3, axis=2)
        opposite_mask = np.repeat(opposite_mask[:, :, np.newaxis], 3, axis=2)
        
        
        texture_image= cv2.cvtColor(texture_image_hsv,cv2.COLOR_HSV2BGR)
        texture_image = distort_image(texture_image,True,sk1,sk2)
        if random.randint(0,100) >= 80:
            blur_kernel= random.randint(5, 39) | 1
            texture_image = cv2.GaussianBlur(texture_image, (blur_kernel, blur_kernel), 0)
            
        # Apply lightburst
        #_,sample_dep=light_burst(texture_image,sample_dep,5,0.5,10)
        # Covnert HSV 
        refImage_image_rgb = cv2.cvtColor(refImage_image_hsv,cv2.COLOR_HSV2RGB)
        # Overlay Image Section
        # ------------------------------------------------------------------------------------------------------------------
        
        
        # Scale depth = 
        sample_dep = (np.round(sample_dep* 1 /  refImage_scale)).astype(np.uint16)
        print(sample_dep)
        
        output_image_rgb = opposite_mask*refImage_image_rgb + masked_area*texture_image
        
   

        # Add the Gaussian noise to the image
        if random.randint(0,100) > 75:
            # Generate Gaussian noise
            mean = 0
            std_dev = random.randint(0,25) # Adjust standard deviation for intensity of noise
            gaussian_noise = np.random.normal(mean, std_dev, output_image_rgb.shape).astype(np.float32)
            noisy_image = cv2.add(output_image_rgb.astype(np.float32), gaussian_noise)
            output_image_rgb = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
        
        


        refImage_kernel_size = random.randint(3, 5) | 1
        output_image_rgb = cv2.GaussianBlur(output_image_rgb, (refImage_kernel_size, refImage_kernel_size), 0)

        if save:
            save_images(output_image_rgb,masked_area*255,sample_dep,output_path,count,seed)
            print(count)
            count +=1
        else:
            cv2.imshow('sample', output_image_rgb)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
