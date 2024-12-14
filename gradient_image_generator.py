import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import colorsys

def extract_main_colors(image_path, num_colors=3):
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img).reshape(-1, 3)
    
    # Convert RGB to LAB color space for better color representation
    from skimage import color
    lab_pixels = color.rgb2lab(pixels[np.newaxis, :, :] / 255.)[0]
    
    kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42)
    kmeans.fit(lab_pixels)
    
    # Convert centers back to RGB
    colors = color.lab2rgb(kmeans.cluster_centers_[np.newaxis, :, :])[0]
    colors = (colors * 255).round().astype(int)
    
    # Sort colors by brightness (sum of RGB values)
    colors = colors[np.argsort([sum(color) for color in colors])][::-1]  # Reverse order
    
    return colors

def create_smooth_radial_gradient(colors, width, height):
    y, x = np.ogrid[:height, :width]
    center = np.array([height / 2, width / 2])
    distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    max_distance = np.sqrt((width/2)**2 + (height/2)**2)
    gradient_positions = distances / max_distance
    
    # Create gradient array
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    
    num_colors = len(colors)
    for i in range(num_colors - 1):
        mask = (gradient_positions >= i / (num_colors - 1)) & (gradient_positions < (i + 1) / (num_colors - 1))
        ratio = (gradient_positions[mask] - i / (num_colors - 1)) * (num_colors - 1)
        for c in range(3):
            gradient[mask, c] = (colors[i][c] * (1 - ratio) + colors[i+1][c] * ratio)
    
    return Image.fromarray(np.uint8(gradient))

def apply_glow_effect(image, intensity=1):
    img_array = np.array(image, dtype=float)
    glow = img_array * intensity
    glow = np.clip(glow, 0, 255).astype(np.uint8)
    return Image.fromarray(glow)

def main(image_path):
    colors = extract_main_colors(image_path)
    with Image.open(image_path) as img:
        width, height = img.size
    
    # Create and save radial gradient
    radial_gradient = create_smooth_radial_gradient(colors, width, height)
    radial_gradient = apply_glow_effect(radial_gradient)
    radial_gradient.save('improved_radial_gradient3.png')

# Usage
main('input3.png')