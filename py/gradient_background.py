import numpy as np
from sklearn.cluster import KMeans
from skimage import color

class RadialGradientNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "num_colors": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 10,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_gradient"
    CATEGORY = "😺dzNodes/LayerUtility"

    def extract_main_colors(self, image: np.ndarray, num_colors: int) -> np.ndarray:
        # Reshape to a 2D array of RGB pixels
        pixels = image.reshape(-1, 3)

        # Convert to LAB color space
        lab_pixels = color.rgb2lab(pixels[np.newaxis, :, :] / 255.)[0]

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42)
        kmeans.fit(lab_pixels)

        # Convert cluster centers back to RGB
        colors = color.lab2rgb(kmeans.cluster_centers_[np.newaxis, :, :])[0]
        colors = (colors * 255).astype(np.uint8)

        # Sort by brightness (sum of RGB values)
        brightness = colors.sum(axis=1)
        sorted_colors = colors[np.argsort(-brightness)]
        return sorted_colors

    def create_smooth_radial_gradient(self, colors: np.ndarray, width: int, height: int) -> np.ndarray:
        # Create coordinate grid
        y, x = np.ogrid[:height, :width]
        center = np.array([height / 2, width / 2])
        distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        max_distance = np.sqrt((width / 2)**2 + (height / 2)**2)
        gradient_positions = distances / max_distance

        # Initialize gradient array
        gradient = np.zeros((height, width, 3), dtype=np.float32)

        # Interpolate between colors
        num_colors = len(colors)
        for i in range(num_colors - 1):
            start = i / (num_colors - 1)
            end = (i + 1) / (num_colors - 1)

            mask = (gradient_positions >= start) & (gradient_positions < end)
            ratio = (gradient_positions[mask] - start) / (end - start)
            for c in range(3):
                gradient[mask, c] = (1 - ratio) * colors[i, c] + ratio * colors[i + 1, c]

        return gradient

    def apply_glow_effect(self, gradient: np.ndarray, intensity: float) -> np.ndarray:
        # Apply glow effect by scaling intensity
        glow = np.clip(gradient * intensity, 0, 255).astype(np.uint8)
        return glow

    def generate_gradient(self, image: np.ndarray, intensity: float, num_colors: int) -> Tuple[np.ndarray]:
        # Convert input image to NumPy array
        img_np = (image[0] * 255).astype(np.uint8).transpose(1, 2, 0)  # Convert (C, H, W) -> (H, W, C)

        # Extract colors
        colors = self.extract_main_colors(img_np, num_colors)

        # Get image dimensions
        height, width = img_np.shape[:2]

        # Create radial gradient
        gradient = self.create_smooth_radial_gradient(colors, width, height)

        # Apply glow effect
        gradient = self.apply_glow_effect(gradient, intensity)

        return (gradient,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: RadialGradientNode": RadialGradientNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: RadialGradientNode": "LayerUtility: Radial Gradient"
}
