#!/usr/bin/env python3
"""
nimslo auto-aligning gif processor

standalone python script for processing nimslo film shots into aligned gifs
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RectangleSelector
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# for gif creation
from PIL import Image as PILImage
import imageio

# for deep learning alignment
from sklearn.feature_extraction import image
from scipy import ndimage
from scipy.spatial.distance import cdist

# for cnn-based alignment
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow.keras.backend as K
    TENSORFLOW_AVAILABLE = True
    print("✅ tensorflow available for cnn alignment")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  tensorflow not available - cnn alignment will use fallback methods")

print("🚀 nimslo processor starting up...")

class NimsloProcessor:
    """
    main class for processing nimslo images into aligned gifs
    handles image loading, alignment, histogram matching, and gif export
    """
    
    def __init__(self):
        self.images = []
        self.image_paths = []
        self.reference_points = []
        self.aligned_images = []
        self.matched_images = []
        self.crop_box = None
        self.all_image_files = []  # store all found images
        
    def load_images(self, folder_path=None):
        """load images from folder or file dialog"""
        if folder_path is None:
            root = tk.Tk()
            root.withdraw()
            folder_path = filedialog.askdirectory(title="select nimslo batch folder")
            root.destroy()
        
        if not folder_path:
            print("❌ no folder selected")
            return False
            
        # look for common image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        image_files.sort()  # ensure consistent ordering
        
        if len(image_files) < 4:
            print(f"❌ need at least 4 images, found {len(image_files)}")
            return False
            
        print(f"📁 found {len(image_files)} images")
        self.all_image_files = image_files
        
        # let user select which images to use
        return self.select_images_manually(image_files)
    
    def select_images_manually(self, image_files):
        """manual image selection interface"""
        print("🎯 opening image selector...")
        
        # create selection window
        root = tk.Tk()
        root.title("select nimslo images")
        root.geometry("1200x800")
        
        # create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # left side - available images
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left_frame, text="available images:").pack()
        
        available_listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED)
        available_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=available_listbox.yview)
        available_listbox.configure(yscrollcommand=available_scrollbar.set)
        
        available_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        available_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # center - preview area
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        ttk.Label(center_frame, text="preview:").pack()
        
        # create a frame for the canvas that can resize
        canvas_frame = ttk.Frame(center_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        preview_canvas = tk.Canvas(canvas_frame, bg='white', width=300, height=200)
        preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # make canvas responsive to window resize
        def resize_canvas(event=None):
            """resize canvas to fit the frame"""
            canvas_frame.update_idletasks()
            width = canvas_frame.winfo_width() - 10
            height = canvas_frame.winfo_height() - 10
            if width > 10 and height > 10:
                preview_canvas.configure(width=width, height=height)
        
        canvas_frame.bind('<Configure>', resize_canvas)
        
        # right side - selected images
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_frame, text="selected images (order matters):").pack()
        
        selected_listbox = tk.Listbox(right_frame)
        selected_scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=selected_listbox.yview)
        selected_listbox.configure(yscrollcommand=selected_scrollbar.set)
        
        selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        selected_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # store image thumbnails
        thumbnails = {}
        preview_photo = None
        
        def create_thumbnail(image_path, size=None):
            """create a thumbnail for preview"""
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # get canvas size for better sizing
                    if size is None:
                        canvas_width = preview_canvas.winfo_width()
                        canvas_height = preview_canvas.winfo_height()
                        if canvas_width > 10 and canvas_height > 10:
                            # leave space for text and padding
                            size = (canvas_width - 20, canvas_height - 60)
                        else:
                            size = (200, 150)  # default size
                    
                    # resize maintaining aspect ratio
                    h, w = img_rgb.shape[:2]
                    scale = min(size[0]/w, size[1]/h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    resized = cv2.resize(img_rgb, (new_w, new_h))
                    
                    # convert to PIL for tkinter
                    pil_img = PILImage.fromarray(resized)
                    return ImageTk.PhotoImage(pil_img)
            except Exception as e:
                print(f"❌ failed to create thumbnail for {image_path}: {e}")
            return None
        
        def show_preview(event=None):
            """show preview of selected image"""
            nonlocal preview_photo
            
            # clear previous preview
            preview_canvas.delete("all")
            
            # get selected item from available list
            selection = available_listbox.curselection()
            if selection:
                idx = selection[0]
                if idx < len(image_files):
                    image_path = image_files[idx]
                    filename = os.path.basename(image_path)
                    
                    # get canvas dimensions
                    canvas_width = preview_canvas.winfo_width()
                    canvas_height = preview_canvas.winfo_height()
                    
                    if canvas_width > 10 and canvas_height > 10:
                        # show filename at top
                        preview_canvas.create_text(canvas_width//2, 15, text=filename, 
                                                anchor=tk.CENTER, font=('Arial', 10))
                        
                        # show thumbnail centered
                        if image_path not in thumbnails:
                            thumbnails[image_path] = create_thumbnail(image_path)
                        
                        if thumbnails[image_path]:
                            preview_photo = thumbnails[image_path]
                            # center the image
                            img_x = canvas_width // 2
                            img_y = canvas_height // 2 + 10
                            preview_canvas.create_image(img_x, img_y, image=preview_photo, anchor=tk.CENTER)
                        else:
                            preview_canvas.create_text(canvas_width//2, canvas_height//2, 
                                                    text="preview not available", anchor=tk.CENTER)
        
        def show_selected_preview(event=None):
            """show preview of selected image from selected list"""
            nonlocal preview_photo
            
            # clear previous preview
            preview_canvas.delete("all")
            
            # get selected item from selected list
            selection = selected_listbox.curselection()
            if selection:
                idx = selection[0]
                filename = selected_listbox.get(idx)
                
                # find the full path
                for file_path in image_files:
                    if os.path.basename(file_path) == filename:
                        image_path = file_path
                        break
                else:
                    return
                
                # get canvas dimensions
                canvas_width = preview_canvas.winfo_width()
                canvas_height = preview_canvas.winfo_height()
                
                if canvas_width > 10 and canvas_height > 10:
                    # show filename at top
                    preview_canvas.create_text(canvas_width//2, 15, text=f"selected: {filename}", 
                                            anchor=tk.CENTER, font=('Arial', 10))
                    
                    # show thumbnail centered
                    if image_path not in thumbnails:
                        thumbnails[image_path] = create_thumbnail(image_path)
                    
                    if thumbnails[image_path]:
                        preview_photo = thumbnails[image_path]
                        # center the image
                        img_x = canvas_width // 2
                        img_y = canvas_height // 2 + 10
                        preview_canvas.create_image(img_x, img_y, image=preview_photo, anchor=tk.CENTER)
                    else:
                        preview_canvas.create_text(canvas_width//2, canvas_height//2, 
                                                text="preview not available", anchor=tk.CENTER)
        
        # populate available list
        for i, file_path in enumerate(image_files):
            filename = os.path.basename(file_path)
            available_listbox.insert(tk.END, f"{i+1:2d}. {filename}")
        
        # bind preview events
        available_listbox.bind('<<ListboxSelect>>', show_preview)
        selected_listbox.bind('<<ListboxSelect>>', show_selected_preview)
        
        # buttons
        button_frame = ttk.Frame(root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def add_selected():
            """add selected images to the selected list"""
            selections = available_listbox.curselection()
            for idx in selections:
                if idx < len(image_files):
                    filename = os.path.basename(image_files[idx])
                    # check if already in selected list
                    if filename not in selected_listbox.get(0, tk.END):
                        selected_listbox.insert(tk.END, f"{filename}")
        
        def remove_selected():
            """remove selected images from the selected list"""
            selections = selected_listbox.curselection()
            for idx in reversed(selections):
                selected_listbox.delete(idx)
        
        def move_up():
            """move selected item up in the list"""
            selections = selected_listbox.curselection()
            if selections and selections[0] > 0:
                idx = selections[0]
                text = selected_listbox.get(idx)
                selected_listbox.delete(idx)
                selected_listbox.insert(idx-1, text)
                selected_listbox.selection_set(idx-1)
        
        def move_down():
            """move selected item down in the list"""
            selections = selected_listbox.curselection()
            if selections and selections[0] < selected_listbox.size() - 1:
                idx = selections[0]
                text = selected_listbox.get(idx)
                selected_listbox.delete(idx)
                selected_listbox.insert(idx+1, text)
                selected_listbox.selection_set(idx+1)
        
        def load_selected():
            """load the selected images in order"""
            selected_files = []
            for i in range(selected_listbox.size()):
                filename = selected_listbox.get(i)
                # find the full path
                for file_path in image_files:
                    if os.path.basename(file_path) == filename:
                        selected_files.append(file_path)
                        break
            
            if len(selected_files) < 4:
                messagebox.showerror("error", "need at least 4 images selected")
                return
            
            self.image_paths = selected_files
            self.images = []
            
            for i, path in enumerate(selected_files):
                img = cv2.imread(path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.images.append(img_rgb)
                    print(f"✅ loaded image {i+1}: {os.path.basename(path)} ({img_rgb.shape})")
                else:
                    print(f"❌ failed to load: {path}")
            
            root.destroy()
        
        def cancel():
            """cancel selection"""
            root.destroy()
        
        # add buttons
        ttk.Button(button_frame, text="add →", command=add_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="remove", command=remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="↑", command=move_up).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="↓", command=move_down).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="load selected", command=load_selected).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="cancel", command=cancel).pack(side=tk.RIGHT, padx=5)
        
        # instructions
        instruction_frame = ttk.Frame(root)
        instruction_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(instruction_frame, text="select 4-6 images in the order you want them in the gif. click on images to see preview.").pack()
        
        # show initial preview if available
        if image_files:
            available_listbox.selection_set(0)
            show_preview()
        
        root.mainloop()
        
        return len(self.images) >= 4
    
    def show_images(self, images=None, title="nimslo batch", save_path=None):
        """display loaded images in a grid"""
        if images is None:
            images = self.images
            
        if not images:
            print("❌ no images to display")
            return
            
        n_images = len(images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        fig.suptitle(title, fontsize=16)
        
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
            
        for i, img in enumerate(images):
            axes[i].imshow(img)
            axes[i].set_title(f"frame {i+1}")
            axes[i].axis('off')
            
        # hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 saved preview to: {save_path}")
        
        # always show the plot
        plt.show()
        
        # add a small delay to ensure window appears
        import time
        time.sleep(0.5)

def preview_alignment(self, title="alignment preview"):
    """show alignment results with user confirmation"""
    if not self.aligned_images:
        print("❌ no aligned images to preview")
        return False
        
    print(f"\n🎯 {title}:")
    print("   - check the preview window that should appear")
    print("   - images should be aligned to the reference")
    print("   - close the preview window when done viewing")
    
    self.show_images(self.aligned_images, title, save_path="preview_aligned.png")
    
    return True

def preview_final(self, title="final preview"):
    """show final processed images with user confirmation"""
    if not self.matched_images:
        print("❌ no processed images to preview")
        return False
        
    print(f"\n🎬 {title}:")
    print("   - check the preview window that should appear")
    print("   - images should have consistent exposure")
    print("   - close the preview window when done viewing")
    
    self.show_images(self.matched_images, title, save_path="preview_final.png")
    
    return True

class InteractiveCropper:
    """interactive gui for selecting crop area and reference points"""
    
    def __init__(self, image):
        self.image = image
        self.crop_coords = None
        self.reference_points = []
        self.fig = None
        self.ax = None
        self.selector = None
        
    def on_crop_select(self, eclick, erelease):
        """callback for crop selection"""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # ensure proper ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # validate crop area
        width = x2 - x1
        height = y2 - y1
        
        if width < 50 or height < 50:
            print(f"⚠️  crop area too small: {width}x{height} pixels")
            print("   please select a larger area (minimum 50x50 pixels)")
            return
        
        self.crop_coords = (x1, y1, x2, y2)
        print(f"📐 crop selected: ({x1}, {y1}) to ({x2}, {y2}) = {width}x{height} pixels")
        
    def on_click(self, event):
        """callback for reference point selection"""
        if event.inaxes != self.ax:
            return
            
        if event.button == 1 and event.dblclick:  # double click
            x, y = int(event.xdata), int(event.ydata)
            self.reference_points.append((x, y))
            
            # plot the point
            self.ax.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
            self.fig.canvas.draw()
            
            print(f"📍 reference point {len(self.reference_points)}: ({x}, {y})")
            
    def select_crop_and_reference(self):
        """interactive selection of crop area and reference point"""
        print("🎯 crop selection mode:")
        print("   - drag to select crop area")
        print("   - double-click to add reference points")
        print("   - close window when done")
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title("select crop area (drag) and reference points (double-click)")
        
        # rectangle selector for cropping
        self.selector = RectangleSelector(
            self.ax, self.on_crop_select,
            useblit=True, button=[1], minspanx=50, minspany=50,
            spancoords='pixels', interactive=True
        )
        
        # click handler for reference points
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.show()
        
        return self.crop_coords, self.reference_points

class ImageAligner:
    """handles the alignment of images using computer vision techniques"""
    
    def __init__(self):
        # initialize feature detectors (try SIFT first, fall back to ORB)
        try:
            self.sift = cv2.SIFT_create()
            self.feature_detector = 'sift'
            print("🔍 using SIFT feature detector")
        except:
            self.sift = cv2.ORB_create(nfeatures=1000)
            self.feature_detector = 'orb'
            print("🔍 using ORB feature detector (SIFT not available)")
            
        # matcher
        if self.feature_detector == 'sift':
            self.matcher = cv2.FlannBasedMatcher()
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def detect_and_match_features(self, img1, img2, ratio_threshold=0.7):
        """detect features and find matches between two images"""
        # convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # detect keypoints and descriptors
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            print("❌ no features detected")
            return [], [], []
            
        # match features
        if self.feature_detector == 'sift':
            matches = self.matcher.knnMatch(des1, des2, k=2)
            # apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
        else:
            matches = self.matcher.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:50]
        
        return kp1, kp2, good_matches
    
    def estimate_transform(self, kp1, kp2, matches, transform_type='homography'):
        """estimate transformation matrix from matched keypoints"""
        if len(matches) < 4:
            print(f"❌ not enough matches ({len(matches)}) for transformation")
            return None
            
        # extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        if transform_type == 'homography':
            matrix, mask = cv2.findHomography(src_pts, dst_pts, 
                                            cv2.RANSAC, 5.0)
        elif transform_type == 'affine':
            matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        else:
            print(f"❌ unsupported transform type: {transform_type}")
            return None
            
        if matrix is None:
            print("❌ failed to estimate transformation")
            return None
            
        # count inliers
        inliers = np.sum(mask) if mask is not None else len(matches)
        print(f"✅ transformation estimated with {inliers}/{len(matches)} inliers")
        
        return matrix
    
    def align_to_reference(self, images, reference_index=0, transform_type='homography'):
        """align all images to a reference image"""
        if not images or len(images) < 2:
            print("❌ need at least 2 images for alignment")
            return []
            
        reference_img = images[reference_index]
        aligned_images = [reference_img.copy()]  # reference stays unchanged
        transforms = [np.eye(3)]  # identity for reference
        
        print(f"🎯 aligning {len(images)} images to reference (image {reference_index})")
        print(f"🔧 using {transform_type} transformation")
        print(f"📐 reference image size: {reference_img.shape}")
        
        for i, img in enumerate(images):
            if i == reference_index:
                continue
                
            print(f"\n🔄 aligning image {i}...")
            print(f"📐 image {i} size: {img.shape}")
            
            # detect and match features
            kp_ref, kp_img, matches = self.detect_and_match_features(reference_img, img)
            
            print(f"🔍 found {len(kp_ref)} features in reference")
            print(f"🔍 found {len(kp_img)} features in image {i}")
            print(f"🔗 matched {len(matches)} features")
            
            if len(matches) < 10:
                print(f"⚠️  warning: only {len(matches)} matches found for image {i}")
                print("💡 try: different reference image, tighter crop, or different transform type")
            
            # estimate transformation
            transform_matrix = self.estimate_transform(kp_ref, kp_img, matches, transform_type)
            
            if transform_matrix is not None:
                print(f"✅ transformation matrix shape: {transform_matrix.shape}")
                
                # apply transformation
                h, w = reference_img.shape[:2]
                if transform_type == 'homography':
                    aligned_img = cv2.warpPerspective(img, transform_matrix, (w, h))
                else:
                    aligned_img = cv2.warpAffine(img, transform_matrix, (w, h))
                    
                aligned_images.insert(i, aligned_img)
                transforms.insert(i, transform_matrix)
                print(f"✅ image {i} aligned successfully")
            else:
                print(f"❌ failed to align image {i}, using original")
                aligned_images.insert(i, img.copy())
                transforms.insert(i, np.eye(3))
        
        return aligned_images, transforms

class HistogramMatcher:
    """handles histogram matching for consistent exposure across images"""
    
    def __init__(self, reference_index=0):
        self.reference_index = reference_index
    
    def match_histogram(self, source, reference, multichannel=True):
        """match histogram of source image to reference image"""
        if multichannel:
            # process each channel separately for color images
            matched = np.zeros_like(source)
            for channel in range(source.shape[2]):
                matched[:, :, channel] = self._match_histogram_single_channel(
                    source[:, :, channel], reference[:, :, channel]
                )
            return matched
        else:
            return self._match_histogram_single_channel(source, reference)
    
    def _match_histogram_single_channel(self, source, reference):
        """match histogram for a single channel"""
        # get histograms
        source_hist, source_bins = np.histogram(source.flatten(), 256, density=True)
        ref_hist, ref_bins = np.histogram(reference.flatten(), 256, density=True)
        
        # calculate cumulative distribution functions
        source_cdf = source_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        
        # normalize cdfs
        source_cdf = source_cdf / source_cdf[-1]
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        # create lookup table
        lookup_table = np.interp(source_cdf, ref_cdf, np.arange(256))
        
        # apply lookup table
        matched = np.interp(source.flatten(), np.arange(256), lookup_table)
        
        return matched.reshape(source.shape).astype(source.dtype)
    
    def adaptive_histogram_match(self, source, reference, strength=0.7):
        """gentler histogram matching that preserves some original character"""
        matched = self.match_histogram(source, reference)
        
        # blend original with matched version
        result = (1 - strength) * source + strength * matched
        
        return result.astype(source.dtype)
    
    def match_exposure_stats(self, images, reference_index=0):
        """match basic exposure statistics (mean, std) across images"""
        if not images:
            return []
            
        reference = images[reference_index].astype(np.float32)
        ref_mean = np.mean(reference)
        ref_std = np.std(reference)
        
        matched_images = []
        
        for i, img in enumerate(images):
            if i == reference_index:
                matched_images.append(img)
                continue
                
            img_float = img.astype(np.float32)
            img_mean = np.mean(img_float)
            img_std = np.std(img_float)
            
            # normalize and rescale
            normalized = (img_float - img_mean) / (img_std + 1e-8)
            rescaled = normalized * ref_std + ref_mean
            
            # clip to valid range
            rescaled = np.clip(rescaled, 0, 255).astype(np.uint8)
            matched_images.append(rescaled)
            
        return matched_images

class CNNBorderAligner:
    """cnn-based alignment using border detection and image subtraction"""
    
    def __init__(self):
        self.model = None
        self.border_model = None
        self.initialized = False
        
    def create_border_detection_model(self):
        """create a lightweight cnn for border detection"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️  TensorFlow not available, skipping cnn border detection model creation.")
            return False
            
        try:
            # simple u-net style architecture for border detection
            inputs = keras.Input(shape=(None, None, 3))
            
            # encoder
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D(2)(x)
            
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D(2)(x)
            
            # decoder
            x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            
            x = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            
            # output border mask
            outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
            
            self.border_model = keras.Model(inputs, outputs)
            print("✅ cnn border detection model created")
            return True
            
        except Exception as e:
            print(f"❌ failed to create cnn model: {e}")
            return False
    
    def detect_borders(self, image, threshold=0.5):
        """detect borders using cnn"""
        if self.border_model is None:
            return self.detect_borders_simple(image)
        
        if not TENSORFLOW_AVAILABLE:
            print("⚠️  TensorFlow not available, falling back to simple border detection.")
            return self.detect_borders_simple(image)
        
        try:
            # preprocess image
            img_normalized = image.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # predict borders
            border_mask = self.border_model.predict(img_batch, verbose=0)[0, :, :, 0]
            
            # threshold to get binary border mask
            borders = (border_mask > threshold).astype(np.uint8) * 255
            
            return borders
            
        except Exception as e:
            print(f"❌ cnn border detection failed: {e}")
            return self.detect_borders_simple(image)
    
    def detect_borders_simple(self, image):
        """fallback simple border detection using canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        borders = cv2.dilate(edges, kernel, iterations=1)
        
        return borders
    
    def calculate_image_difference(self, img1, img2):
        """calculate difference between two images (like photoshop difference layer)"""
        # convert to float for better precision
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        # calculate absolute difference
        diff = np.abs(img1_float - img2_float)
        
        # normalize to 0-255 range
        diff_normalized = np.clip(diff, 0, 255).astype(np.uint8)
        
        return diff_normalized
    
    def find_optimal_alignment(self, reference_img, target_img, max_shift=50):
        """find optimal alignment using border detection and image subtraction"""
        print("🔍 finding optimal alignment using cnn borders...")
        
        # detect borders
        ref_borders = self.detect_borders(reference_img)
        target_borders = self.detect_borders(target_img)
        
        # calculate border difference
        border_diff = self.calculate_image_difference(ref_borders, target_borders)
        
        # try different shifts and find minimum difference
        best_shift = (0, 0)
        min_diff = np.sum(border_diff)
        
        h, w = reference_img.shape[:2]
        
        for dx in range(-max_shift, max_shift + 1, 2):
            for dy in range(-max_shift, max_shift + 1, 2):
                # create transformation matrix
                transform_matrix = np.array([
                    [1, 0, dx],
                    [0, 1, dy]
                ], dtype=np.float32)
                
                # apply shift to target
                shifted_target = cv2.warpAffine(target_img, transform_matrix, (w, h))
                shifted_borders = cv2.warpAffine(target_borders, transform_matrix, (w, h))
                
                # calculate difference
                diff = self.calculate_image_difference(ref_borders, shifted_borders)
                total_diff = np.sum(diff)
                
                if total_diff < min_diff:
                    min_diff = total_diff
                    best_shift = (dx, dy)
        
        print(f"✅ optimal shift found: {best_shift} (difference: {min_diff})")
        
        # create final transformation matrix
        final_transform = np.array([
            [1, 0, best_shift[0]],
            [0, 1, best_shift[1]]
        ], dtype=np.float32)
        
        return final_transform
    
    def align_to_reference_cnn(self, images, reference_index=0):
        """align all images using cnn border detection"""
        if not images or len(images) < 2:
            print("❌ need at least 2 images for alignment")
            return []
        
        # initialize cnn if needed
        if not self.initialized:
            print("🧠 initializing cnn border detection...")
            self.create_border_detection_model()
            self.initialized = True
        
        reference_img = images[reference_index]
        aligned_images = [reference_img.copy()]
        transforms = [np.eye(3)]
        
        print(f"🎯 cnn-aligning {len(images)} images to reference (image {reference_index})")
        
        for i, img in enumerate(images):
            if i == reference_index:
                continue
            
            print(f"\n🔄 cnn-aligning image {i}...")
            
            # find optimal alignment
            transform_matrix = self.find_optimal_alignment(reference_img, img)
            
            if transform_matrix is not None:
                # apply transformation
                h, w = reference_img.shape[:2]
                aligned_img = cv2.warpAffine(img, transform_matrix, (w, h))
                
                aligned_images.insert(i, aligned_img)
                transforms.insert(i, transform_matrix)
                print(f"✅ image {i} cnn-aligned successfully")
            else:
                print(f"❌ failed to cnn-align image {i}, using original")
                aligned_images.insert(i, img.copy())
                transforms.insert(i, np.eye(3))
        
        return aligned_images, transforms

class GifExporter:
    """handles creation and export of animated gifs"""
    
    def __init__(self):
        self.output_folder = "nimslo_gifs"
        os.makedirs(self.output_folder, exist_ok=True)
    
    def create_gif(self, images, output_path, duration=0.2, loop=0, optimize=True, bounce=False, quality='high'):
        """create animated gif from list of images with quality options"""
        if not images:
            print("❌ no images to create gif from")
            return False
            
        print(f"🎬 creating {quality} quality gif with {len(images)} frames...")
        
        # convert numpy arrays to PIL images
        pil_images = []
        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                # ensure uint8 format
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                pil_img = PILImage.fromarray(img)
            else:
                pil_img = img
            pil_images.append(pil_img)
            print(f"✅ frame {i+1} converted")
        
        # create bounce effect if requested
        if bounce and len(pil_images) > 1:
            print("🔄 creating bounce effect...")
            # forward sequence: 0, 1, 2, 3, ...
            forward_frames = pil_images
            # backward sequence: n-2, n-3, ..., 1 (skip first and last to avoid duplicates)
            backward_frames = pil_images[-2:0:-1] if len(pil_images) > 2 else []
            
            # combine: forward + backward
            bounce_frames = forward_frames + backward_frames
            print(f"📊 bounce sequence: {len(forward_frames)} forward + {len(backward_frames)} backward = {len(bounce_frames)} total frames")
            
            pil_images = bounce_frames
        
        # quality-specific settings
        save_kwargs = {
            'save_all': True,
            'append_images': pil_images[1:],
            'duration': int(duration * 1000),  # convert to milliseconds
            'loop': loop,
            'optimize': optimize
        }
        
        if quality == 'high':
            # high quality: no additional compression
            pass
        elif quality == 'medium':
            # medium quality: some optimization
            save_kwargs['optimize'] = True
        elif quality == 'optimized':
            # optimized: maximum compression
            save_kwargs['optimize'] = True
            # reduce color palette for smaller file size
            for i, img in enumerate(pil_images):
                pil_images[i] = img.quantize(colors=256, method=PILImage.ADAPTIVE)
            save_kwargs['append_images'] = pil_images[1:]
        
        # save as gif
        try:
            pil_images[0].save(output_path, **save_kwargs)
            
            # show quality info
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"🎉 {quality} quality gif saved to: {output_path}")
            print(f"📁 file size: {file_size:.2f} MB")
            
            return True
        except Exception as e:
            print(f"❌ failed to save gif: {e}")
            return False

def add_methods_to_processor(processor):
    """add all the processing methods to the processor class"""
    
    def select_crop_and_reference(self, image_index=0):
        """select crop area and reference points on specified image"""
        if not self.images:
            print("❌ no images loaded")
            return False
            
        if image_index >= len(self.images):
            print(f"❌ image index {image_index} out of range")
            return False
            
        cropper = InteractiveCropper(self.images[image_index])
        crop_coords, ref_points = cropper.select_crop_and_reference()
        
        if crop_coords:
            self.crop_box = crop_coords
            print(f"✅ crop area saved: {crop_coords}")
        
        if ref_points:
            self.reference_points = ref_points
            print(f"✅ {len(ref_points)} reference points saved")
            
        return crop_coords is not None or ref_points
    
    def align_images(self, reference_index=0, transform_type='homography'):
        """align all loaded images"""
        if not self.images:
            print("❌ no images loaded")
            return False
            
        aligner = ImageAligner()
        
        # apply crop if selected
        images_to_align = self.images
        if self.crop_box:
            x1, y1, x2, y2 = self.crop_box
            
            # validate crop area
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                print(f"❌ invalid crop area: {width}x{height} pixels")
                print("💡 please reselect crop area with proper rectangle")
                return False
            
            if width < 50 or height < 50:
                print(f"⚠️  crop area very small: {width}x{height} pixels")
                print("💡 this might cause alignment issues")
            
            try:
                images_to_align = [img[y1:y2, x1:x2] for img in self.images]
                print(f"🔄 applying crop ({x1},{y1}) to ({x2},{y2}) = {width}x{height} pixels")
                
                # check that cropped images are valid
                for i, img in enumerate(images_to_align):
                    if img.size == 0:
                        print(f"❌ cropped image {i} is empty")
                        return False
                    print(f"✅ cropped image {i}: {img.shape}")
                    
            except Exception as e:
                print(f"❌ error applying crop: {e}")
                return False
        
        aligned_imgs, transforms = aligner.align_to_reference(
            images_to_align, reference_index, transform_type
        )
        
        if aligned_imgs:
            self.aligned_images = aligned_imgs
            self.transforms = transforms
            print(f"✅ aligned {len(aligned_imgs)} images successfully!")
            return True
        
        return False
    
    def align_images_cnn(self, reference_index=0):
        """align all loaded images using cnn border detection"""
        if not self.images:
            print("❌ no images loaded")
            return False
            
        cnn_aligner = CNNBorderAligner()
        
        # apply crop if selected
        images_to_align = self.images
        if self.crop_box:
            x1, y1, x2, y2 = self.crop_box
            
            # validate crop area
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                print(f"❌ invalid crop area: {width}x{height} pixels")
                print("💡 please reselect crop area with proper rectangle")
                return False
            
            if width < 50 or height < 50:
                print(f"⚠️  crop area very small: {width}x{height} pixels")
                print("💡 this might cause alignment issues")
            
            try:
                images_to_align = [img[y1:y2, x1:x2] for img in self.images]
                print(f"🔄 applying crop ({x1},{y1}) to ({x2},{y2}) = {width}x{height} pixels")
                
                # check that cropped images are valid
                for i, img in enumerate(images_to_align):
                    if img.size == 0:
                        print(f"❌ cropped image {i} is empty")
                        return False
                    print(f"✅ cropped image {i}: {img.shape}")
                    
            except Exception as e:
                print(f"❌ error applying crop: {e}")
                return False
        
        aligned_imgs, transforms = cnn_aligner.align_to_reference_cnn(
            images_to_align, reference_index
        )
        
        if aligned_imgs:
            self.aligned_images = aligned_imgs
            self.transforms = transforms
            print(f"✅ cnn-aligned {len(aligned_imgs)} images successfully!")
            return True
        
        return False
    
    def match_histograms(self, reference_index=0, method='adaptive', strength=0.7):
        """match histograms of all images"""
        if not self.aligned_images:
            print("❌ no aligned images available - run alignment first")
            return False
            
        matcher = HistogramMatcher(reference_index)
        
        print(f"🌈 matching histograms using {method} method...")
        
        if method == 'adaptive':
            matched_images = []
            reference = self.aligned_images[reference_index]
            
            for i, img in enumerate(self.aligned_images):
                if i == reference_index:
                    matched_images.append(img)
                    print(f"📊 image {i}: reference (unchanged)")
                else:
                    matched = matcher.adaptive_histogram_match(img, reference, strength)
                    matched_images.append(matched)
                    print(f"📊 image {i}: histogram matched (strength={strength})")
                    
        elif method == 'full':
            matched_images = []
            reference = self.aligned_images[reference_index]
            
            for i, img in enumerate(self.aligned_images):
                if i == reference_index:
                    matched_images.append(img)
                else:
                    matched = matcher.match_histogram(img, reference)
                    matched_images.append(matched)
                    print(f"📊 image {i}: full histogram matched")
                    
        elif method == 'exposure':
            matched_images = matcher.match_exposure_stats(self.aligned_images, reference_index)
            print("📊 exposure statistics matched")
            
        else:
            print(f"❌ unknown method: {method}")
            return False
        
        self.matched_images = matched_images
        print(f"✅ histogram matching complete!")
        return True
    
    def create_nimslo_gif(self, output_filename=None, duration=0.15, bounce=True, quality='high'):
        """create the final nimslo gif"""
        if not self.matched_images:
            print("❌ no processed images available - run full pipeline first")
            return False
            
        exporter = GifExporter()
        
        if output_filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"nimslo_{quality}_{timestamp}.gif"
        
        output_path = os.path.join(exporter.output_folder, output_filename)
        
        success = exporter.create_gif(self.matched_images, output_path, duration, bounce=bounce, quality=quality)
        
        if success:
            # show file info (file size is now shown in create_gif)
            print(f"🎯 frames: {len(self.matched_images)}")
            if bounce and len(self.matched_images) > 1:
                bounce_frames = len(self.matched_images) + (len(self.matched_images) - 2)
                print(f"🔄 bounce frames: {bounce_frames} (forward + backward)")
            print(f"⏱️  duration per frame: {duration}s")
            print(f"🏆 quality level: {quality}")
        
        return success
    
    def create_comparison_gif(self, output_filename=None, duration=0.3, bounce=True):
        """create a before/after comparison gif"""
        if not self.images or not self.matched_images:
            print("❌ need both original and processed images")
            return False
            
        exporter = GifExporter()
        
        if output_filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"nimslo_comparison_{timestamp}.gif"
        
        output_path = os.path.join(exporter.output_folder, output_filename)
        
        # use cropped originals if crop was applied
        original_images = self.images
        if self.crop_box:
            x1, y1, x2, y2 = self.crop_box
            original_images = [img[y1:y2, x1:x2] for img in self.images]
        
        return exporter.create_gif(original_images + self.matched_images, output_path, duration, bounce=bounce)
    
    def set_output_folder(self, folder_path):
        """set custom output folder for gifs"""
        os.makedirs(folder_path, exist_ok=True)
        GifExporter().output_folder = folder_path
        print(f"📁 output folder set to: {folder_path}")
    
    def crop_to_valid_area(self, images):
        """crop images to remove black bars and ensure consistent area"""
        if not images:
            return images
            
        print("🔄 cropping to valid area to remove black bars...")
        
        # find the common valid area across all images
        h, w = images[0].shape[:2]
        min_x, min_y = 0, 0
        max_x, max_y = w, h
        
        for img in images:
            # find non-black pixels
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            non_zero = cv2.findNonZero(gray)
            
            if non_zero is not None:
                # get bounding box of non-zero pixels
                x, y, w_box, h_box = cv2.boundingRect(non_zero)
                
                # update bounds
                min_x = max(min_x, x)
                min_y = max(min_y, y)
                max_x = min(max_x, x + w_box)
                max_y = min(max_y, y + h_box)
        
        # ensure minimum size
        crop_w = max_x - min_x
        crop_h = max_y - min_y
        
        if crop_w < 100 or crop_h < 100:
            print("⚠️  crop area too small, using original size")
            return images
        
        print(f"📐 cropping to area: ({min_x}, {min_y}) to ({max_x}, {max_y}) = {crop_w}x{crop_h}")
        
        # crop all images
        cropped_images = []
        for i, img in enumerate(images):
            cropped = img[min_y:max_y, min_x:max_x]
            cropped_images.append(cropped)
            print(f"✅ cropped image {i}: {cropped.shape}")
        
        return cropped_images
    
    def apply_quality_settings(self, images, quality='high'):
        """apply quality settings to final images before gif creation"""
        if not images:
            return images
            
        print(f"🎨 applying {quality} quality settings...")
        
        processed_images = []
        
        for i, img in enumerate(images):
            if quality == 'high':
                # high quality: minimal processing, preserve detail
                processed = img.copy()
                
            elif quality == 'medium':
                # medium quality: slight sharpening and noise reduction
                processed = img.copy()
                
                # slight gaussian blur to reduce noise
                processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
                
                # sharpen slightly
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                processed = cv2.filter2D(processed, -1, kernel * 0.1)
                processed = np.clip(processed, 0, 255).astype(np.uint8)
                
            elif quality == 'optimized':
                # optimized: balance quality and file size
                processed = img.copy()
                
                # slight noise reduction
                processed = cv2.bilateralFilter(processed, 5, 75, 75)
                
                # enhance contrast slightly
                lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                processed = cv2.merge([l, a, b])
                processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
                
            else:
                processed = img.copy()
            
            processed_images.append(processed)
            print(f"✅ processed image {i+1} with {quality} quality")
        
        return processed_images
    
    # properly bind methods to the processor instance
    processor.select_crop_and_reference = select_crop_and_reference.__get__(processor, NimsloProcessor)
    processor.align_images = align_images.__get__(processor, NimsloProcessor)
    processor.align_images_cnn = align_images_cnn.__get__(processor, NimsloProcessor)
    processor.match_histograms = match_histograms.__get__(processor, NimsloProcessor)
    processor.create_nimslo_gif = create_nimslo_gif.__get__(processor, NimsloProcessor)
    processor.create_comparison_gif = create_comparison_gif.__get__(processor, NimsloProcessor)
    processor.set_output_folder = set_output_folder.__get__(processor, NimsloProcessor)
    processor.preview_alignment = preview_alignment.__get__(processor, NimsloProcessor)
    processor.preview_final = preview_final.__get__(processor, NimsloProcessor)
    processor.crop_to_valid_area = crop_to_valid_area.__get__(processor, NimsloProcessor)
    processor.apply_quality_settings = apply_quality_settings.__get__(processor, NimsloProcessor)

def main():
    """main function to run the nimslo processor"""
    print("🎬 nimslo auto-aligning gif processor")
    print("=" * 50)
    
    # create processor
    processor = NimsloProcessor()
    add_methods_to_processor(processor)
    
    # load images
    print("\n📁 loading images...")
    if not processor.load_images():
        print("❌ failed to load images")
        return
    
    # show preview
    processor.show_images(save_path="preview_original.png")
    
    # interactive crop selection
    print("\n🎯 crop selection (close window when done)...")
    processor.select_crop_and_reference()
    
    # align images
    print("\n🧩 aligning images...")
    
    # force cnn alignment (no user choice)
    use_cnn = True
    print("🧠 using cnn border detection alignment (forced)")
    
    # try different alignment approaches if needed
    alignment_success = False
    reference_index = 0
    
    while not alignment_success:
        if use_cnn:
            print("🧠 using cnn border detection alignment...")
            if not processor.align_images_cnn(reference_index=reference_index):
                print("❌ cnn alignment failed")
                return
        else:
            print("🔍 using traditional feature-based alignment...")
            if not processor.align_images(reference_index=reference_index, transform_type='homography'):
                print("❌ alignment failed")
                return
        
        # show aligned preview
        processor.preview_alignment("aligned images")
        
        # auto-continue (no user confirmation)
        print(f"\n🎯 alignment preview (cnn border detection):")
        print("   - check preview_aligned.png to see alignment results")
        print("   - continuing automatically...")
        
        # crop to valid area to remove black bars
        processor.aligned_images = processor.crop_to_valid_area(processor.aligned_images)
        
        # simple retry logic - try different reference image if needed
        if reference_index == 0:
            reference_index = 1
            print(f"🔄 trying reference image {reference_index}")
        else:
            # if we've tried both reference images, just continue
            alignment_success = True
            print("✅ alignment complete, proceeding to histogram matching")
    
    # match histograms
    print("\n🌈 matching histograms...")
    if not processor.match_histograms(method='adaptive', strength=0.7):
        print("❌ histogram matching failed")
        return
    
    # show final preview
    processor.preview_final("final processed images")
    
    # auto-continue (no user confirmation)
    print("\n🎬 final preview:")
    print("   - check preview_final.png to see final result")
    print("   - proceeding to quality processing and gif creation...")
    
    # apply quality settings
    print("\n🎨 applying quality settings...")
    quality = 'high'  # default to high quality
    processor.matched_images = processor.apply_quality_settings(processor.matched_images, quality=quality)
    
    # create final gif with quality settings
    print(f"\n🎬 creating final {quality} quality gif...")
    if processor.create_nimslo_gif(duration=0.15, bounce=True, quality=quality):
        print("🎉 nimslo gif created successfully!")
    else:
        print("❌ gif creation failed")
    
    # cleanup preview images
    print("\n🧹 cleaning up preview images...")
    import os
    preview_files = ["preview_original.png", "preview_aligned.png", "preview_final.png"]
    for file in preview_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   ✅ removed {file}")
    
    print("\n✅ processing complete!")

if __name__ == "__main__":
    main() 