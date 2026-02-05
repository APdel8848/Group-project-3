import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

# Group Assignment 3
# Image Editor using Tkinter and OpenCV

# Class to store settings for the slider
# We use this to keep track of values even when switching between effects
class FilterSettings:
    def __init__(self):
        self.brightness = 0
        self.blur = 0
        self.contrast = 1.0
        self.scale = 100
        self.grayscale = False
    
    def reset(self):
        self.brightness = 0
        self.blur = 0
        self.contrast = 1.0
        self.scale = 100
        self.grayscale = False

# Class that handles the actual image processing
# This separates the logic from the GUI code
class ImageProcessor:
    def __init__(self):
        self.true_original = None   # The actual file we loaded
        self.original_image = None  # The working copy for edits
        self.current_image = None   # The final image shown on screen
        self.history = []           # For undo feature
        self.redo_stack = []        # For redo feature

    def load_file(self, file_path):
        image = cv2.imread(file_path)
        # OpenCV loads as BGR by default, so we need to convert to RGB 
        # otherwise the colors look wrong in Tkinter
        self.true_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reset everything for the new file
        self.original_image = self.true_original.copy()
        self.current_image = self.original_image.copy()
        self.history = [] 
        self.redo_stack = []
        return self.current_image

    def save_file(self, file_path):
        if self.current_image is None: return
        # Need to convert back to BGR before saving to disk
        save_img = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, save_img)

    def revert_to_original(self):
        # Goes back to the original loaded file
        if self.true_original is None: return None
        self.original_image = self.true_original.copy()
        self.history = []
        self.redo_stack = []
        return self.original_image

    # Helper function to save history before making changes
    def save_state_for_undo(self):
        if self.original_image is not None:
            self.history.append(self.original_image.copy())
            self.redo_stack.clear()

    def undo(self):
        if not self.history: return None
        self.redo_stack.append(self.original_image.copy())
        self.original_image = self.history.pop()
        return self.original_image

    def redo(self):
        if not self.redo_stack: return None
        self.history.append(self.original_image.copy())
        self.original_image = self.redo_stack.pop()
        return self.original_image

    # This function applies all the sliders at once
    # We do this so brightness, blur, and resize work together smoothly
    def apply_transformations(self, settings):
        if self.original_image is None: return None
        
        temp_img = self.original_image.copy()

        # Check if grayscale is on
        if settings.grayscale:
            gray = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
            temp_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Apply brightness and contrast using the formula: new = alpha*old + beta
        temp_img = cv2.convertScaleAbs(temp_img, alpha=settings.contrast, beta=settings.brightness)

        # Apply blur if the value is greater than 0
        if settings.blur > 0:
            # Kernel size needs to be odd (1, 3, 5...)
            ksize = (settings.blur * 2) + 1
            temp_img = cv2.GaussianBlur(temp_img, (ksize, ksize), 0)

        # Handle resizing
        if settings.scale != 100:
            width = int(temp_img.shape[1] * settings.scale / 100)
            height = int(temp_img.shape[0] * settings.scale / 100)
            
            # Make sure width and height don't hit 0 to avoid crash
            if width < 1: width = 1
            if height < 1: height = 1
            
            temp_img = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_AREA)

        self.current_image = temp_img
        return self.current_image

    # These functions permanently change the base image
    
    def apply_canny_edge(self):
        self.save_state_for_undo()
        # Edge detection works best on grayscale images
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        self.original_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return self.original_image

    def rotate_90(self):
        self.save_state_for_undo()
        self.original_image = cv2.rotate(self.original_image, cv2.ROTATE_90_CLOCKWISE)
        return self.original_image

    def flip(self, mode):
        self.save_state_for_undo()
        # 1 is horizontal, 0 is vertical
        self.original_image = cv2.flip(self.original_image, mode)
        return self.original_image
    
# Main GUI Class
class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Group 3 Image Editor")
        self.root.geometry("1200x850")

        self.processor = ImageProcessor()
        self.settings = FilterSettings()
        
        self.create_menu()
        self.create_layout()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit Menu for Undo/Redo
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo_action)
        edit_menu.add_command(label="Redo", command=self.redo_action)

    def create_layout(self):
        bg_color = '#d9d9d9'
        text_color = 'black'

        # Status Bar at the bottom
        self.status_label = tk.Label(self.root, text="Welcome! Open an image.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Left Sidebar
        self.sidebar = tk.Frame(self.root, width=260, bg=bg_color)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        
        # File Operations Section
        frame_file = tk.LabelFrame(self.sidebar, text="File Operations", bg=bg_color, fg=text_color, font=("Arial", 10, "bold"))
        frame_file.pack(fill=tk.X, pady=5, padx=10)
        
        tk.Button(frame_file, text="Open Image", bg="white", fg="black", highlightbackground=bg_color, command=self.open_image).pack(pady=5, padx=5, fill=tk.X)
        tk.Button(frame_file, text="Save Image", bg="white", fg="black", highlightbackground=bg_color, command=self.save_image).pack(pady=5, padx=5, fill=tk.X)
        tk.Button(frame_file, text="Revert to Original", bg="white", fg="black", highlightbackground=bg_color, command=self.revert_all).pack(pady=5, padx=5, fill=tk.X)

        # Geometry Section
        frame_geo = tk.LabelFrame(self.sidebar, text="Geometry", bg=bg_color, fg=text_color, font=("Arial", 10, "bold"))
        frame_geo.pack(fill=tk.X, pady=5, padx=10)

        f_row1 = tk.Frame(frame_geo, bg=bg_color)
        f_row1.pack(fill=tk.X, padx=2)
        tk.Button(f_row1, text="Rotate 90", bg="white", fg="black", highlightbackground=bg_color, command=self.rotate_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2, pady=2)
        tk.Button(f_row1, text="Flip Horiz", bg="white", fg="black", highlightbackground=bg_color, command=lambda: self.flip_image(1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2, pady=2)
        
        f_row2 = tk.Frame(frame_geo, bg=bg_color)
        f_row2.pack(fill=tk.X, padx=2)
        tk.Button(f_row2, text="Flip Vert", bg="white", fg="black", highlightbackground=bg_color, command=lambda: self.flip_image(0)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2, pady=2)
        tk.Button(f_row2, text="Undo", bg="white", fg="black", highlightbackground=bg_color, command=self.undo_action).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2, pady=2)

        # Sliders Section
        frame_adj = tk.LabelFrame(self.sidebar, text="Adjustments", bg=bg_color, fg=text_color, font=("Arial", 10, "bold"))
        frame_adj.pack(fill=tk.X, pady=5, padx=10)

        tk.Label(frame_adj, text="Resize (%)", bg=bg_color, fg=text_color, font=("Arial", 9)).pack(pady=(5,0))
        self.scale_slider = tk.Scale(frame_adj, from_=10, to=200, orient=tk.HORIZONTAL, bg=bg_color, fg=text_color, command=self.update_filters)
        self.scale_slider.set(100)
        self.scale_slider.pack(padx=5, fill=tk.X)

        tk.Label(frame_adj, text="Brightness", bg=bg_color, fg=text_color, font=("Arial", 9)).pack(pady=(5, 0))
        self.brightness_slider = tk.Scale(frame_adj, from_=-100, to=100, orient=tk.HORIZONTAL, bg=bg_color, fg=text_color, command=self.update_filters)
        self.brightness_slider.pack(padx=5, fill=tk.X)

        tk.Label(frame_adj, text="Contrast", bg=bg_color, fg=text_color, font=("Arial", 9)).pack(pady=(5, 0))
        self.contrast_slider = tk.Scale(frame_adj, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, bg=bg_color, fg=text_color, command=self.update_filters)
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(padx=5, fill=tk.X)

        tk.Label(frame_adj, text="Blur", bg=bg_color, fg=text_color, font=("Arial", 9)).pack(pady=(5, 0))
        self.blur_slider = tk.Scale(frame_adj, from_=0, to=20, orient=tk.HORIZONTAL, bg=bg_color, fg=text_color, command=self.update_filters)
        self.blur_slider.pack(padx=5, fill=tk.X)

        tk.Button(frame_adj, text="Reset Sliders", bg="white", fg="black", highlightbackground=bg_color, command=self.reset_sliders).pack(pady=10, padx=5, fill=tk.X)

        # Special Effects Section
        frame_spec = tk.LabelFrame(self.sidebar, text="Special", bg=bg_color, fg=text_color, font=("Arial", 10, "bold"))
        frame_spec.pack(fill=tk.X, pady=5, padx=10)
        
        self.btn_bw = tk.Button(frame_spec, text="Toggle Black & White", bg="white", fg="black", highlightbackground=bg_color, command=self.toggle_bw)
        self.btn_bw.pack(pady=5, padx=5, fill=tk.X)
        
        tk.Button(frame_spec, text="Canny Edge Detect", bg="white", fg="black", highlightbackground=bg_color, command=self.apply_edge_detect).pack(pady=5, padx=5, fill=tk.X)

        # Main Canvas Area
        self.canvas = tk.Canvas(self.root, bg='grey')
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            img = self.processor.load_file(file_path)
            self.reset_sliders()
            self.update_display(img, file_path)

    def save_image(self):
        if self.processor.current_image is None: 
            messagebox.showwarning("Warning", "No image to save!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if file_path:
            self.processor.save_file(file_path)
            messagebox.showinfo("Success", "Image Saved!")

    def update_filters(self, _):
        # Update settings
        self.settings.brightness = int(self.brightness_slider.get())
        self.settings.blur = int(self.blur_slider.get())
        self.settings.contrast = float(self.contrast_slider.get())
        self.settings.scale = int(self.scale_slider.get())
        
        # Apply the new settings
        new_img = self.processor.apply_transformations(self.settings)
        self.update_display(new_img)

    def toggle_bw(self):
        self.settings.grayscale = not self.settings.grayscale
        
        if self.settings.grayscale:
            self.btn_bw.config(text="Toggle Color (B&W On)", bg="#ccc")
        else:
            self.btn_bw.config(text="Toggle Black & White", bg="white")
            
        new_img = self.processor.apply_transformations(self.settings)
        self.update_display(new_img)

    def rotate_image(self):
        self.processor.rotate_90()
        new_img = self.processor.apply_transformations(self.settings)
        self.update_display(new_img)

    def flip_image(self, mode):
        self.processor.flip(mode)
        new_img = self.processor.apply_transformations(self.settings)
        self.update_display(new_img)
        
    def apply_edge_detect(self):
        self.processor.apply_canny_edge()
        new_img = self.processor.apply_transformations(self.settings)
        self.update_display(new_img)

    def undo_action(self):
        base_img = self.processor.undo()
        if base_img is not None:
            new_img = self.processor.apply_transformations(self.settings)
            self.update_display(new_img)
        else:
            messagebox.showinfo("Info", "Nothing to Undo")

    def redo_action(self):
        base_img = self.processor.redo()
        if base_img is not None:
            new_img = self.processor.apply_transformations(self.settings)
            self.update_display(new_img)

    def revert_all(self):
        base_img = self.processor.revert_to_original()
        if base_img is not None:
            self.reset_sliders() 
        else:
            messagebox.showwarning("Info", "No image loaded!")

    def reset_sliders(self):
        self.settings.reset()
        self.brightness_slider.set(0)
        self.blur_slider.set(0)
        self.contrast_slider.set(1.0)
        self.scale_slider.set(100)
        self.btn_bw.config(text="Toggle Black & White", bg="white")
        
        new_img = self.processor.apply_transformations(self.settings)
        self.update_display(new_img)

    def update_display(self, img_array, filename=None):
        if img_array is None: return
        
        h, w, _ = img_array.shape
        info = f"Size: {w}x{h}"
        if filename: info += f" | File: {filename}"
        self.status_label.config(text=info)

        image = Image.fromarray(img_array)
        
        # Fit image to the canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 50: canvas_w, canvas_h = 800, 600
        
        # Only resize the display version, not the real image
        img_w, img_h = image.size
        if img_w > canvas_w or img_h > canvas_h:
            image.thumbnail((canvas_w, canvas_h))
        
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=self.tk_image, anchor=tk.CENTER)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
