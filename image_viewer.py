import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ImageViewer(tk.Tk):
    def __init__(self, dir1, dir2, dir3, dir4, dir5):
        super().__init__()

        self.dir1 = dir1
        self.dir2 = dir2
        self.dir3 = dir3
        self.dir4 = dir4
        self.dir5 = dir5

        self.image_dirs = [dir1, dir2, dir3, dir4, dir5]
        self.current_dir_index = 0

        self.image_files = []
        self.image_index = 0

        self.title("Image Viewer")
        self.geometry("800x1000")

        self.canvas = tk.Canvas(self, width=800, height=800)
        self.canvas.pack()

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side=tk.BOTTOM)

        self.prev_button = ttk.Button(self.button_frame, text="<< Prev", command=self.prev_page)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = ttk.Button(self.button_frame, text="Next >>", command=self.next_page)
        self.next_button.pack(side=tk.LEFT)

        self.refresh_button = ttk.Button(self.button_frame, text="Set All", command=self.refresh_all)
        self.refresh_button.pack(side=tk.LEFT)

        self.dir_label = ttk.Label(self, text="Current Directory: " + self.image_dirs[self.current_dir_index])
        self.dir_label.pack()

        self.image_labels = []
        for i in range(25):
            lbl = ttk.Label(self.canvas)
            lbl.bind("<Button-1>", lambda event, idx=i: self.on_image_click(event, idx))
            self.image_labels.append(lbl)
            self.canvas.create_window((i % 5) * 160 + 80, (i // 5) * 160 + 80, window=lbl)

        self.load_images()

    def load_images(self):
        self.image_files = [f for f in os.listdir(self.image_dirs[self.current_dir_index]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for i in range(25):
            if self.image_index + i < len(self.image_files):
                file_name = self.image_files[self.image_index + i]
                file_path = os.path.join(self.image_dirs[self.current_dir_index], file_name)
                image = Image.open(file_path)
                image.thumbnail((150, 150))
                photo = ImageTk.PhotoImage(image)
                self.image_labels[i].config(image=photo)
                self.image_labels[i].image = photo
                self.image_labels[i].file_name = file_name
            else:
                self.image_labels[i].config(image=None)
                self.image_labels[i].image = None
                self.image_labels[i].file_name = None

    def next_page(self):
        if self.image_index + 25 < len(self.image_files):
            self.image_index += 25
            self.load_images()

    def prev_page(self):
        if self.image_index - 25 >= 0:
            self.image_index -= 25
            self.load_images()

    def refresh_all(self):
        self.load_images()

    def on_image_click(self, event, idx):
        self.current_dir_index = (self.current_dir_index + 1) % 5
        self.dir_label.config(text="Current Directory: " + self.image_dirs[self.current_dir_index])

        file_name = self.image_labels[idx].file_name
        if file_name:
            file_path = os.path.join(self.image_dirs[self.current_dir_index], file_name)
            image = Image.open(file_path)
            image.thumbnail((150, 150))
            photo = ImageTk.PhotoImage(image)
            self.image_labels[idx].config(image=photo)
            self.image_labels[idx].image = photo

if __name__ == "__main__":
    dir1 = "leaf_samples/224x224/leaf_tiles"
    dir2 = "leaf_samples/224x224/s_and_v_normalized_masked"
    dir3 = "leaf_samples/224x224/leaf_tiles"
    dir4 = "leaf_samples/224x224/s_and_v_normalized_masked"
    dir5 = "leaf_samples/224x224/leaf_tiles"
    app = ImageViewer(dir1, dir2, dir3, dir4, dir5)
    app.mainloop()