import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import openai
from typing import Dict, List, Any, Optional
from openai import AzureOpenAI

class LegendDialog(tk.Toplevel):
    def __init__(self, parent, image_name: str, object_data: Dict[str, Any], 
                 save_callback=None):
        super().__init__(parent)
        self.title(f"Image Legend - {image_name}")
        self.geometry("600x500")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        
        self.parent = parent
        self.image_name = image_name
        self.object_data = object_data
        self.save_callback = save_callback
        
        # Initialize variables
        self.legend_text = tk.StringVar(value="")
        
        # Initialize UI elements
        self.create_ui()
        
        # If we have existing legend data, load it
        if "legend" in object_data:
            self.legend_text.set(object_data["legend"])
            self.legend_textarea.delete("1.0", tk.END)
            self.legend_textarea.insert("1.0", object_data["legend"])
    
    def create_ui(self):
        """Create the user interface elements"""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Legend frame
        legend_frame = ttk.LabelFrame(main_frame, text="Image Description", padding="10")
        legend_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Info text
        ttk.Label(
            legend_frame, 
            text="Edit the legend below or generate a new one using Azure OpenAI:",
            wraplength=550
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Legend text area with scrollbars
        text_frame = ttk.Frame(legend_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar_y = ttk.Scrollbar(text_frame)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scrollbar_x = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.legend_textarea = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set,
            height=10
        )
        self.legend_textarea.pack(fill=tk.BOTH, expand=True)
        
        scrollbar_y.config(command=self.legend_textarea.yview)
        scrollbar_x.config(command=self.legend_textarea.xview)
        
        # Load existing legend if any
        if self.legend_text.get():
            self.legend_textarea.insert("1.0", self.legend_text.get())
        
        # Buttons frame
        btn_frame = ttk.Frame(legend_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            btn_frame, 
            text="Generate Legend", 
            command=self.generate_legend
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Save Legend",
            command=self.save_legend
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Close",
            command=self.destroy
        ).pack(side=tk.RIGHT, padx=5)
    
    def generate_legend(self):
        """Generate a legend using Azure OpenAI API"""
        try:
            # Format object data for prompt
            class_counts = self.object_data.get("class_counts", {})
            prompt = self._create_prompt(class_counts)
            
            # ---- OpenAI API Configuration ----
            API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
            if not API_KEY:
                messagebox.showerror("Error", "AZURE_OPENAI_API_KEY environment variable is not set.")
                return
                
            API_VERSION = "2023-03-15-preview"
            AZURE_ENDPOINT = "https://ai-rd-sweden.openai.azure.com/"
            
            client = AzureOpenAI(
                api_key=API_KEY, 
                api_version=API_VERSION, 
                azure_endpoint=AZURE_ENDPOINT
            )
            
            # Call API to generate text
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use the appropriate model deployed on your Azure instance
                messages=[
                    {"role": "system", "content": "You are an assistant that creates concise image descriptions in French based on object counts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content
            
            # Update text area with generated text
            self.legend_textarea.delete("1.0", tk.END)
            self.legend_textarea.insert("1.0", generated_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate legend: {e}")
    
    def _create_prompt(self, class_counts):
        """Create a prompt for the OpenAI model based on object data"""
        # Filter out classes with zero count
        actual_objects = {cls: count for cls, count in class_counts.items() if count > 0}
        
        objects_text = ", ".join([f"{count} {cls}{'s' if count > 1 else ''}" for cls, count in actual_objects.items()])
        
        # Build the prompt
        prompt = (
            f"Create a concise description in French for an image containing: {objects_text}.\n"
            f"The description should mention all objects and describe their relative positions. "
            f"Start with 'Dans cette image, on peut voir...' and be specific about quantities. "
            f"Keep it under 3 sentences."
        )
        
        return prompt
    
    def save_legend(self):
        """Save the legend text and call the callback"""
        legend_text = self.legend_textarea.get("1.0", tk.END).strip()
        
        if self.save_callback:
            self.save_callback(legend_text)
        
        messagebox.showinfo("Success", "Legend saved successfully.")

# Function to open the legend dialog
def open_legend_dialog(parent, image_name, object_data, save_callback=None):
    """
    Open the legend dialog
    
    Args:
        parent: The parent window
        image_name: Name of the image
        object_data: Dictionary containing object counts and other data
        save_callback: Callback function to save the legend
    """
    dialog = LegendDialog(parent, image_name, object_data, save_callback)
    return dialog

# For testing purposes - this runs only if this file is executed directly
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Legend Dialog Test")
    
    # Sample object data
    sample_data = {
        "class_counts": {
            "card": 9,
            "token": 2,
            "dice": 0,
            "person": 0
        },
        "legend": "Dans cette image, on peut voir 9 cartes et 2 jetons. Les jetons se trouvent à droite des cartes."
    }
    
    def save_test(text):
        print(f"Saved legend: {text}")
    
    # Button to open the dialog
    ttk.Button(
        root, 
        text="Open Legend Dialog",
        command=lambda: open_legend_dialog(root, "test_image", sample_data, save_test)
    ).pack(padx=20, pady=20)
    
    root.mainloop()
