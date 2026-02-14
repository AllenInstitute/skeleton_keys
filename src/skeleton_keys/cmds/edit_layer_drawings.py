import tkinter as tk
import numpy as np
import math
import json
from collections import defaultdict
from tkinter import messagebox
import argschema as ags

class DrawingEditSchema(ags.ArgSchema):
    input_file = ags.fields.InputFile(
        description="Path to input json file"
    )
    output_file = ags.fields.OutputFile(
        description = "Path to output json file",
    )


class LayerEditor(tk.Frame):

    def __init__(self, parent, input_file, output_file):
        tk.Frame.__init__(self, parent)
        
        self.input_file = input_file
        self.output_file = output_file
        self._load_data()
        self._create_canvas()
        self._populate_canvas()
        
    def _load_data(self):
        with open(self.input_file, "r") as f:
            data_dict = json.load(f)

        self.pia_dict = data_dict['pia_path']
        self.wm_dict = data_dict['wm_path']
        self.soma_dict = data_dict['soma_path']
        self.list_of_layer_dicts = data_dict['layer_polygons']

        self.list_of_drawing_dict = [
            self.pia_dict] + self.list_of_layer_dicts + [self.wm_dict, self.soma_dict]

    def _create_canvas(self):
        self.canvas = tk.Canvas(width=1000, height=1000, background="bisque")
        self.canvas.pack(fill="both", expand=True)
        
        # scale data to fit canvas
        min_x,min_y = np.inf,np.inf
        max_x,max_y = 0,0
        for draw_dict in self.list_of_drawing_dict:
            data = np.array(draw_dict['path'])

            this_min_x = min(point[0] for point in data)
            min_x = this_min_x if this_min_x < min_x else min_x 
            
            this_min_y = min(point[1] for point in data)
            min_y = this_min_y if this_min_y < min_y else min_y
            
            this_max_x = max(point[0] for point in data)
            max_x = this_max_x if this_max_x > max_x else max_x
            
            this_max_y = max(point[1] for point in data)
            max_y = this_max_y if this_max_y > max_y  else max_y

        x_span = max_x - min_x
        y_span = max_y - min_y

        offset = 0.1
        norm_range = (offset, 1 - offset)

        x_scale = (norm_range[1] - norm_range[0])
        y_scale = (norm_range[1] - norm_range[0])
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.norm_range = norm_range

    def _populate_canvas(self):
        
        # this data is used to keep track of an item being dragged
        self._drag_data = {"x": 0, "y": 0, "item": None}
        self.token_order = defaultdict(list)
        self.token_coords = {}
                
        self.color_dict = {
            "Layer1":'red',
            "Layer2/3":'orange',
            "Layer4":'yellow',
            "Layer5":"green",
            "Layer6a":'blue',
            "Layer6b":'purple',
            "Pia":"grey",
            'White Matter':'black',
            "Soma":'black'
        }
        
        for draw_dict in self.list_of_drawing_dict:
            data = np.array(draw_dict['path'])
            draw_name = draw_dict['name']
            color = self.color_dict[draw_name]
            for d in data:           
                scaled_x = 1000 * (((d[0] - self.min_x) / (self.max_x - self.min_x)) * self.x_scale + self.norm_range[0])
                scaled_y = 1000 * (((d[1] - self.min_y) / (self.max_y - self.min_y)) * self.y_scale + self.norm_range[0])
                token_id = self.create_token(scaled_x, scaled_y, color, 5)
                self.token_coords[token_id] = (scaled_x, scaled_y)
                self.token_order[draw_name].append(token_id)

        # Draw lines connecting tokens
        self.draw_lines()
        # bind canvas click to add or move a token based on the mode
        self.canvas.bind("<Button-1>", self.canvas_click)
        
        self.add_token_mode_variables = {}
        for draw_dict in self.list_of_drawing_dict:
            name = draw_dict['name']
            this_bool_var = tk.BooleanVar()
            self.add_token_mode_variables[name] = this_bool_var

            add_token_button = tk.Checkbutton(self, 
                                              text=f"Add {name}", 
                                              variable=self.add_token_mode_variables[name]
                                              )
            add_token_button.pack(side="left", padx=5)

        self.eraser_mode = tk.BooleanVar()
        eraser_button = tk.Checkbutton(self, 
                                              text="Node Eraser", 
                                              variable=self.eraser_mode
                                              )
        eraser_button.pack(side="left", padx=5)
        
        # Create a Save button
        save_button = tk.Button(self, text="Save", command=lambda: self.save_canvas_state(self.output_file))
        save_button.pack(side="left", padx=5)      


        
    def save_canvas_state(self, output_file):
        """Save the current state of the canvas to a JSON file"""
        name_dict = {"Pia":"pia_path", 'White Matter':"wm_path", "Soma":"soma_path"}
        final_dict = {}
        layer_drawings = []
        for draw_dict in self.list_of_drawing_dict:
            draw_name = draw_dict['name']
            
            token_coords_original_space = []
            for token_id in self.token_order[draw_name]:
                token_coord_canvas_space = self.token_coords[token_id]
                original_x = (token_coord_canvas_space[0] / 1000 - self.norm_range[0]) / self.x_scale * (self.max_x - self.min_x) + self.min_x
                original_y = (token_coord_canvas_space[1] / 1000 - self.norm_range[0]) / self.y_scale * (self.max_y - self.min_y) + self.min_y
                token_coords_original_space.append([original_x, original_y])
                    
            new_draw_dict = {k:v for k,v in draw_dict.items()}   
            new_draw_dict['path'] = token_coords_original_space
            
            if "Layer" in draw_name:
                layer_drawings.append(new_draw_dict)
                
            else:
                final_dict[name_dict[draw_name]] = new_draw_dict
                
        final_dict['layer_polygons'] = layer_drawings

        with open(output_file, 'w') as f:
            json.dump(final_dict, f)
    
    def canvas_click(self, event):
        """Handle canvas click to add or move a token based on the mode"""
        x, y = event.x, event.y

        if not any([v.get() for v in self.add_token_mode_variables.values()]) and not self.eraser_mode.get():
            # If neither add token mode nor eraser mode is active, handle drag and drop
            self.drag_start(event)
            self.canvas.bind("<ButtonRelease-1>", self.drag_stop)
            self.canvas.bind("<B1-Motion>", self.drag)

        elif any([v.get() for v in self.add_token_mode_variables.values()]) and not self.eraser_mode.get():
            # If only add token mode is active
            self.add_token(event)

        elif not any([v.get() for v in self.add_token_mode_variables.values()]) and self.eraser_mode.get():
            # If only eraser mode is active
            self.delete_nearest_token(x, y)
        else:
            # multiple modes selected
            token_type_to_add = [k for k,v in self.add_token_mode_variables.items() if v.get()]
            modes_selected = ["Node Eraser"] + token_type_to_add
            error_message = "Specify only one mode type. currently.\n"
            error_message += "You have selected:\n{}".format(modes_selected)
            # Display an error message in a pop-up box
            messagebox.showerror("Error", error_message)

    def add_token(self, event):
        """Add token at the clicked position"""
        x, y = event.x, event.y
        token_type_to_add = [k for k, v in self.add_token_mode_variables.items() if v.get()]
        if len(token_type_to_add) != 1:
            error_message = "PLEASE SPECIFY ONLY ONE COORDINATE TYPE TO ADD.\n"
            error_message += "You have selected:\n{}".format(token_type_to_add)
            # Display an error message in a pop-up box
            messagebox.showerror("Error", error_message)
        else:
            token_type_to_add = token_type_to_add[0]
            token_type_color = self.color_dict[token_type_to_add]
            nearest_tokens = self.find_two_nearest_tokens(x, y, token_type_to_add)
            nearest_token_indices = [self.token_order[token_type_to_add].index(t) for t in nearest_tokens]
            if nearest_tokens:
                new_token_id = self.create_token(x, y, token_type_color, 5)
                nearest_token_1_idx = max(nearest_token_indices)
                self.token_coords[new_token_id] = (x, y)
                self.token_order[token_type_to_add].insert(nearest_token_1_idx, new_token_id)
                self.draw_lines()
            
    def create_token(self, x, y, color, radius):
        """Create a token at the given coordinate in the given color"""
        token_id = self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            outline=color,
            fill=color,
            tags=("token",),
        )
        return token_id

    def draw_lines(self):
        """Draw lines connecting tokens"""
        self.canvas.delete("line")  # Delete existing lines        
        for label, token_order_list in self.token_order.items():
            line_color = self.color_dict[label]
            for i in range(len(token_order_list) - 1):
                k1, k2 = token_order_list[i], token_order_list[i + 1]
                x1, y1 = self.token_coords[k1]
                x2, y2 = self.token_coords[k2]
                self.canvas.create_line(x1, y1, x2, y2, fill=line_color, tags=("line",))

    def update_lines(self):
        """Update lines connecting tokens"""
        self.canvas.delete("line")  # Delete existing lines
        self.draw_lines()  # Draw new lines based on updated coordinates

    def find_nearest_token(self, x, y):
        """Find the nearest token to the given coordinates"""
        min_distance = 5
        nearest_token = None
        for token_id, (token_x, token_y) in self.token_coords.items():
            distance = math.sqrt((x - token_x)**2 + (y - token_y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_token = token_id
        return nearest_token

    def drag_start(self, event):
        """Beginning drag of an object"""
        x, y = event.x, event.y
        nearest_token = self.find_nearest_token(x, y)
        if nearest_token is not None:
            self._drag_data["item"] = nearest_token
            self._drag_data["x"] = x
            self._drag_data["y"] = y

    def drag_stop(self, event):
        """End drag of an object"""
        # Reset the drag information
        self._drag_data["item"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0
        # Update lines after dragging
        self.update_lines()

    def drag(self, event):
        """Handle dragging of an object"""
        # Compute how much the mouse has moved
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]
        item_id = self._drag_data["item"]
        if item_id is not None:
            # Move the object the appropriate amount
            self.canvas.move(self._drag_data["item"], delta_x, delta_y)
            # Record the new position
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y
            self.token_coords[item_id] = (event.x, event.y)
            # Update lines during dragging
            self.update_lines()    
    
    def point_to_line_distance(self, x, y, x1, y1, x2, y2):
        # Step 1: Find the equation of the line passing through (x1, y1) and (x2, y2)
        m = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # slope of the line
        c = y1 - m * x1  # y-intercept of the line

        # Step 2: Determine the perpendicular distance from the point (x, y) to the line
        if m != 0:
            perpendicular_distance = abs(m * x - y + c) / math.sqrt(m**2 + 1)
        else:
            perpendicular_distance = abs(x - x1)

        # Step 3: Check if the perpendicular projection of the point is within the line segment
        if x1 <= x <= x2 or x2 <= x <= x1:
            if y1 <= y <= y2 or y2 <= y <= y1:
                return perpendicular_distance

        # If the perpendicular projection is outside the line segment, find the distance to the nearest endpoint
        distance_to_endpoint1 = math.sqrt((x - x1)**2 + (y - y1)**2)
        distance_to_endpoint2 = math.sqrt((x - x2)**2 + (y - y2)**2)

        return min(distance_to_endpoint1, distance_to_endpoint2)

        
    def find_two_nearest_tokens(self, x, y, token_type):
        """Find the two nearest tokens to the given coordinates"""
        min_distance = float('inf')
        nearest_tokens = None
        for i in range(len(self.token_order[token_type]) - 1):
            k1, k2 = self.token_order[token_type][i], self.token_order[token_type][i + 1]
            x1, y1 = self.token_coords[k1]
            x2, y2 = self.token_coords[k2]

            # Calculate the distance from the point to the line formed by tokens k1 and k2
            distance = self.point_to_line_distance(x, y, x1, y1, x2, y2)
            if distance < min_distance:
                min_distance = distance
                nearest_tokens = (k1, k2)
        return nearest_tokens
    
    def delete_nearest_token(self, x, y):
        
        min_distance = float('inf')
        nearest_token = None
        nearest_token_type = ""
        for token_type, token_order in self.token_order.items():
            for token_id in token_order:
                token_coord = self.token_coords[token_id]
                dist = math.sqrt((x - token_coord[0])**2 + (y - token_coord[1])**2)
                if dist<min_distance:
                    min_distance=dist
                    nearest_token = token_id
                    nearest_token_type = token_type
               
        # remove it from our records           
        self.token_order[nearest_token_type].remove(nearest_token)
        del self.token_coords[nearest_token]
        
        # destroy token from display 
        self.canvas.delete(nearest_token)

        self.draw_lines()
    

def console_script():
    module = ags.ArgSchemaParser(schema_type=DrawingEditSchema)
    args = module.args
    root = tk.Tk()
    LayerEditor(root, args['input_file'], args['output_file']).pack(fill="both", expand=True)
    root.mainloop()

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=DrawingEditSchema)
    args = module.args
    root = tk.Tk()
    LayerEditor(root, args['input_file'], args['output_file']).pack(fill="both", expand=True)
    root.mainloop()