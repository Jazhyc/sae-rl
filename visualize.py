import tkinter as tk
from tkinter import ttk
import pickle
from functools import partial

# Constants for window size
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 650
BOARD_SIZE = 300
FEATURE_WIDTH = 400

# Load the pickle dump
with open('output/features.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f"Loaded {len(data)} board states")

# Prepare the list of board states
board_states_list = list(data.keys())
current_board_index = 0
board_values = [''] * 9

# Function to calculate the mean of feature values for a given board state
def get_mean_features(board_state):
    if board_state in data:
        actions = data[board_state]
        feature_sums = {}
        counts = {}
        for action in actions:
            for feature, value in action:
                feature_name = feature.label
                feature_sums[feature_name] = feature_sums.get(feature_name, 0) + value
                counts[feature_name] = counts.get(feature_name, 0) + 1
        mean_features = {k: feature_sums[k] / counts[k] for k in feature_sums}
        # Get the top 5 features by absolute mean value
        top_features = sorted(mean_features.items(), key=lambda x: -abs(x[1]))[:5]
        return top_features
    else:
        return []
    
def translate_board(board_state):
    for i, cell in enumerate(board_state):
        if cell == '':
            board_state[i] = i + 1
    return board_state

# Style configurations
COLORS = {
    'bg': '#f0f0f0',
    'button_bg': '#ffffff',
    'x_color': '#FF6B6B',
    'o_color': '#4ECDC4',
    'grid_color': '#2C3E50'
}

def setup_styles():
    style = ttk.Style()
    style.configure('GameButton.TButton', 
                    padding=20,
                    font=('Arial', 24, 'bold'),
                    background=COLORS['button_bg'])
    style.configure('Next.TButton',
                    padding=10,
                    font=('Arial', 12),
                    background=COLORS['button_bg'])
    style.configure('Feature.Horizontal.TProgressbar',
                    troughcolor=COLORS['bg'],
                    background=COLORS['o_color'])

def create_game_button(parent, index):
    btn = ttk.Button(parent, style='GameButton.TButton')
    btn.grid(row=index//3, column=index%3, padx=5, pady=5, sticky='nsew')
    return btn

def update_features():
    board_state = tuple(translate_board(board_values.copy()))
    top_features = get_mean_features(board_state)
    for widget in feature_frame.winfo_children():
        widget.destroy()

    if top_features:
        for feature, value in top_features:
            frame = ttk.Frame(feature_frame)
            frame.pack(fill='x', pady=5, padx=10)
            # Feature name
            name_label = ttk.Label(frame, 
                                 text=feature[:50] + "..." if len(feature) > 50 else feature,
                                 font=('Arial', 10))
            name_label.pack(anchor='w')

            # Add tooltip to name_label
            tooltip = Tooltip(name_label, feature)
            
            # Progress bar for value visualization
            progress = ttk.Progressbar(frame, 
                                     style='Feature.Horizontal.TProgressbar',
                                     length=200, 
                                     mode='determinate')
            progress.pack(fill='x', pady=(2, 5))

            # Set progress value (convert from -1:1 to 0:100 range)
            progress['value'] = (value + 1) * 50

            # Value label
            value_label = ttk.Label(frame, 
                                  text=f"{value:.3f}",
                                  font=('Arial', 10, 'bold'))
            value_label.pack(anchor='e')
    else:
        label = ttk.Label(feature_frame, 
                         text="No data for this state.",
                         font=('Arial', 12, 'italic'))
        label.pack(pady=20)

def update_board_display():
    for i, value in enumerate(board_values):
        if value == 'X':
            buttons[i].configure(text='X')
        elif value == 'O':
            buttons[i].configure(text='O')
        else:
            buttons[i].configure(text='')

def on_cell_click(index):
    current_value = board_values[index]
    if current_value == '':
        board_values[index] = 'X'
        buttons[index]['text'] = 'X'
    elif current_value == 'X':
        board_values[index] = 'O'
        buttons[index]['text'] = 'O'
    else:
        board_values[index] = ''
        buttons[index]['text'] = ''
    update_features()

def next_board_state():
    global current_board_index, board_values
    current_board_index = (current_board_index + 1) % len(board_states_list)
    next_state = board_states_list[current_board_index]
    # Update board_values
    board_values = list(next_state)
    # Replace integers (positions) with empty strings
    board_values = [str(cell) if cell in ['X', 'O'] else '' for cell in board_values]
    update_board_display()
    update_features()
    update_counter()

def update_counter():
    counter_label.config(text=f"State {current_board_index + 1} of {len(board_states_list)}")

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        widget.bind("<Enter>", self.enter)
        widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        self.schedule = self.widget.after(500, self.showtip)

    def leave(self, event=None):
        if self.schedule:
            self.widget.after_cancel(self.schedule)
            self.schedule = None
        self.hidetip()

    def showtip(self):
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 20
        y = y + cy + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.overrideredirect(1)
        tw.geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("Arial", 10, 'normal'))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

# Initialize the main window
root = tk.Tk()
root.title("TicTacToe Feature Visualizer")
root.configure(bg=COLORS['bg'])
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
setup_styles()

# Main container
main_container = ttk.Frame(root, padding="20")
main_container.pack(fill='both', expand=True)

# Configure grid in main_container
main_container.columnconfigure(0, weight=2)
main_container.columnconfigure(1, weight=3)
main_container.rowconfigure(1, weight=1)

# Title
title = ttk.Label(main_container,
                  text="TicTacToe Feature Visualization",
                  font=('Arial', 16, 'bold'))
title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

# Game board frame
board_frame = ttk.Frame(main_container)
board_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)

# Configure grid
for i in range(3):
    board_frame.grid_columnconfigure(i, weight=1)
    board_frame.grid_rowconfigure(i, weight=1)

# Create buttons
buttons = [create_game_button(board_frame, i) for i in range(9)]
for i, button in enumerate(buttons):
    button.configure(command=partial(on_cell_click, i))

# Feature display
feature_frame = ttk.LabelFrame(main_container,
                               text="Top Features",
                               padding="10")
feature_frame.grid(row=1, column=1, sticky='nsew', padx=10, pady=10)

# Control panel
control_frame = ttk.Frame(main_container)
control_frame.grid(row=2, column=0, columnspan=2, pady=10)

counter_label = ttk.Label(control_frame,
                          text=f"State {current_board_index + 1} of {len(board_states_list)}",
                          font=('Arial', 12))
counter_label.pack(pady=5)

next_button = ttk.Button(control_frame,
                         text="Next Board State",
                         command=next_board_state,
                         style='Next.TButton')
next_button.pack()

update_features()
update_board_display()
update_counter()

root.mainloop()