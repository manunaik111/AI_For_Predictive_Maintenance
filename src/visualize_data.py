import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Ensure this path correctly points to your processed data directory
PROCESSED_DATA_DIR = r'C:\project file 002\processed data001'
X_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'X_data.npy')
Y_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'y_data.npy')

def visualize_data_point(x_data, y_data, index):
    """
    Plots a single vibration segment and displays its corresponding RUL label.
    """
    # Get the specific segment (the "question") and label (the "answer")
    segment = x_data[index]
    label_rul = y_data[index]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(segment, color='royalblue')

    # Set clear titles and labels for the plot
    ax.set_title(f"Vibration Segment (Sample Index: {index})", fontsize=16)
    ax.set_xlabel("Time Steps within Segment", fontsize=12)
    ax.set_ylabel("Normalized Vibration", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Display the all-important SUPERVISED LABEL in a prominent way
    fig.suptitle(f'Supervised Label (RUL): {int(label_rul):,} cycles', fontsize=20, color='red', weight='bold')

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Supervised Learning Data Visualizer ---")
    
    # Load the datasets
    try:
        print(f"Loading data from {PROCESSED_DATA_DIR}...")
        X_data = np.load(X_DATA_FILE)
        y_data = np.load(Y_DATA_FILE)
        print("Data loaded successfully.")
        max_index = len(X_data) - 1
        print(f"You can choose any index from 0 to {max_index}.")
    except FileNotFoundError:
        print(f"ERROR: Could not find data files in '{PROCESSED_DATA_DIR}'. Please check the path.")
        exit()

    # Interactive loop to get user input
    while True:
        try:
            user_input = input(f"\nEnter an index to visualize (0-{max_index}) or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
            
            index_to_show = int(user_input)
            
            if 0 <= index_to_show <= max_index:
                visualize_data_point(X_data, y_data, index_to_show)
            else:
                print(f"Invalid index. Please enter a number between 0 and {max_index}.")

        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
        except Exception as e:
            print(f"An error occurred: {e}")
