import numpy as np

def remove_off_screen_lander(data_file, save_file):
    '''
    Loads a .npz file from lunarlander_datagenerator.py and removes entries
    where the lander is off screen.

    Args:
        data_file (str): Path to the file with the data
        save_file (str): File where the cleaned data will be stored
    
    Returns: (int) The size of the cleaned dataset
    '''
    
    data = dict(np.load(data_file))
    positions = [obs[:2] for obs in data["observations"]]
    for key, value in data.items():
        if key[:9] == "parameter":
            continue
        new_value = []
        for (x_pos, y_pos), v in zip(positions, value):
            if x_pos <= 1 and x_pos >= -1 and y_pos >= -0.5 and y_pos <= 1.5:
                new_value.append(v)
        data[key] = new_value

    np.savez(save_file, **data)

    return len(data["observations"])

if __name__ == "__main__":
    '''
    If run directly this will remove the entries of off screen landers
    from a predefined .npz file

    Parameters:
        LOAD_FILE (str): File path to the dataset to clean
        SAVE_FILE (str): File path to where the new dataset will be stored
    '''

    LOAD_FILE = "LunarLander-v2_105000_Dataset.npz"
    SAVE_FILE = "LunarLander-v2_105000_Cleaned"

    new_size = remove_off_screen_lander(LOAD_FILE, SAVE_FILE)

    print("New dataset {} created with size {}".format(SAVE_FILE, new_size))