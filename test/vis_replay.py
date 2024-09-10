import pickle
import pprint

if __name__ == '__main__':
    # Path to the pickle file
    file_path = 'data/replay_sample.pkl'

    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Print the content
    print("Content of the pickle file:")
    pprint.pprint(data, width=80, depth=None, compact=False)