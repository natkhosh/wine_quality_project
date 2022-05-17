import pandas as pd
import os
import pathlib
import pickle


def load_dataset(dataset_name, load_dir='external'):
    """
    Loads a Dataset object by name.
    :param dataset_name: name of dataset to load
    :param load_dir: directory to load the file
    :return: Pandas DataFrame
    """
    project_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    data_path = project_dir / 'data' / load_dir
    df = pd.read_csv(data_path / dataset_name)

    return df


def save_dataset(df, dataset_name, save_dir='intermid'):
    """
    Saves a Dataset object by name.
    :param df: Pandas DataFrame
    :param dataset_name: name of dataset to save
    :param save_dir: directory to save the file
    """
    project_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    data_path = project_dir / 'data' / save_dir
    df.to_csv(data_path / dataset_name, encoding='utf-8', index=False)
    
    
def save_model(model, model_name, save_dir='models'):
    """
    Saves a Model object by name.
    :param model: Sklearn model
    :param model_name: name of model to save
    :param save_dir: directory to save the file
    """
    project_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    data_path = project_dir / save_dir 
    file_name = str(data_path) + '/' + model_name + '.pkl'
    
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)
        
        
def load_model(model_name, load_dir='models'):
    """
    Loads a Model object by name.
    :param model_name: name of model to load
    :param load_dir: directory to load the file
    :return: Sklearn model
    """
    project_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    data_path = project_dir / load_dir/ model_name
    
    with open(data_path, 'rb') as f:
        model = pickle.load(f)
    
    return model
