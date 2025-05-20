import logging
from datasets import load_from_disk, DatasetDict

from data_load import DataLoader


def load_opp115_data(base_path:str, save_path:str, include_metadata:bool) -> bool:
    obj = DataLoader()
    ret = obj.set_path(base_path)
    if not ret:
        logging.error("load_opp115_data: set_path(base_path) failed.")
        logging.error("load_opp115_data: Failed, returning False")
        return False

    ret = obj.load_policies()
    if not ret:
        logging.error("load_opp115_data: load_policies() failed.")
        logging.error("load_opp115_data: Failed, returning False")
        return False
    
    ret = obj.prepare_segments()
    if not ret:
        logging.error("load_opp115_data: prepare_segments() failed.")
        logging.error("load_opp115_data: Failed, returning False")
        return False

    hf_dataset = DataLoader.convert_to_hf_dataset(obj, include_metadata)
    ret = DataLoader.save_hf_dataset(hf_dataset, save_path)
    if not ret:
        logging.error("load_opp115_data: save_hf_dataset() failed.")
        logging.error("load_opp115_data: Failed, returning False")
        return False

    logging.info("load_opp115_data: Succeed, returning True")
    return True

def upload_opp115_dataset(dataset_path:str, repo_id:str, hf_token:str, commit_message:str, is_private:bool) -> None:
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, DatasetDict):
        return 
    
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=31)
    test_validation = train_testvalid["test"].train_test_split(test_size=0.5, seed=31)


    dataset_dict = DatasetDict({
        'train': train_testvalid["train"],
        'test': test_validation["train"],
        'validation': test_validation["test"]
    })
    
    
    print(f"Wrapped dataset in DatasetDict: {dataset_dict}")  
    dataset_dict.push_to_hub(
        repo_id = repo_id,
        token=hf_token,
        commit_message=commit_message,
        private=is_private,
    )
    print(f"Dataset pushed to Hub repository: {repo_id}")

def handle_load_opp115() -> None:
    base_path = input("BasePath:")
    save_path = input("SavePath:")
    include_metadata = input("IncludeMetadata(Y/N):").lower()

    load_opp115_data(base_path, save_path, include_metadata == "y")

def handle_upload_opp115() -> None:
    dataset_path = input("DatasetPath:")
    repo_id = input("RepoID:")
    token = input("Token:")
    commit_message = input("CommitMessage:")
    is_private = input("IsPrivate(Y/N):").lower()
    
    upload_opp115_dataset(dataset_path, repo_id, token, commit_message, is_private == "y")
