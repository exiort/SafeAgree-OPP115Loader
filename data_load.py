from __future__ import annotations
import json
import os
import pandas as pd
import logging
from bs4 import BeautifulSoup
from datasets import Dataset, Features, Value
from data_structs import Policy, Annotation, Segment



class DataLoader:
    annotations_dir_path:str
    sanitized_dir_path:str
    pretty_print_dir_path:str
    all_paths_exist:bool
    

    policy_list:list[Policy]
    policy_list_available = False
    
    segment_dict:dict[tuple[int, str], list[Segment]]
    segment_dict_available = False

    
    def __init__(self) -> None:
        logging.info("DataLoader: Object created.")
        self.all_paths_exist = False
        self.policy_list_available = False
        self.segment_dict_available = False
        
    def set_path(self, base_dir_path:str) -> bool:
        annotations_dir_path = os.path.join(base_dir_path, "annotations")
        sanitized_dir_path = os.path.join(base_dir_path, "sanitized_policies")
        pretty_print_path = os.path.join(base_dir_path, "pretty_print")

        all_paths_exist = True
        for path in [annotations_dir_path, sanitized_dir_path, pretty_print_path]:
            if not os.path.isdir(path):
                logging.error(f"DataLoader/set_path: Path={path} is not exist.")
                all_paths_exist = False
                break

        if not all_paths_exist:
            logging.error("DataLoader/set_path: Failed, returning False.")
            return False

        self.annotations_dir_path = annotations_dir_path
        self.sanitized_dir_path = sanitized_dir_path
        self.pretty_print_dir_path = pretty_print_path
        self.all_paths_exist = True
        logging.info("DataLoader/set_path: Succeed, returning True.")
        
        return True

    def load_policies(self) -> bool:
        if not self.all_paths_exist:
            logging.error("DataLoader/load_policies: all_paths_exist is False.")
            logging.error("DataLoader/load_policies: Failed, returning False.")
            return False

        sanitized_files = os.listdir(self.sanitized_dir_path)
        annotation_files = os.listdir(self.annotations_dir_path)
        pretty_print_files = os.listdir(self.pretty_print_dir_path)

        sanitized_pairs = []
        annotation_pairs = []
        pretty_print_pairs = []

        
        for san_file in sanitized_files:
            san_file_tokens = san_file.split("_")
            san_policy_id = int(san_file_tokens[0])
            san_base_filename = san_file_tokens[1][:-5]
            sanitized_pairs.append([san_policy_id, san_base_filename, san_file])

        for ann_file in annotation_files:
            ann_file_tokens = ann_file.split("_")
            ann_policy_id = int(ann_file_tokens[0])
            ann_base_filename = ann_file_tokens[1][:-4]
            annotation_pairs.append([ann_policy_id, ann_base_filename, ann_file])

        for pretty_file in pretty_print_files:
            pretty_base_filename = pretty_file[:-4]
            pretty_print_pairs.append([-1, pretty_base_filename, pretty_file])

        policy_list:list[Policy] = []

        is_success = True
        while True:
            if len(sanitized_pairs) == 0:
                logging.info("DataLoader/load_policies: All sanitized pairs are matched.")
                break
            
            match_san_pair = sanitized_pairs[0]
            logging.info(f"DataLoader/load_policies: Matching started for sanitized pair {match_san_pair}")
            
            match_ann_pair = None
            for ann_pair in annotation_pairs:
                if ann_pair[:-1] == match_san_pair[:-1]:
                    match_ann_pair = ann_pair
                    break

            if match_ann_pair == None:
                is_success = False
                logging.error("DataLoader/load_policies: Could not found matching annotaion pair.")
                break
            logging.info(f"DataLoader/load_policies: Found match annotation pair {match_ann_pair}")
            
            match_pretty_pair = None
            for pretty_pair in pretty_print_pairs:
                if pretty_pair[1] == match_san_pair[1]:
                    match_pretty_pair = pretty_pair
                    break

            if match_pretty_pair == None:
                is_success = False
                logging.error("DataLoader/load_policies: Could not found matching pretty print pair.")
                break
            logging.info(f"DataLoader/load_policies: Found match pretty print pair {match_pretty_pair}")
            
            policy = Policy(match_san_pair[0], match_san_pair[1])
            policy.sanitized_name = match_san_pair[2]
            policy.annotation_name = match_ann_pair[2]
            policy.pretty_print_name = match_pretty_pair[2]
            policy_list.append(policy)
            logging.info(f"DataLoader/load_policies: New policy added for {policy.policy_name}")
            
            sanitized_pairs.remove(match_san_pair)
            annotation_pairs.remove(match_ann_pair)
            pretty_print_pairs.remove(match_pretty_pair)

        if len(annotation_pairs) != 0:
            logging.warning(f"DataLoader/load_policies: There are {len(annotation_pairs)} orphans in annotation pairs.")

        if len(pretty_print_pairs) != 0:
            logging.warning(f"DataLoader/load_policies: There are {len(pretty_print_pairs)} orphans in pretty print pairs.")

        if not is_success:
            logging.error("DataLoader/load_policies: Failed, returning False ")
            return False
        
        self.policy_list = policy_list
        self.policy_list_available = True
        logging.info("DataLoader/load_policies: Succeed, policy list set as attribute, returning True")
    
        return True 

    def prepare_segments(self) -> bool:
        if not self.policy_list_available:
            logging.error("DataLoader/prepare_segments: polict_list_available is False.")
            logging.error("DataLoader/prepare_segment: Failed, returning False")
            return False

        segment_dict:dict[tuple[int, str], list[Segment]] = dict()
        for policy in self.policy_list:
            segment_list = self.__prepare_segment(policy)
            if segment_list is None:
                logging.warning(f"DataLoader/prepare_segments: None returned for {policy.policy_name}")
                continue
            
            logging.info(f"DataLoader/prepare_segments: {policy.policy_name} added.")
            segment_dict[(policy.policy_id, policy.policy_name)] = segment_list

        self.segment_dict = segment_dict
        self.segment_dict_available = True
        logging.info("DataLoader/prepare_segment: Succeed, returning True")
            
        return True
        
    def __prepare_segment(self, policy:Policy) -> list[Segment]|None:
        segment_list:list[Segment] = []
        
        with open(os.path.join(self.sanitized_dir_path, policy.sanitized_name)) as s_file:
            full_s_file = s_file.read()

        html_segments = full_s_file.split("|||")
        for segment_id, segment in enumerate(html_segments):
            soup = BeautifulSoup(segment, 'html.parser')
            raw_segment = soup.get_text(separator=" ", strip=True)
            logging.info(f"DataLoader/__prepare_segment: Segment with id {segment_id} is loader into memory.")
            segment_list.append(Segment(segment_id, raw_segment))


        ann_df = pd.read_csv(os.path.join(self.annotations_dir_path, policy.annotation_name), header=None)
        ann_df = ann_df.sort_values(by=0, ascending=True)
        logging.info(f"DataLoader/__prepare_segment: Annotations loaded, and ordered by AnnotationID.")

        pretty_df = pd.read_csv(os.path.join(self.pretty_print_dir_path, policy.pretty_print_name), header=None)
        pretty_df = pretty_df.sort_values(by=0, ascending=True)
        logging.info(f"DataLoader/__prepare_segment: PrettyPrint loaded, and ordered by AnnotationID.")
        
        if len(ann_df) != len(pretty_df):
            logging.error(f"DataLoader/__prepare_segment: {policy.policy_name} Count of annotations ({len(ann_df)}) and pretty pretty prints ({len(pretty_df)}) is not same!")
            logging.error("DataLoader/__prepare_segment: Failed, returning None")
            return None
            
        for i in range(len(ann_df)):
            annotation_id = int(ann_df.iloc[i, 0])
            segment_id = int(ann_df.iloc[i, 4])
            category_name = ann_df.iloc[i, 5]
            att_val_pair = ann_df.iloc[i, 6]

            ann_id = int(pretty_df.iloc[i, 0])
            seg_id = int(pretty_df.iloc[i, 1])
            pretty_print = pretty_df.iloc[i, 3]
            logging.info(f"DataLoader/__prepare_segment: {i}. row loaded.")
            
            if annotation_id != ann_id or segment_id != seg_id:
                logging.error(f"DataLoader/__prepare_segment: {policy.policy_name} Annotation ids ({annotation_id}-{ann_id}) or Segment ids ({segment_id}-{seg_id}) is not same!")
                logging.error("DataLoader/__prepare_segment: Failed, returning None")        
                return None

            annotation = Annotation(annotation_id, category_name, att_val_pair, pretty_print)

            if segment_list[segment_id].segment_id != segment_id:
                logging.error(f"DataLoader/__prepare_segment: segment_id of items in segment_list must be its index.({segment_id}-{segment_list[segment_id].segment_id})")      
                logging.error("DataLoader/__prepare_segment: Failed, returning None")        
                return None

            logging.info(f"DataLoader/__prepare_segment: New annotation {annotation} added.")
            segment_list[segment_id].add_annotation(annotation)

        logging.info("DataLoader/__prepare_segment: Succeed, returning segment_list")
        return segment_list
        

    @staticmethod
    def convert_to_hf_dataset(dataloader:DataLoader, include_metadata:bool) -> Dataset:
        hf_data = {
            "input_text": [],
            "target_json_string" : []
        }
        features = Features({
            "input_text": Value("string"),
            "target_json_string": Value("string")
        })
        
        if include_metadata:
            logging.info("DataLoader/convert_to_hf_dataset: include_metadata is True, Adding policy_id and segment_id")
            hf_data["policy_id"] = []
            hf_data["segment_id"] = []
            features.update({
                "policy_id": Value("int64"),
                "segment_id": Value("int64")
            })

        for (policy_id, policy_name), segment_list in dataloader.segment_dict.items():
            logging.info(f"DataLoader/convert_to_hf_dataset: {policy_id}-{policy_name}, adding.")

            if not segment_list:
                logging.warning(f"DataLoader/convert_to_hf_dataset: Segment list missing for {policy_name}")
                continue

            for segment in segment_list:
                hf_data["input_text"].append(segment.segment)

                if include_metadata:
                    hf_data["policy_id"].append(policy_id)
                    hf_data["segment_id"].append(segment.segment_id)

                target_dict = {}
                summary_parts = []

                for ann in segment.annotations:
                    summary_parts.append(ann.pretty_print.strip())

                target_dict["Summary"] = "|".join(summary_parts)

                categories_dict = {}
                for ann in segment.annotations:
                    categories_dict[ann.category_name] = {}

                    for attribute, attribute_value in ann.att_val_pair.items():
                        categories_dict[ann.category_name][attribute] = attribute_value["selectedText"].strip()

                target_dict["Categories"] = categories_dict
                target_json_string = json.dumps(target_dict, ensure_ascii=False, indent=2)
                hf_data["target_json_string"].append(target_json_string)

        hf_dataset = Dataset.from_dict(hf_data, features=features)
        return hf_dataset

    @staticmethod
    def save_hf_dataset(hf_dataset:Dataset, path:str) -> bool:
        try:
            hf_dataset.save_to_disk(path)
            logging.info(f"DataLoad/save_hf_dataset: Saved to path={path}")
            logging.info("DataLoad/save_hf_dataset: Succeed, returning True")
            return True
            
        except Exception as e:
            logging.info(f"DataLoad/save_hf_dataset: Throw exception={e}")
            logging.info("DataLoad/save_hf_dataset: Failed, returning False")
            return False
            
