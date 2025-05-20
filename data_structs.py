import json
import logging



class Policy:
    policy_id:int
    policy_name:str
    
    annotation_name:str
    sanitized_name:str
    pretty_print_name:str


    def __init__(self, policy_id:int, policy_name:str) -> None:
        logging.info("Policy: Object created.")
        self.policy_id = policy_id
        self.policy_name = policy_name
        self.annotation_name = ""
        self.sanitized_name = ""
        self.pretty_print_name = ""
        
    def is_all_set(self) -> bool:
        return not (self.annotation_name == "" or self.sanitized_name == "" or self.pretty_print_name == "")

    
class Annotation:
    annotation_id:int

    category_name:str
    att_val_pair:dict
    pretty_print:str

    def __init__(self, annotation_id:int, category_name:str, att_val_pair_str:str, pretty_print:str) -> None:
        logging.info("Annotation: Object created.")
        self.annotation_id = annotation_id
        self.category_name = category_name
        self.att_val_pair = self.__prepare_att_val_pair(att_val_pair_str)
        self.pretty_print = pretty_print

    def __repr__(self) -> str:
        return f"ID:{self.annotation_id}, category:{self.category_name}, pretty:{self.pretty_print}"
    
    def __prepare_att_val_pair(self, att_val_pair_str:str) -> dict:
        att_val_pair = json.loads(att_val_pair_str)

        att_to_del = []
        
        for attribute, attribute_value in att_val_pair.items():
            if attribute_value["endIndexInSegment"] == -1 or attribute_value["startIndexInSegment"] == -1:
                logging.info(f"Annotation/__prepare_att_val_pair: Attribute={attribute} has invalid segment index, discarding.")
                att_to_del.append(attribute)

        for att in att_to_del:
            del att_val_pair[att]
        
        return att_val_pair

    
class Segment:
    segment_id:int

    segment:str
    annotations:list[Annotation]

    def __init__(self, segment_id:int, segment:str) -> None:
        logging.info("Segment: Object created.")
        self.segment_id = segment_id
        self.segment = segment
        self.annotations = []

    def __repr__(self) -> str:
        if len(self.segment) < 50:
            return f"{self.segment_id}->[{len(self.segment)}]-{self.segment}"
        else:
            return f"{self.segment_id}->[{len(self.segment)}]-{self.segment[:50]}..."

    def add_annotation(self, new_ann:Annotation) -> None:
        is_new = True
        for old_ann in self.annotations:
            if old_ann.category_name == new_ann.category_name:
                logging.info(f"Segment/add_annotaion: Annotation has same category name {new_ann.category_name}. Trying to extend previous one.")
                self.__extent_annotation(old_ann.att_val_pair, new_ann.att_val_pair)
                is_new = False
                break
            
        if not is_new:
            logging.info(f"Segment/add_annotation: Annotation with category name {new_ann.category_name} is not new. Returning.")
            return

        logging.info(f"Segment/add_annotation: Annotation with category name {new_ann.category_name} is new. Adding.")
        self.annotations.append(new_ann)

    def __extent_annotation(self, old_att_val_pair:dict, new_att_val_pair) -> None:
        for new_attribute, new_attribute_value in new_att_val_pair.items():
            is_new = True
            for old_attribute_value in old_att_val_pair.values():
                if new_attribute_value["selectedText"] == old_attribute_value["selectedText"]:
                    logging.info(f"Segment/__extend_annotation: Attribute={new_attribute} has same selectedText. Discarding.")
                    is_new = False

            if not is_new:
                continue

            if new_attribute not in old_att_val_pair:
                logging.info(f"Segment/__extend_annotation: Attribute={new_attribute} is new. Adding.")
                old_att_val_pair[new_attribute] = new_attribute_value
                continue

            else:
                logging.info(f"Segment/__extend_annotation: Attribute={new_attribute} is not new. Selecting longer selectedText")
                old_text_len = len(old_att_val_pair[new_attribute]["selectedText"])
                new_text_len = len(new_attribute_value["selectedText"])

                if old_text_len > new_text_len:
                    logging.info("Segment/__extend_annotation: Old attribute has longer selectedText. No changing")
                    continue

                logging.info("Segment/__extent_annotation: New attribute has longer selectedText. Changing.")
                old_att_val_pair[new_attribute] = new_attribute_value
                        
