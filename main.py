import sys
from scripts import handle_load_opp115, handle_upload_opp115


def default_err():
    print("Correct Usage: main.py [Function]")
    print("Avaliable Function(s): 'load_opp115', 'upload_opp115'")
        

def main():
    if(len(sys.argv)) != 2:
        default_err()
        return

    if sys.argv[1] == "load_opp115":
        handle_load_opp115()
    elif sys.argv[1] == "upload_opp115":
        handle_upload_opp115()
    else:
        default_err()
    
    
if __name__ == "__main__":
    main()
