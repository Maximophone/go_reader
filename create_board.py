from functions import create_board_position_file
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',type=str)
    parser.add_argument('--data-folder',type=str,default='./data')
    args = parser.parse_args()

    create_board_position_file(args.filename,folder=args.data_folder)
