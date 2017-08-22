from functions import parse_pics
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pics-folder',type=str,default='./pics')
    parser.add_argument('--data-folder',type=str,default='./data')
    parser.add_argument('--data-filename',type=str,default='data.csv')
    args = parser.parse_args()

    parse_pics(args.pics_folder,args.data_filename,data_folder=args.data_folder)
