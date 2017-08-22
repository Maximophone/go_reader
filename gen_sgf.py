import pygame
import argparse
import cv2
import numpy as np
import datetime as dt
import time

from functions import get_corners, get_board_surface, get_features
from models import conv_large, conv_small

models_dict = {
    "conv_large":conv_large,
    "conv_small":conv_small}

pygame.font.init()
myfont = pygame.font.SysFont('Arial', 20)

class DiffError(Exception):
    pass

def diff_move(state0, state1):
    black0, white0 = state0
    black1, white1 = state1

    new_black = black1.difference(black0)
    new_white = white1.difference(white0)

    if len(new_black)+len(new_white)>1:
        raise DiffError("More than one stone was added")
    
    if len(new_black)==0 and len(new_white)==0:
        raise DiffError("No new stone were added")

    if len(new_black)==1:
        return 'b', new_black.pop()
    else:
        return 'w', new_white.pop()

def init_vc(input_id):
    vc = cv2.VideoCapture(input_id)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        raise Exception("Unable to open Video Capture {}".format(input_id))
    return vc

def read_vc(vc):
    rval, frame = vc.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    return np.swapaxes(frame, 0, 1)

def array_to_surface(arr):
    return pygame.pixelcopy.make_surface(arr)

def gen_sgf_string(moves,name_black,name_white):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    game_string = ['']
    for col, (i_x, i_y) in moves:
        game_string.append(col.upper()+'['+letters[i_x]+letters[i_y]+']')
    game_string = ';'.join(game_string)
    final_string = """(;GM[1]SZ[19]KM[7.0]RU[Japanese]
PB[{}]
PW[{}]
{})""".format(name_black,name_white,game_string)
    return final_string

def calibration(vc):
    # First adjust the camera
    camera_ready = False
    screen = pygame.display.set_mode(read_vc(vc).shape[:2])
    textsurface = myfont.render('Press a key when camera is ready', False, (0, 0, 0))
    while not camera_ready:
        frame = read_vc(vc)
        screen.blit(array_to_surface(frame),(0,0))
        screen.blit(textsurface,(0,0))
        pygame.display.flip()

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                camera_ready = True
    
    # Then get the corners positions
    pos_corners = get_corners(np.swapaxes(frame,0,1),height=562)

    return pos_corners

def main(args):

    print "Loading model"
    model = models_dict[args.model_name].build()
    model.load_weights(args.model_weights)

    vc = init_vc(args.vc_input)
    pygame.init()

    pos_corners = calibration(vc)
    
    # Start the diff process
    prev_black = set()
    prev_white = set()
    init = False
    moves_list = []

    game_over = False

    while not game_over:
        frame = np.swapaxes(read_vc(vc),0,1)
        t0 = time.time()
        features, new_image = get_features(frame, pos_corners, height=562)
        t1 = time.time()
        if not init:
            pygame.init()
            width, height = new_image.shape[:2]
            screen = pygame.display.set_mode((args.screen_res,args.screen_res))
            init = True
        
        classes = model.predict(features).argmax(axis=1)
        
        t2 = time.time()
        
        black = set([(i%19,i/19) for i,cl in enumerate(classes) if cl==2])
        white = set([(i%19,i/19) for i,cl in enumerate(classes) if cl==1])

        # get diff
        diff_error = False
        try:
            col,move = diff_move((prev_black,prev_white),(black,white))
        except DiffError:
            diff_error = True
            col, move = None, None

        screen.fill((0,0,0))

        global_surf = pygame.Surface((2*width,2*width))

        # Display camera
        surface = pygame.pixelcopy.make_surface(np.swapaxes(new_image, 0, 1))
        global_surf.blit(surface,(width,width))

        # Display previous
        surface = get_board_surface(540, prev_black, prev_white,clear=False)
        global_surf.blit(surface,(width,0))

        # Display computer vision
        surface = get_board_surface(540, black, white,clear=False,color=(255,0,0) if diff_error else (0,255,0))
        global_surf.blit(surface,(0,0))

        pygame.transform.smoothscale(global_surf, (args.screen_res,args.screen_res), screen)
        pygame.display.flip()

        t3 = time.time()
        print "Times: {:.2f}, {:.2f}, {:.2f}".format(t1-t0,t2-t1,t3-t2)

        recalibrate = False
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN and event.key==pygame.K_ESCAPE:
                game_over = True
            elif event.type == pygame.KEYDOWN and event.key==pygame.K_c:
                recalibrate = True
            elif event.type == pygame.KEYDOWN and not diff_error:
                moves_list.append((col,move))
                prev_black, prev_white = black, white
        if recalibrate:
            pygame.quit()
            pygame.init()
            pos_corners = calibration(vc)
            init = False
            

    sgf_string = gen_sgf_string(moves_list, args.name_black, args.name_white)
    with open(args.folder+'/'+args.filename,'w') as f:
        f.write(sgf_string)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vc-input",type=int,default=1)
    parser.add_argument("--model-weights",type=str,default="./saved_models/model_best_5.h5")
    parser.add_argument("--filename",type=str,default="savegame_{}.sgf".format(dt.datetime.now().strftime("%Y%m%d_%H:%M")))
    parser.add_argument("--folder",type=str,default="/home/max/sgf")
    parser.add_argument("--name-black",type=str,default="Black")
    parser.add_argument("--name-white",type=str,default="White")
    parser.add_argument("--screen-res",type=int,default=1000)
    parser.add_argument("--model-name",type=str,default="conv_small")
    args = parser.parse_args()
    main(args)
