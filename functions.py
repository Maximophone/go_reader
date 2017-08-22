from PIL import Image
from PIL import ImageFilter
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cPickle as pickle

import datetime as dt

import pygame

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def get_position_click():
    mouse_not_clicked = True
    while mouse_not_clicked:
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                mouse_not_clicked = False
    return pos

def int_to_rgb(x):
    return (x&255,x>>16,(x>>8)&255)

int_to_rgb = np.vectorize(int_to_rgb)

def image_to_array(img):
    arr_r = pygame.surfarray.pixels_red(img)[:,:,None]
    arr_g = pygame.surfarray.pixels_green(img)[:,:,None]
    arr_b = pygame.surfarray.pixels_blue(img)[:,:,None]
    return np.concatenate([arr_r,arr_g,arr_b],axis=2)

def plot_im_and_classes(arr_im,classes,L,r=2):
    
    class_to_color = {
        0:[255,0,0],
        1:[0,255,0],
        2:[0,0,255]
    }

    arr_im = arr_im.copy()
    interval = L/19.
    
    margin = round(L/19/2)

    for i in range(19):
        for j in range(19):
            clazz = classes[i*19+j]
            arr_im[
                int(margin+interval*i-r):int(margin+interval*i+r),
                int(margin+interval*j-r):int(margin+interval*j+r),:] = class_to_color[clazz]

    plt.figure(figsize=(15,8))
    plt.imshow(arr_im)

    
def parse_pics(pics_folder,data_file,data_folder='./data_go'):
    if not os.path.isfile(data_folder+'/'+data_file):
        df = pd.DataFrame(columns=['filename','a','b','c','d','board_filename'])
    else:    
        df = pd.read_csv(data_folder+'/'+data_file)
        
    boards = {filename:load_board_position_file(filename,data_folder) for filename in os.listdir(data_folder) if filename[:5]=='board'}

    for filename in os.listdir(pics_folder):
        
        if filename[-4:]!='.jpg':
            continue
        if filename in df.filename.values:
            continue
        posA,posB,posC,posD,board_filename = get_corners_and_board(pics_folder+'/'+filename,boards)
        df = df.append({'filename':filename,'a':posA,'b':posB,'c':posC,'d':posD,'board_filename':board_filename},ignore_index=True)

    df.to_csv(data_folder+'/'+data_file,index=False)

def get_corners(arr_im,r=4,width=1000,height=None,landscape=True):
    go_img = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))

    w,h = go_img.get_width(), go_img.get_height()

    if height:
        width = int(height*w/float(h))
    else:
        height = int(width*h/float(w))

    if not landscape:
        print "rotating image..."
        go_img = pygame.transform.rotate(go_img,90)
    new_img = pygame.transform.scale(go_img,(width,height))

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    screen.blit(new_img, (0,0))
    pygame.display.flip()

    arr_im = image_to_array(new_img).transpose((1,0,2))
    
    posA = get_position_click()
    yA,xA = posA
    arr_im[xA-r:xA+r,yA-r:yA+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    posB = get_position_click()
    yB,xB = posB
    arr_im[xB-r:xB+r,yB-r:yB+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    posC = get_position_click()
    yC,xC = posC
    arr_im[xC-r:xC+r,yC-r:yC+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    posD = get_position_click()
    yD,xD = posD
    arr_im[xD-r:xD+r,yD-r:yD+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()

    pygame.quit()

    return posA, posB, posC, posD

def get_corners_and_board(img_path,boards,r=4,width=1000):    
    go_img = pygame.image.load(img_path)

    w,h = go_img.get_width(), go_img.get_height()

    max_dim = max(w,h)
    min_dim = min(w,h)

    ratio = min_dim/float(max_dim)

    portrait = w==max_dim

    height = int(ratio*width)

    if not portrait:
        go_img = pygame.transform.rotate(go_img,90)
    new_img = pygame.transform.scale(go_img,(width,height))

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    screen.blit(new_img, (0,0))
    
    board_keys = boards.keys()
    board_counter = 0
    board_key = board_keys[board_counter]
    black,white = boards[board_key]
    n_boards = len(board_keys)
    
    draw_board(screen,height,black,white,clear=False,fill_black=False)
    
    pygame.display.flip()
    
    enter_not_pressed = True
    while enter_not_pressed:
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    board_counter = (board_counter+1)%n_boards
                    board_key = board_keys[board_counter]
                    black,white = boards[board_key]
                    screen.blit(new_img, (0,0))
                    draw_board(screen,height,black,white,clear=False,fill_black=False)
                    pygame.display.flip()
                elif event.key == pygame.K_RETURN:
                    enter_not_pressed = False
                    screen.blit(new_img, (0,0))
                    pygame.display.flip()
                    
    arr_im = image_to_array(new_img).transpose((1,0,2))
    
    posA = get_position_click()
    yA,xA = posA
    arr_im[xA-r:xA+r,yA-r:yA+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    posB = get_position_click()
    yB,xB = posB
    arr_im[xB-r:xB+r,yB-r:yB+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    posC = get_position_click()
    yC,xC = posC
    arr_im[xC-r:xC+r,yC-r:yC+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    posD = get_position_click()
    yD,xD = posD
    arr_im[xD-r:xD+r,yD-r:yD+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    pygame.quit()
    
    return posA,posB,posC,posD,board_key

def get_features_and_targets(pics_folder,data_file,data_folder='./data_go',width=1000):
    df = pd.read_csv(data_folder+'/'+data_file)
    boards = {filename:load_board_position_file(filename,data_folder) for filename in os.listdir(data_folder) if filename[:5]=='board'}
    
    all_features = []
    all_targets = []
    all_images = []
    
    for _,row in df.iterrows():
        img_path = pics_folder+'/'+row['filename']
        go_img = pygame.image.load(img_path)
        
        black,white = boards[row['board_filename']]
        
        (posA, posB, posC, posD) = (map(int,row['a'][1:-1].split(',')), 
            map(int,row['b'][1:-1].split(',')), 
            map(int,row['c'][1:-1].split(',')), 
            map(int,row['d'][1:-1].split(',')))

        w,h = go_img.get_width(), go_img.get_height()

        max_dim = max(w,h)
        min_dim = min(w,h)

        ratio = min_dim/float(max_dim)

        portrait = w==max_dim

        height = int(ratio*width)

        if not portrait:
            go_img = pygame.transform.rotate(go_img,90)
        new_img = pygame.transform.scale(go_img,(width,height))
        
        pil_im = Image.fromarray(np.uint8(image_to_array(new_img)))
        
        L = min(width,height)
        margin = round(L/19/2)

        coeffs = find_coeffs(
                [(margin, margin), (L-margin, margin), (L-margin, L-margin), (margin, L-margin)],
                [posA[::-1], posB[::-1], posC[::-1], posD[::-1]])

        img_t = pil_im.transform((width, height), Image.PERSPECTIVE, coeffs,
                Image.BICUBIC).filter(ImageFilter.RankFilter(3, 0))

        classes = np.zeros(361)

        for i,j in white:
            classes[j*19+i] = 1
        for i,j in black:
            classes[j*19+i] = 2
        arr_im = np.array(img_t)
        plot_im_and_classes(arr_im,classes,arr_im.shape[0])
        
        all_images.append(arr_im)
        
        interval = L/19.

        r_focus = int(interval/2.)

        features = []

        for i in range(19):
            for j in range(19):
                square_focus = arr_im[
                    int(margin+interval*i-r_focus):int(margin+interval*i+r_focus),
                    int(margin+interval*j-r_focus):int(margin+interval*j+r_focus),:]
                features.append(square_focus.copy())

        features = np.array(features)
        
        all_features.append(features)
        all_targets.append(classes)
    
    features = np.concatenate(all_features)
    targets = np.concatenate(all_targets)
    images = np.array(all_images)
    
    return features,targets,images
    
def get_features(arr_im,pos_corners,width=1000,height=None,landscape=True):
    
    #go_img = pygame.pixelcopy.make_surface(arr_im)
    go_img = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))

    posA, posB, posC, posD = pos_corners

    w,h = go_img.get_width(), go_img.get_height()

    if height:
        width = int(height*w/float(h))
    else:
        height = int(width*h/float(w))

    if not landscape:
        print "rotating image..."
        go_img = pygame.transform.rotate(go_img,90)
    new_img = pygame.transform.scale(go_img,(width,height))
    
    pil_im = Image.fromarray(np.uint8(image_to_array(new_img)))

    L = min(width,height)
    margin = round(L/19/2)

    coeffs = find_coeffs(
        [(margin, margin), (L-margin, margin), (L-margin, L-margin), (margin, L-margin)],
                [posA[::-1], posB[::-1], posC[::-1], posD[::-1]])

    img_t = pil_im.transform((width, height), Image.PERSPECTIVE, coeffs,
                             Image.BICUBIC).filter(ImageFilter.RankFilter(3, 0))

    arr_im = np.array(img_t)

    interval = L/19.

    r_focus = int(interval/2.)

    features = []

    for i in range(19):
        for j in range(19):
            square_focus = arr_im[
                int(margin+interval*i-r_focus):int(margin+interval*i+r_focus),
                int(margin+interval*j-r_focus):int(margin+interval*j+r_focus),:]
            features.append(square_focus.copy())

    features = np.array(features)

    return features, arr_im

def get_features_OLD(img_path,r=4,width=1000):
    go_img = carImg = pygame.image.load(img_path)

    w,h = go_img.get_width(), go_img.get_height()

    max_dim = max(w,h)
    min_dim = min(w,h)

    ratio = min_dim/float(max_dim)

    portrait = w==max_dim

    height = int(ratio*width)

    if not portrait:
        go_img = pygame.transform.rotate(go_img,90)
    new_img = pygame.transform.scale(go_img,(width,height))

    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    screen.blit(new_img, (0,0))
    pygame.display.flip()

    arr_im = image_to_array(new_img).transpose((1,0,2))
    
    posA = get_position_click()
    yA,xA = posA
    arr_im[xA-r:xA+r,yA-r:yA+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    posB = get_position_click()
    yB,xB = posB
    arr_im[xB-r:xB+r,yB-r:yB+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    posC = get_position_click()
    yC,xC = posC
    arr_im[xC-r:xC+r,yC-r:yC+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    posD = get_position_click()
    yD,xD = posD
    arr_im[xD-r:xD+r,yD-r:yD+r,:] = [0,0,255]
    
    new_surf = pygame.pixelcopy.make_surface(np.swapaxes(arr_im, 0, 1))
    screen.blit(new_surf, (0,0))
    pygame.display.flip()
    
    pygame.quit()

#     plt.figure(figsize=(15,8))
#     plt.imshow(arr_im)

    pil_im = Image.fromarray(np.uint8(image_to_array(new_img)))

    L = min(width,height)
    margin = round(L/19/2)

    coeffs = find_coeffs(
            [(margin, margin), (L-margin, margin), (L-margin, L-margin), (margin, L-margin)],
            [posA[::-1], posB[::-1], posC[::-1], posD[::-1]])

    img_t = pil_im.transform((width, height), Image.PERSPECTIVE, coeffs,
            Image.BICUBIC).filter(ImageFilter.RankFilter(3, 0))

    plt.figure(figsize=(15,8))
    plt.imshow(img_t)

    arr_im_t = np.array(img_t)

    interval = L/19.

    r_focus = int(interval/2.)

    features = []

    for i in range(19):
        for j in range(19):
            square_focus = arr_im_t[
                int(margin+interval*i-r_focus):int(margin+interval*i+r_focus),
                int(margin+interval*j-r_focus):int(margin+interval*j+r_focus),:]
            color = square_focus.reshape((-1,3)).mean(axis=0)
            std = square_focus.reshape((-1,3)).mean(axis=1).std(axis=0)
            # min_val = square_focus.min()

            features.append(square_focus.copy())
    #         features.append(np.concatenate([color,[std]]))

#             arr_im_t[
#                 int(margin+interval*i-r_focus):int(margin+interval*i+r_focus),
#                 int(margin+interval*j-r_focus):int(margin+interval*j+r_focus),:] = color

    features = np.array(features)
#     plt.figure(figsize=(15,8))
#     plt.imshow(arr_im_t)
    
    return features,arr_im_t

def get_board_surface(L_board,black,white,clear=True,fill_black=True,color=(0,0,255)):

    r_stone = L_board/20/2
    interval = L_board/20

    surface = pygame.Surface([L_board, L_board])

    for i in range(1,20):
        x0 = i*interval
        y0 = interval
        
        x1 = i*interval
        y1 = L_board - interval
        
        pygame.draw.line(surface, color, (x0,y0), (x1,y1))
        pygame.draw.line(surface, color, (y0,x0), (y1,x1))
    
    #hoshi
    pygame.draw.circle(surface,color,(4*interval,4*interval),6)
    pygame.draw.circle(surface,color,(16*interval,4*interval),6)
    pygame.draw.circle(surface,color,(4*interval,16*interval),6)
    pygame.draw.circle(surface,color,(16*interval,16*interval),6)
    
    #tengen
    pygame.draw.circle(surface,color,(10*interval,10*interval),6)
    
    #sides
    pygame.draw.circle(surface,color,(10*interval,4*interval),6)
    pygame.draw.circle(surface,color,(10*interval,16*interval),6)
    pygame.draw.circle(surface,color,(4*interval,10*interval),6)
    pygame.draw.circle(surface,color,(16*interval,10*interval),6)

    for x,y in black:
        pygame.draw.circle(surface,color,((x+1)*interval,(y+1)*interval),r_stone,2)
        if fill_black:
            pygame.draw.circle(surface,(0,0,0),((x+1)*interval,(y+1)*interval),r_stone-1)
    for x,y in white:
        pygame.draw.circle(surface,color,((x+1)*interval,(y+1)*interval),r_stone)
    
    return surface

def draw_board(screen,L_screen,black,white,L_board=500,clear=True,fill_black=True):
    
    margin = (L_screen - L_board)/2
    
    r_stone = L_board/18/2
    
    color = (0, 0, 255)
    
    if clear:
        screen.fill((0,0,0))
    
    for i in range(19):
        x0 = margin+i*L_board/18.
        y0 = margin
        
        x1 = margin+i*L_board/18.
        y1 = L_screen-margin
        
        
        pygame.draw.line(screen, color, (x0, y0), (x1, y1))
        pygame.draw.line(screen, color, (y0,x0), (y1, x1))
    
    #hoshi
    pygame.draw.circle(screen,color,(margin+3*L_board/18,margin+3*L_board/18),5)
    pygame.draw.circle(screen,color,(margin+15*L_board/18,margin+3*L_board/18),5)
    pygame.draw.circle(screen,color,(margin+3*L_board/18,margin+15*L_board/18),5)
    pygame.draw.circle(screen,color,(margin+15*L_board/18,margin+15*L_board/18),5)
    
    #tengen
    pygame.draw.circle(screen,color,(margin+9*L_board/18,margin+9*L_board/18),5)
    
    pygame.draw.circle(screen,color,(margin+9*L_board/18,margin+3*L_board/18),5)
    pygame.draw.circle(screen,color,(margin+9*L_board/18,margin+15*L_board/18),5)
    pygame.draw.circle(screen,color,(margin+3*L_board/18,margin+9*L_board/18),5)
    pygame.draw.circle(screen,color,(margin+15*L_board/18,margin+9*L_board/18),5)
    
    for x,y in black:
        pygame.draw.circle(screen,color,(margin+x*L_board/18,margin+y*L_board/18),r_stone,2)
        if fill_black:
            pygame.draw.circle(screen,(0,0,0),(margin+x*L_board/18,margin+y*L_board/18),r_stone-1)
    for x,y in white:
        pygame.draw.circle(screen,color,(margin+x*L_board/18,margin+y*L_board/18),r_stone)
    
    pygame.display.flip()

def create_board(black=set(),white=set()):
    L_screen = 600
    L_board = 500
    margin = (L_screen - L_board)/2
    
#     r_stone = L_board/18/2
    
#     color = (0, 0, 255)
    
    pygame.init()
    screen = pygame.display.set_mode((L_screen, L_screen))
    
    draw_board(screen,L_screen,black,white,L_board)
    
#     for i in range(19):
#         x0 = margin+i*L_board/18.
#         y0 = margin
        
#         x1 = margin+i*L_board/18.
#         y1 = L_screen-margin
        
        
#         pygame.draw.line(screen, color, (x0, y0), (x1, y1))
#         pygame.draw.line(screen, color, (y0,x0), (y1, x1))
    
#     pygame.draw.circle(screen,color,(margin+3*L_board/18,margin+3*L_board/18),5)
#     pygame.draw.circle(screen,color,(margin+15*L_board/18,margin+3*L_board/18),5)
#     pygame.draw.circle(screen,color,(margin+3*L_board/18,margin+15*L_board/18),5)
#     pygame.draw.circle(screen,color,(margin+15*L_board/18,margin+15*L_board/18),5)
#     pygame.draw.circle(screen,color,(margin+9*L_board/18,margin+9*L_board/18),5)
    
#     for x,y in black:
#         pygame.draw.circle(screen,color,(margin+x*L_board/18,margin+y*L_board/18),r_stone,2)
#         pygame.draw.circle(screen,(0,0,0),(margin+x*L_board/18,margin+y*L_board/18),r_stone-1)
#     for x,y in white:
#         pygame.draw.circle(screen,color,(margin+x*L_board/18,margin+y*L_board/18),r_stone)
    
#     pygame.display.flip()
    
    running = True
    while running:
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.QUIT:
                running=False
            if event.type == pygame.MOUSEBUTTONUP:
                pos = event.pos
                nearest_point = (int((pos[0]-margin)*18./L_board+0.5),int((pos[1]-margin)*18./L_board+0.5))
                if event.button == 1:
                    if nearest_point in black:
                        black.remove(nearest_point)
                    else:
                        black.add(nearest_point)
                        if nearest_point in white:
                            white.remove(nearest_point)
#                     pygame.draw.circle(screen,color,(margin+nearest_point[0]*L_board/18,margin+nearest_point[1]*L_board/18),r_stone,2)
#                     pygame.draw.circle(screen,(0,0,0),(margin+nearest_point[0]*L_board/18,margin+nearest_point[1]*L_board/18),r_stone-1)
                elif event.button == 3:
                    if nearest_point in white:
                        white.remove(nearest_point)
                    else:
                        white.add(nearest_point)
                        if nearest_point in black:
                            black.remove(nearest_point)
#                     pygame.draw.circle(screen,color,(margin+nearest_point[0]*L_board/18,margin+nearest_point[1]*L_board/18),r_stone)
                draw_board(screen,L_screen,black,white,L_board)
#                 pygame.display.flip()
    pygame.quit()
    
    return black,white

def create_board_position_file(filename,black=None,white=None,folder='./data_go'):
    if not black:
        black = set()
    if not white:
        white=set()
    black,white = create_board(black,white)
    pickle.dump((black,white),open(folder+'/'+filename,'w'))

def load_board_position_file(filename,folder='./data_go'):
    black,white = pickle.load(open(folder+'/'+filename,'r'))
    return black,white
