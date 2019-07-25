# colors
"""
names for basic colors; 
COLORS[] - list of color codes, ordered to not merge so that neighboring colors are not merging
COLOR_NAMES[] - list of color names;
"""
BGR_BLACK = (0,0,0)
BGR_RED = (0,0,255)
BGR_GREEN = (0,255,0)
BGR_BLUE = (255,0,0)
BGR_YELLOW = (0,255,255)
BGR_NAVAJO = (173,222,255)
BGR_DARK = (79,79,47)
BGR_NAVY = (128,0,0)
BGR_SKY = (235,206,135)
BGR_SEA = (87,139,46)
BGR_FOREST = (34,139,34)
BGR_KHAKI = (107,183,189)
BGR_ROSY = (143,143,188)
BGR_INDIAN = (92,92,205)
BGR_BROWN = (42,42,165)
BGR_PINK = (180,105,255)
BGR_CYAN = (255,255,0)
BGR_MAGENTA = (255,0,255)
BGR_ORANGE = (0,154,238)
BGR_CHOCOLATE = (19,69,139)
BGR_PLUM = (238,174,238)
BGR_GRAY = (255/2, 255/2, 255/2)
BGR_WHITE = (255, 255, 255)
COLOR_NAMES = [ v for v in globals() if v.startswith('BGR_')]
COLORS=[ eval(v) for v in COLOR_NAMES]

if __name__ == '__main__':

    import cv2
    import numpy as np

    print(f'total: {len(COLORS)} colors')
    img = np.zeros((720, 1280, 3), np.uint8)
    sx=10
    sy=5
    for n,c in zip(COLOR_NAMES,COLORS):
        cv2.putText(img, f'{n}:{c}', (sx,sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        cv2.circle(img,(sx+300,sy-5),10,c,-1)
        sy += 25
    cv2.imshow('test',img)
    while cv2.waitKeyEx() != ord('q'):
        pass
