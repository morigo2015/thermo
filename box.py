import cv2 as cv
import colors


class Box:

    def __init__(self, startX=None, startY=None, endX=None, endY=None,
                 sides=None,  # sides-order (top,right,bottom,left)
                 corners=None,  # corners-order  (startX,startY,endX,endY)
                 box=None):
        """
        create new box based on corners coordinates or corner-order string
        :param coordinate_str: corner-order string
        """
        # self.is_empty = False
        if startX is not None and startY is not None and endX is not None and endY is not None:
            self.startX = startX
            self.startY = startY
            self.endX = endX
            self.endY = endY
        elif sides is not None:  # sides-order (top,right,bottom,left)
            self.startX, self.startY, self.endX, self.endY = Box.sides_2_corners(sides_tuple=sides)
        elif corners is not None:  # corners-order  (startX,startY,endX,endY)
            self.startX, self.startY, self.endX, self.endY = corners
        elif box is not None:
            self.startX = box.startX
            self.startY = box.startY
            self.endX = box.endX
            self.endY = box.endY
            # self.is_empty = box.is_empty
        else:
            print('!!!!! error while initializing Box !!!!!! ')

    def __repr__(self):
        return f'box(({self.startX},{self.startY}),({self.endX},{self.endY}))'

    def _check_limits(self, frame):
        def _limit(val, max_val):
            if val < 0: return 0
            if val > max_val - 1:
                return max_val - 1
            else:
                return val

        # check if coordinates are inside a frame limit
        self.startX = _limit(self.startX, frame.shape[1])
        self.startY = _limit(self.startY, frame.shape[0])
        self.endX = _limit(self.endX, frame.shape[1])
        self.endY = _limit(self.endY, frame.shape[0])

    def draw(self, frame, color=None, label=None, thickness=1):
        self._check_limits(frame)
        if color is None:
            color = colors.BGR_GREEN
        cv.rectangle(frame, (self.startX, self.startY), (self.endX, self.endY), color, thickness)  # 1 <--> 2
        if label is not None:
            y = self.startY - 15 if self.startY - 15 > 15 else self.startY + 15
            cv.putText(frame, label, (self.startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    def draw2(self, frame, color=None, label=None, thickness=1):
        if self.is_empty():
            return
        else:
            self.draw(frame, color, label, thickness)

    def is_empty(self):
        return True if self.startY >= self.endY or self.startX >= self.endX else False

    def center(self):
        x_center = self.startX + int((self.endX - self.startX) / 2)
        y_center = self.startY + int((self.endY - self.startY) / 2)
        return x_center, y_center

    def draw_center(self, frame, color=None, size=None):
        self._check_limits(frame)
        if color is None:
            color = colors.BGR_GREEN
        if size is None:
            size = 5
        x, y = self.center()
        cv.circle(frame, (x, y), radius=size, color=color, thickness=-1)

    def sides(self):
        # return top,right,bottom,left
        return self.startY, self.endX, self.endY, self.startX

    def corners(self):
        return self.startX, self.startY, self.endX, self.endY

    @staticmethod
    def width(corners_tuple=None, sides_tuple=None):
        if sides_tuple is not None:
            top, right, bottom, left = sides_tuple
            return right - left
        return

    @staticmethod
    def height(corners_tuple=None, sides_tuple=None):
        if sides_tuple is not None:
            top, right, bottom, left = sides_tuple
            return bottom - top
        return

    def intersect(self, box):
        isect = Box(startX=max(self.startX, box.startX), startY=max(self.startY, box.startY),
                    endX=min(self.endX, box.endX), endY=min(self.endY, box.endY))
        return isect

    def union(self, box):
        if self.is_empty: return box
        if box.is_empty: return self
        u = Box(startX=min(self.startX, box.startX), startY=min(self.startY, box.startY),
                endX=max(self.endX, box.endX), endY=max(self.endY, box.endY))
        return u

    def area(self):
        if self.is_empty():
            return 0
        else:
            return (self.endX - self.startX) * (self.endY - self.startY)

    def iou(self, box):
        i = self.intersect(box)
        u = self.union(box)
        if u.area() == 0:
            print(f'!!!!!! union({self},{box}=0 !!!! ????')
        return i.area() / u.area()

    def __eq__(self, other):  # equal if all corners|sides are the same
        if isinstance(other, Box):
            return self.startX == other.startX and self.startY == other.startY \
                   and self.endX == other.endX and self.endY == other.endY
        return NotImplemented

    def box_2_str(self):
        """
        coordinates of existing box --> string for filename (in corners-order)
        return 'corners'-order(sXYeXY), not 'sides'-order(trbl) !!
        """
        return '{:04d}{:04d}{:04d}{:04d}'.format(self.startX, self.startY, self.endX, self.endY)

    @staticmethod
    def str_2_coord(str):
        """
        string for filename -->  box coordinates
        """
        startX = int(str[0:4])
        startY = int(str[4:8])
        endX = int(str[8:12])
        endY = int(str[12:16])
        return startX, startY, endX, endY

    @staticmethod
    def coord_2_str(startX, startY, endX, endY):
        """
        box coordinates  -->  string for filename
        """
        str = '{:04d}{:04d}{:04d}{:04d}'.format(startX, startY, endX, endY)
        return str

    # conversion:    sides-order (top,right,bottom,left)   <--->   corners-order  (startX,startY,endX,endY)

    @staticmethod
    def corners_2_sides(startX=None, startY=None, endX=None, endY=None
                        , corners_tuple=None):
        if corners_tuple is not None:
            (startX, startY, endX, endY) = corners_tuple
        return startY, endX, endY, startX

    @staticmethod
    def sides_2_corners(top=None, right=None, bottom=None, left=None
                        , sides_tuple=None):
        if sides_tuple is not None:
            top, right, bottom, left = sides_tuple
        return left, top, right, bottom

    # string conversion: sides-order  <---> corners-order
    # each value - 4 digits

    @staticmethod
    def corners_2_sides_str(str):
        (startX, startY, endX, endY) = Box.str_2_coord(str)
        (left, top, right, bottom) = Box.corners_2_sides(startX, startY, endX, endY)
        return Box.coord_2_str(left, top, right, bottom)

    @staticmethod
    def sides_2_corners_str(str):
        (left, top, right, bottom) = Box.str_2_coord(str)
        (startX, startY, endX, endY) = Box.sides_2_corners(left, top, right, bottom)
        return Box.coord_2_str(startX, startY, endX, endY)


if __name__ == '__main__':
    b1 = Box(corners=(1, 1, 5, 5))
    b2 = Box(corners=(3, 3, 6, 6))
    print(b1.center(), b1.area())
    print(b1.union(b2))
    print(b1.iou(b1))
    print('iou=', b1.iou(b2))
    b3 = b1.union(b2)
    b4 = b1.intersect(b2)
    print(b1, b2, b3, b4)
