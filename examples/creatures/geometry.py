from typing import NamedTuple


class Rectangle(NamedTuple):
    x: int
    y: int
    width: int
    height: int


def segment_intersection(segment1, segment2):
    return _segment_intersection(segment1[0][0], segment1[0][1], segment1[1][0], segment1[1][1],
                                 segment2[0][0], segment2[0][1], segment2[1][0], segment2[1][1])


def _segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """Точка пересечения двух отрезков A=(x1,y1)/(x2,y2) и  B=(x3,y3)/(x4,y4).
        Возвращаем точку или None"""
    denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)
    if denom == 0:
        return None
    ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom
    ub = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / denom
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)
    else:
        return None


def segment_intersections_with_rect(line, rect: Rectangle):
    """пересечение отрезка ((x1, y1), (x2, y2)) с прямоугольником (x, y, width, height) """
    rect_sides = [((rect.x, rect.y), (rect.x + rect.width, rect.y)),
                  ((rect.x + rect.width, rect.y), (rect.x + rect.width, rect.y + rect.height)),
                  ((rect.x + rect.width, rect.y + rect.height), (rect.x, rect.y + rect.height)),
                  ((rect.x, rect.y + rect.height), (rect.x, rect.y))]

    intersections = []
    for side in rect_sides:
        intersection = segment_intersection(line, side)
        if intersection is not None:
            intersections.append(intersection)
    return intersections


def distance_to_rect(line, rect: Rectangle):
    intersections = segment_intersections_with_rect(line, rect)
    if not intersections:
        return None
    else:
        return min(map(lambda o: distance(line[0], o), intersections))


def distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) ** (1/2)
