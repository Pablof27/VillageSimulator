from enum import Enum


class Resource(Enum):
    FOOD = "food"
    WOOD = "wood"


def other(resource: Resource) -> Resource:
    return Resource.FOOD if resource == Resource.WOOD else Resource.WOOD
