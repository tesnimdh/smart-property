from pydantic import BaseModel



class ClientPreferences(BaseModel):
    budget_max: float | None = None
    location: str | None = None
    city: str | None = None
    governorate: str | None = None
    min_area: float | None = None
    min_pieces: float | None = None
    min_room: float | None = None
    min_bathroom: float | None = None
    state: str | None = None

    garage: bool = False
    garden: bool = False
    concierge: bool = False
    beach_view: bool = False
    mountain_view: bool = False
    pool: bool = False
    elevator: bool = False
    furnished: bool = False
    equipped_kitchen: bool = False
    central_heating: bool = False
    air_conditioning: bool = False




class HouseFeatures(BaseModel):
    area: float
    pieces: float
    room: float
    bathroom: float
    distance_to_capital: float

    location: str
    city: str
    governorate: str
    state: str

    garage: float = 0
    garden: float = 0
    concierge: float = 0
    beach_view: float = 0
    mountain_view: float = 0
    pool: float = 0
    elevator: float = 0
    furnished: float = 0
    equipped_kitchen: float = 0
    central_heating: float = 0
    air_conditioning: float = 0

    age_num: float | None = None
    latt: float | None = None
    long: float | None = None
