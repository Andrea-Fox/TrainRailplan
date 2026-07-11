from replay import World
from env import Env
w = World("data/cz_rail_20260714"); e = Env(w, "data/cz_rail_20260714")

print(e.find_station("praha"))
print(e.departures("S9402.P1", "09:00"))   # Kolín
print(e.find_station("caslav"))