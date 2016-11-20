#AIzaSyADuv1109TAE_eDK-g0abWMkYrIYjJahmw
#AIzaSyCL9rEHZn_GZvIWwxDAnPajDKS1Uz1JWH8

import psycopg2 as dbapi
import googlemaps
from datetime import datetime
import sys

key4 = 'AIzaSyAJPpV2PxwzS10swy85Edbc8NzUkfbacdY'
key3 = 'AIzaSyCL9rEHZn_GZvIWwxDAnPajDKS1Uz1JWH8'
key2 = 'AIzaSyBqIfaDT91opEDK7962OCh6iKA5RIRA5Kc'
key1 = 'AIzaSyADuv1109TAE_eDK-g0abWMkYrIYjJahmw'


gmaps1 = googlemaps.Client(key=key1, queries_per_second=20)
gmaps2 = googlemaps.Client(key=key2, queries_per_second=20)
gmaps3 = googlemaps.Client(key=key3, queries_per_second=20)
gmaps4 = googlemaps.Client(key=key4, queries_per_second=20)

theClients = [gmaps1, gmaps2, gmaps3, gmaps4]

# Geocoding an address


con = dbapi.connect(host="localhost", port="5432", database="postgres", user="shelbyvanhooser")

cur = con.cursor()

#cur.execute("SELECT DISTINCT street, city, zip FROM contributor_valid WHERE lat IS NULL AND lon IS NULL;")
cur.execute("SELECT DISTINCT street, city, zipcode FROM points_map_political WHERE lat IS NULL AND lon IS NULL;")


at = 0
for street, city, zipcode in cur.fetchall()[30:]:
	try:
		print('Now computing on ', street, ' ', city, ' ', zipcode)
		curClient = theClients[at]
		print('Current client is ', at)
		d = curClient.geocode('{0}, {1}, OK {2}'.format(street, city, zipcode))[0]['geometry']['location']
		lat, lon = d['lat'], d['lng']
		print(lat, lon, street)
		cur.execute("""UPDATE points_map_political SET lat = {0}, lon = {1} WHERE street = '{2}' AND city = '{3}' AND zipcode = '{4}';""".format(lat, lon, street.replace("'", "''"), city, zipcode))
		con.commit()
		#at = (at + 1) % len(theClients)
	except:
		e = sys.exc_info()[0]
		print(e)
		print("Failed for address : ", street, " ", e)
		
		at = (at + 1) % len(theClients)

	

