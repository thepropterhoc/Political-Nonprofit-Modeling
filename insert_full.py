#AIzaSyADuv1109TAE_eDK-g0abWMkYrIYjJahmw
#AIzaSyCL9rEHZn_GZvIWwxDAnPajDKS1Uz1JWH8

import psycopg2 as dbapi
from datetime import datetime
import sys
import csv

con = dbapi.connect(host="localhost", port="5432", database="postgres", user="shelbyvanhooser")

cur = con.cursor()

cur.execute("""DROP TABLE IF EXISTS gifts;  CREATE TABLE IF NOT EXISTS gifts (first_name varchar(50), last_name varchar(80), amount real, fmv real, reference varchar(100), gift_date date);""")


reader = csv.reader(open("../datasets/GIFTS+NAME+AND+AMOUNT+ONLY.csv", 'r'))
reader.__next__()
for row in reader:
	row = [v.replace("'", "`") if not v == '' else None for v in row]
	last_name, first_name, amount, fmv, reference, gift_date = row
	insert_string = """INSERT INTO gifts VALUES ('{0}', '{1}', '{2}', '{3}', '{4}', TO_DATE('{5}', 'MM/DD/YYYY'));""".format(first_name, last_name, amount, fmv, reference, gift_date)
	cur.execute(insert_string)

con.commit()

"""
at = 0
for street, city, zipcode in cur.fetchall()[30:]:
	try:
		print('Now computing on ', street, ' ', city, ' ', zipcode)
		curClient = theClients[at]
		print('Current client is ', at)
		d = curClient.geocode('{0}, {1}, OK {2}'.format(street, city, zipcode))[0]['geometry']['location']
		lat, lon = d['lat'], d['lng']
		print(lat, lon, street)
		cur.execute("UPDATE contributor_valid SET lat = {0}, lon = {1} WHERE street = '{2}' AND city = '{3}' AND zip = '{4}';".format(lat, lon, street.replace("'", "''"), city, zipcode))
		con.commit()
		#at = (at + 1) % len(theClients)
	except:
		e = sys.exc_info()[0]
		print(e)
		print("Failed for address : ", street, " ", e)
		
		at = (at + 1) % len(theClients)

	

"""