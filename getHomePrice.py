import psycopg2 as dbapi
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults
import requests as req 
from xml.etree import ElementTree

zillow_data = ZillowWrapper(api_key='X1-ZWz1fhk8534tfv_47pen')

con = dbapi.connect(host="localhost", port="5432", database="postgres", user="shelbyvanhooser")

cur = con.cursor()

cur.execute("""
SELECT DISTINCT points_map.address, points_map.city, points_map.state, points_map.zip
FROM with_home_value
INNER JOIN points_map 
	USING(lat, lon)
;
""")
addresses = cur.fetchall()

for address, city, state, zipcode in addresses:
	print("Estimating for ", address, zipcode[:5])
	payload = {'citystatezip': '{0},{1} {2}'.format(city, state, zipcode), 'address': address}
	zidResponse = req.get('http://www.zillow.com/webservice/GetSearchResults.htm?zws-id=X1-ZWz1fhk8534tfv_47pen&address="{0}"&citystatezip="{1}, {2} {3}"'.format(address, city, state, zipcode))
	print(zidResponse.text)
	tree = ElementTree.fromstring(zidResponse.content)
	print(tree)
	break
	result = GetDeepSearchResults(deep_search_response)
	value = float(result.zestimate_amount)
	updateQuery = """UPDATE with_home_value SET home_value = {0} WHERE lat = {1} AND lon = {2};""".format(value, lat, lon)
	print(updateQuery)
	cur.execute(updateQuery)
	con.commit()

