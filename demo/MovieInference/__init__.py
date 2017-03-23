#read all entries
entryfile = open("data/data-1092682-s.txt", "r")
allmid = []
did = []
aid = []
gid = []
cid = []
line = entryfile.readline()
while line:
	linedata = line.replace("\r", "").replace("\n", "").split("\t")
	allmid.append(int(linedata[0]))
	did.append(int(linedata[1]))
	aid.append(int(linedata[2]))
	gid.append(int(linedata[3]))
	cid.append(int(linedata[4]))
	line = entryfile.readline()
allid = [did, aid, gid, cid]
entryfile.close()

#read all movie names
moviefile = open("data/movielist.txt", "r")
moviename = []
line = moviefile.readline()
while line:
	linedata = line.replace("\r", "").replace("\n", "").split("\t")
	moviename.append(linedata[1])
	line = moviefile.readline()
moviefile.close()

#read all factor names
idname = ["data/directorid-directorname.table",
"data/actorid-actorname.table",
"data/genreid-genrename.table",
"data/countryid-countryname.table"]
factorsname = []
for i in range(0, 4):
    file = open(idname[i], "r")
    namelst = []
    line = file.readline()
    while line:
        linedata = line.replace("\n","").replace("\r","").split("\t")
        namelst.append(linedata[1])
        line = file.readline()
    file.close()
    factorsname.append(namelst)

#read all idmaps from old ids to new ids
mapoldtonew = ["data/map-director-oldtonew.txt",
"data/map-actor-oldtonew.txt",
"data/map-genre-oldtonew.txt",
"data/map-country-oldtonew.txt"]
oldidslst = []
newidslst = []
for i in range(0, 4):
    oldidlst = []
    newidlst = []
    if i != 2:
	    file = open(mapoldtonew[i], "r")
	    line = file.readline()
	    while line:
	        linedata = line.replace("\n","").replace("\r","").split("\t")
	        oldidlst.append(int(linedata[0]))
	        newidlst.append(int(linedata[1]))
	        line = file.readline()
	    file.close()
    oldidslst.append(oldidlst)
    newidslst.append(newidlst)

#read all idmaps from new ids to old ids
mapnewtoold = ["data/map-director-newtoold.txt",
"data/map-actor-newtoold.txt",
"data/map-genre-newtoold.txt",
"data/map-country-newtoold.txt"]
maptooldids = []
for i in range(0, 4):
	lst = []
	if i != 2:
		file = open(mapnewtoold[i], "r")
		line = file.readline()
		while line:
			linedata = line.replace("\n","").replace("\r","").split("\t")
			lst.append(int(linedata[1]))
			line = file.readline()
		file.close()
	maptooldids.append(lst)

#read all factors names for demo
listname = ["data/directorlist.txt",
"data/actorlist.txt",
"data/genrelist.txt",
"data/countrylist.txt"]
demonames = []
for i in range(0, 4):
    file = open(listname[i], "r")
    namelst = []
    line = file.readline()
    while line:
        linedata = line.replace("\n","").replace("\r","").split("\t")
        namelst.append(linedata[1])
        line = file.readline()
    file.close()
    demonames.append(namelst)