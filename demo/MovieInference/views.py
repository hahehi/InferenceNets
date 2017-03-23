# -*- encoding=UTF-8 -*-
__author__ = 'HHY'


from django.shortcuts import render_to_response
from django.core import serializers
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpResponseRedirect, HttpResponse
from django.template import RequestContext
from django.views.decorators.csrf import csrf_exempt
from __init__ import allmid, allid, moviename, factorsname, oldidslst, newidslst, maptooldids, demonames
import os

def getfactorid(factorname):
    if factorname == "director":
        return 0
    if factorname == "actor":
        return 1
    if factorname == "genre":
        return 2
    if factorname == "country":
        return 3

def getfactorname(factorid):
    if factorid == 0:
        return "director"
    if factorid == 1:
        return "actor"
    if factorid == 2:
        return "genre"
    if factorid == 3:
        return "country"

def getid(factorname, name):
    factorid = getfactorid(factorname)
    namelst = factorsname[factorid]
    left = 0
    right = len(namelst) - 1
    oid = -1
    if factorid != 3:
        while left <= right:
            mid = int((left + right) / 2)
            if str(namelst[mid]) < str(name):
                left = mid + 1
            elif str(namelst[mid]) > str(name):
                right = mid - 1
            else:
                oid = mid
                break
    else:
        while left <= right:
            if str(namelst[left]) == str(name):
                oid = left
                break
            left += 1
    nid = oid
    if oid != -1 and factorid != 2:
        oldidlst = oldidslst[factorid]
        newidlst = newidslst[factorid]
        left = 0
        right = len(oldidlst) - 1
        while left <= right:
            mid = int((left + right) / 2)
            if oldidlst[mid] < oid:
                left = mid + 1
            elif oldidlst[mid] > oid:
                right = mid - 1
            else:
                nid = newidlst[mid]
                break
    return factorid, nid

def makequery(factorids, ids):
    query = [0] * 4
    for i in range(0, len(factorids)):
        query[factorids[i]] = ids[i] + 1
    return query

def queryresult(query, topn):
    file = open("results/query.txt", "w")
    for i in range(0, 4):
        print >> file, str(query[i])
    print >> file, str(topn)
    file.close()
    resultfilename = "results/" + str(topn)
    for i in range(0, 4):
        resultfilename += "-" + str(query[i])
    resultfilename += ".txt"
    while os.path.exists(resultfilename) == False:
        x = 1
    file = open(resultfilename, "r")
    indslst = []
    conslst = []
    line = file.readline()
    while line:
        linedata = line.replace("\n","").replace("\r","").split("\t")
        indslst.append(int(linedata[0]))
        conslst.append(float(linedata[1]))
        line = file.readline()
    file.close()
    resultlen = int(len(indslst) / topn)
    indss = []
    conss = []
    for i in range(0, resultlen):
        inds = indslst[(i*topn):(i*topn+topn)]
        indss.append(inds)
        cons = conslst[(i*topn):(i*topn+topn)]
        conss.append(cons)
    return indss, conss

def getname(factorid, nid):
    oid = nid
    if factorid != 2:
        oid = maptooldids[factorid][nid]
    name = factorsname[factorid][oid]
    return name

def getmoviestring(evifids, evis, qryfid, qry):
    entries = len(allmid)
    movies = []
    matchnum = 0
    for i in range(0, entries):
        matched = True
        for j in range(0, len(evifids)):
            if allid[evifids[j]][i] != evis[j]:
                matched = False
                break
        if matched and allid[qryfid][i] == qry and moviename[allmid[i] - 3000000] not in movies:
            if len(movies) < 3:
                movies.append(moviename[allmid[i] - 3000000])
            matchnum += 1
    if matchnum == 0:
        return "None"
    moviestr = str(matchnum) + " movies, e.g."
    for i in movies:
        moviestr += ', "' + i + '"'
    return moviestr

def getalldata():
    variables = {}
    for i in range(0, 4):
        variables[getfactorname(i)] = demonames[i]
    return variables

@csrf_exempt
def main(request):
    variables = getalldata()
    variables["status"] = 0
    variables = RequestContext(request, variables)
    return render_to_response("main.html", variables)

@csrf_exempt
def inference(request):
    factors = ["director", "actor", "genre", "country"]
    factorsdict = {"director": "Director", "actor": "Actor", "genre": "Genre", "country": "Country"}
    director = request.POST.get('director')
    actor = request.POST.get('actor')
    genre = request.POST.get('genre')
    country = request.POST.get('country')
    topn = int(request.POST.get('topn'))
    ifmovie = request.POST.get('ifmovie')
    factornames = []
    names = []
    if director != "nothingchosen":
        factornames.append('director')
        names.append(director)
    if actor != "nothingchosen":
        factornames.append('actor')
        names.append(actor)
    if genre != "nothingchosen":
        factornames.append('genre')
        names.append(genre)
    if country != "nothingchosen":
        factornames.append('country')
        names.append(country)
    factorids = []
    ids = []
    for i in range(0, len(factornames)):
        fid, tid = getid(factornames[i], names[i])
        if tid < 0 or topn <= 0 or topn > 100 or len(factornames) == 4:
            wrongmsg = "Top-n should be between 1 and 100."
            if tid < 0:
                wrongmsg = "Wrong name: " + names[i] + " for " + factornames[i] + "."
            if len(factornames) == 4:
                wrongmsg = "Nothing to infer."
            variables = getalldata()
            variables["status"] = 2
            variables["wrongmsg"] = wrongmsg
            variables = RequestContext(request, variables)
            return render_to_response("main.html", variables)
        factorids.append(fid)
        ids.append(tid)
    query = makequery(factorids, ids)
    indss, conss = queryresult(query, topn)
    variables = getalldata()
    index = 0
    for i in range(0, 4):
        if i not in factorids:
            wait = True
            while wait:
                try:
                    inds = indss[index]
                    wait = False
                except:
                    indss, conss = queryresult(query, topn)
                    wait = True
            cons = conss[index]
            variables[factors[i]+"res"] = {True}
            namelst = []
            conflst = []
            movielst = []
            if i != 2:
                for j in range(0, topn):
                    name = getname(i, inds[j] - 1)
                    namelst.append(name)
                    conflst.append(cons[j])
                    if ifmovie:
                        moviestring = getmoviestring(factorids, ids, i, inds[j] - 1)
                        movielst.append(moviestring)
            else:
                maximum = topn
                if maximum > 32:
                    maximum = 32
                for j in range(0, maximum):
                    name = getname(i, inds[j] - 1)
                    namelst.append(name)
                    conflst.append(cons[j])
                    if ifmovie:
                        moviestring = getmoviestring(factorids, ids, i, inds[j] - 1)
                        movielst.append(moviestring)
                for j in range(maximum, topn):
                    name = "-"
                    namelst.append(name)
                    conflst.append(0.0)
                    if ifmovie:
                        moviestring = "-"
                        movielst.append(moviestring)
            variables[factors[i]+"resnamelst"] = namelst
            variables[factors[i]+"resconflst"] = conflst
            if ifmovie:
                variables[factors[i]+"resmovielst"] = movielst
            index += 1
    data = []
    for j in range(0, topn):
        onedata = []
        for i in range(0, 4):
            if i not in factorids:
                onedata.append(j+1)
                onedata.append(variables[factors[i]+"resnamelst"][j])
                onedata.append(variables[factors[i]+"resconflst"][j])
                if ifmovie:
                    onedata.append(variables[factors[i]+"resmovielst"][j])
        data.append(onedata)
    evidence = "Evidence: "
    for i in range(0, len(factornames)):
        evidence += factorsdict[factornames[i]] + ": "+ names[i] + "; "
    evidence += "Inference Top-n: " + str(topn)
    variables["topn"] = topn
    variables["evidence"] = evidence
    variables["data"] = data
    variables["ifmovie"] = ifmovie
    variables["status"] = 1
    variables = RequestContext(request, variables)
    return render_to_response("main.html", variables)