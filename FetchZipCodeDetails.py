import zipcodes


def getZipCodeDetails(zipcode):
    details = zipcodes.matching(str(zipcode))
    if len(details) == 0:
        return {}
    else:
        return details


def getCityDetails(city):
    geo = str(city).split(', ')
    details = zipcodes.filter_by(city=str(geo[0].strip()), state=str((geo[1].strip())))
    if len(details) == 0:
        return {}
    else:
        return details


def changeZipCode(zipcode):
    pass
