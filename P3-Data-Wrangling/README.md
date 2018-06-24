# Introduction

Xiaodont TAN

I chose Honolulu as it has a beautiful Chinese name.

https://mapzen.com/data/metro-extracts/metro/honolulu_hawaii/


# 1. Data Audit and Problems Encountered

## 1.1 Street Name

The street names have the following problems:

- Some of them used abbreviations.
- Some of them used lower cases.
- Some names include things other than street names, such as "Pualei Cir, Apt 106" and 'Kaelepulu Dr, Kailua,'
- Two records 'South King' and 'King' are likely to miss a "Street" in the end.
- The street types of 'Pali Momi' and 'Ala Ike' are unknown.

The problematic street names are shown as follows.

    defaultdict(set,
                {'106': {'Pualei Cir, Apt 106'},
                 'Ave': {'Kalakaua Ave'},
                 'Blvd': {'Ala Moana Blvd'},
                 'Dr': {'Kipapa Dr'},
                 'Hwy': {'Kamehameha Hwy'},
                 'Ike': {'Ala Ike'},
                 'Kailua,': {'Kaelepulu Dr, Kailua,'},
                 'King': {'South King'},
                 'Momi': {'Pali Momi'},
                 'Pkwy': {'Meheula Pkwy'},
                 'St': {'Ala Pumalu St', 'Lusitania St'},
                 'St.': {'Lusitania St.'},
                 'highway': {'kanehameha highway'},
                 'king': {'king'}})

The street names are updated as follows.

    king => King Street
    South King => South King Street
    Pualei Cir, Apt 106 => Apt 106 Pualei Circle
    Lusitania St. => Lusitania Street
    Meheula Pkwy => Meheula Parkway
    Kaelepulu Dr, Kailua, => Kaelepulu Drive
    Pali Momi => Pali Momi
    Ala Ike => Ala Ike
    Kamehameha Hwy => Kamehameha Highway
    Ala Moana Blvd => Ala Moana Boulevard
    Kalakaua Ave => Kalakaua Avenue
    Ala Pumalu St => Ala Pumalu Street
    Lusitania St => Lusitania Street
    Kipapa Dr => Kipapa Drive
    kanehameha highway => Kanehameha Highway

## 1.2 Audit postal codes

The postal codes in Hawaii are expected to be five-digit numbers starting with "967" or "968". 

Auditing the postal codes show that some of the postal codes used zip5 + 4 digits format, which is also acceptable.

The only exepction is "HI 96819", in which "HI " should be deleted.

The postal codes that include more than five-digit number starting with "967" and "968" are as follows.

    defaultdict(int,
                {'96712-9998': 1,
                 '96734-9998': 1,
                 '96815-2518': 1,
                 '96815-2830': 1,
                 '96815-2834': 2,
                 '96817-1713': 1,
                 '96825-9998': 1,
                 '96826-4427': 1,
                 'HI 96819': 1})

## 1.3 Adudit phone numbers

Audit shows that the formats of the phone numbers are very inconsistent. 
- Some use "-", "." or "()" to seperate digits while others only use space. 
- Some have the country code while others don't.
- Some records have multiple phone numbers.

Some of the phone numbers are cleaned as follows.

    (808) 924-3303 => +1 808 9243303
    8089231234 => +1 808 9231234
    +1-808-545-3008 => +1 808 5453008
    (808) 769-6921 => +1 808 7696921
    +18089234852 => +1 808 9234852
    (808) 955-6329 => +1 808 9556329
    (808) 625-0411 => +1 808 6250411
    808-923-7024 => +1 808 9237024
    +1-808-922-4911 => +1 808 9224911
    808-831-6831 => +1 808 8316831
    +18089221544 => +1 808 9221544
    +1 (808) 946-0253 => +1 808 9460253
    (808) 926-9717 => +1 808 9269717
    +1 800 463 3339 => +1 800 4633339
    +1-808-836-7665 => +1 808 8367665
    808-922-8838 => +1 808 9228838
    +1-808-892-1820 => +1 808 8921820
    808-637-7472 => +1 808 6377472
    +1-808-532-8700 => +1 808 5328700
    808-486-2167 => +1 808 4862167
    1-808-955-7470 => +1 808 9557470
    +1-808-831-4820 => +1 808 8314820
    8088458044 => +1 808 8458044
    (808) 734-9000 => +1 808 7349000
    808-591-2513 => +1 808 5912513
    1 (808) 677-3335 => +1 808 6773335
    808-637-7710 => +1 808 6377710
    808-625-0320 => +1 808 6250320
    18089264167 => +1 808 9264167
    +1 (808) 733-0277 => +1 808 7330277
    808-637-6241 => +1 808 6376241
    +1 (808) 733-1540 => +1 808 7331540
    +1-808-9266162 => +1 808 9266162
    Honolulu CC Campus Security: (808) 284-1270 (cell), (808) 271-4836 (cell) => HonoluluCCCampusSecurity:8082841270cell,8082714836cell
    +1-808-839-6306 => +1 808 8396306
    +18089268955 => +1 808 9268955
    808-561-1000 => +1 808 5611000
    1.888.236.0799 => +1 888 2360799
    808-343-5501 => +1 808 3435501
    (808) 492-1637 => +1 808 4921637
    808-263-4414 => +1 808 2634414
    +1-808-836-9828 => +1 808 8369828
    (808)9242233 => +1 808 9242233
    8089224646 => +1 808 9224646
    +18088619966 => +1 808 8619966
    +1 888-389-3199 => +1 888 3893199
    +18088417984 => +1 808 8417984
    808.394.8770 => +1 808 3948770
    808-266-3996 => +1 808 2663996
    18003677060 => +1 800 3677060
    533-7557 => 5337557
    808-948-6111 => +1 808 9486111
    (808) 668-7367 => +1 808 6687367
    808-944-8968 => +1 808 9448968
    +1-808-954-7000 => +1 808 9547000
    6373000 => 6373000
    (808) 536-1330 => +1 808 5361330
    18084541434 => +1 808 4541434
    (808) 845-9498 => +1 808 8459498
    (808) 486-5100 => +1 808 4865100
    +1 323 423 6076 => +1 323 4236076


# 2. Data Overview with SQL

## Size of the files

    honolulu_hawaii.osm 54.2MB
    hawaii.db 29.3MB
    nodes.csv 21.1MB
    nodes_tags.csv 508.2kB
    ways.csv 1.6MB
    ways_nodes.csv 7.1MB
    ways_tags.csv 3.7MB

## Number of nodes

    SELECT COUNT(*) FROM nodes


Output

    248609

## Number of ways

    SELECT COUNT(*) FROM ways

Output

    26276

## Number of unique users

    SELECT COUNT(DISTINCT(e.uid))
    FROM (SELECT uid FROM nodes UNION ALL SELECT uid FROM ways) e

Output

    471

## Top 10 contributing users

    SELECT e.user, COUNT(*)
    as num FROM (SELECT user
    FROM nodes UNION ALL SELECT user FROM ways) e
    GROUP BY e.user 
    ORDER BY num DESC 
    LIMIT 10

Output

    [(u'Tom_Holland', 102060),
     (u'cbbaze', 14769),
     (u'ikiya', 12806),
     (u'kr4z33', 9852),
     (u'Chris Lawrence', 9214),
     (u'pdunn', 9067),
     (u'aaront', 8461),
     (u'woodpeck_fixbot', 8378),
     (u'julesreid', 7262),
     (u'bdiscoe', 5095)]

## Number of users who posted only one post

    SELECT COUNT(*)
    FROM 
    (SELECT e.user, COUNT(*) as num
    FROM (SELECT user FROM nodes UNION ALL SELECT user FROM ways) e
    GROUP BY e.user
    HAVING num=1) u

Output

    97

# 3. Additional Data Exploration

## Top 10 amenities

    SELECT value, COUNT(*) as num
    FROM nodes_tags
    WHERE key = 'amenity'
    GROUP BY value
    ORDER BY num DESC
    LIMIT 10

Output

    restaurant : 215
    fast_food : 108
    parking : 73
    toilets : 71
    cafe : 63
    waste_basket : 36
    bench : 30
    fire_station : 30
    parking_entrance : 25
    drinking_water : 24

## Types and number of tourism places

    SELECT value, COUNT(*) as num
    FROM nodes_tags
    WHERE key = 'tourism'
    GROUP BY value
    ORDER BY num DESC

output 

    attraction : 76
    viewpoint : 26
    hotel : 18
    museum : 13
    artwork : 10
    picnic_site : 8
    information : 5
    guest_house : 3
    hostel : 3
    gallery : 2
    alpine_hut : 1
    camp_site : 1

## If the restaurants have smoking areas

    SELECT nodes_tags.value, COUNT(*) as num
    FROM nodes_tags
    JOIN (SELECT DISTINCT(id) FROM nodes_tags WHERE value='restaurant') i
    ON nodes_tags.id=i.id
    WHERE nodes_tags.key='smoking' 
    GROUP BY nodes_tags.value
    ORDER BY num DESC

Output

    no 8
    isolated 1

## Sources of information

    SELECT value, COUNT(*) as num
    FROM nodes_tags
    WHERE key = 'source'
    GROUP BY value
    ORDER BY num DESC

Output

    Bing : 392
    USGS Geonames : 116
    survey : 42
    kr4z33 Survey : 25
    Yahoo : 13
    OpenStreetBugs : 2
    osmsync:dero : 2
    ourairports.com : 2
    wikipedia : 2
    (URL) : 1
    Hotel Guest : 1
    Landsat : 1
    NOAA U.S. Vector Shoreline : 1
    Owner : 1
    coal : 1
    http://jhchawaii.net/ : 1
    https://plus.google.com/101962710376509404433/about : 1
    tiger_import_dch_v0.6_20070809 : 1

# 4. Conclusion and Reflections

Some observations from the map are as follows. 1) The top1 contributor contributed over 30% of the posts. Top10 contributors contributed 70% of the posts 2) A large part of the data comes from Bing. 3) The information on the amenities, such as resturants are not very complete. 4) For some fields, such as phone number, the input stypes are very inconsistent.

Suggestions on encouraging data contribution: 1) The open street map can be built into a online community with rewarding systems 2) Functions such as photo uploading can be added.


Some observations from the map are as follows.
1) The top1 contributor contributed over 30% of the posts. Top10 contributors contributed 70% of the posts
2) A large part of the data comes from Bing.
3) The information on the amenities, such as resturants are not very complete.
4) For some fields, such as phone number, the input stypes are very inconsistent. 

Suggestions on encouraging data contribution:
1) The open street map can be built into a online community with rewarding systems
2) Functions such as photo uploading can be added.

However, although these methods might increase the data quantity, they might compromise data quality. On one hand, the users might input inaccurate or invalid information just for the rewards or for fun. On the other, the more sources the data come from, the more inconsistent the data might be.

For the issues regarding the validity and consistency of the data, data entry with forced schema or format might help. However, strict restrictions might discourage the users from participating. A less strict schema/format guide might also work.

For the issue regarding data inaccuracy, inviting users to "report inaccuracy" or “revise information” might be help identify and revise the inaccuracy data.

Besides, periodical data cleaning can help reduce the accumulated workload.

#  Reference

http://www.sqlitetutorial.net/sqlite-import-csv/ http://www.w3schools.com/sql/sql_syntax.asp https://stackoverflow.com/questions/19524554/suppress-code-in-nbconvert-ipython
https://guides.github.com/features/mastering-markdown/

