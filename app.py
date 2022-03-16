# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from numpy import delete
from dash import Dash, html, dcc, Input, Output, callback, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

import dash_bootstrap_components as dbc


# db
import secret
from sqlalchemy import create_engine, inspect, Table, Column, MetaData, VARCHAR
from pymongo import MongoClient
from py2neo import Graph


# static
# overview of keywords
# 1. Overview of keywords, number of keywords, most popular keywords
# 2. keyword with respect to universities

# input
# 3. Compare schools with number of publications & keyword score in computer science
# 4. Most relevant publications given an input keyword

# update
# 5. update?
# 6. update?


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# init databases


def connectMySQL():
    conn = "mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
        secret.mysqldbuser,
        secret.mysqldbpass,
        secret.mysqldbhost,
        secret.mysqlport,
        secret.mysqldbname,
    )
    engine = create_engine(conn)
    return engine


def connectMongo():
    conn = "mongodb://{0}".format(secret.mongodbhost)
    myclient = MongoClient(conn)
    return myclient


def connectNeo():
    graph = Graph(
        secret.neo4jhost,
        auth=(secret.neo4juser, secret.neo4jpass),
        name=secret.neo4jname,
    )
    return graph


def getInstitutes():
    mssql_engine = connectMySQL()
    with mssql_engine.connect() as connection:
        sql = """
            SELECT name
            FROM institute
        """
        rows = pd.read_sql(sql, connection)
        return [row["name"] for _, row in rows.iterrows()]


def widget1(limit):
    graph = connectNeo()
    keywordCountQuery = """
            MATCH (k:KEYWORD)
            RETURN count(k) AS c
    """
    topKeywordQuery = f"""
            MATCH (p:PUBLICATION)-[rels:LABEL_BY]->(k:KEYWORD)
            UNWIND rels as rel 
            RETURN k.name as name ,count(p) as c
            ORDER BY c DESC
            LIMIT {limit}
    """
    keywordCount = list(graph.run(keywordCountQuery))
    topKeyword = list(graph.run(topKeywordQuery))
    d = {
        "name": [tk["name"] for tk in topKeyword],
        "count": [tk["c"] for tk in topKeyword],
    }
    df = pd.DataFrame(data=d)
    fig = px.scatter(
        df,
        x="name",
        y="count",
        color="name",
        size="count",
        hover_data=["name", "count"],
    )
    count = 0
    for row in keywordCount:
        count = row["c"]
    fig.update_layout(title="Total keywords: {}, Top 10 Popular Keywords".format(count))
    return fig


@callback(Output("widget1", "figure"), Input("widget-1-input", "value"))
def updateWidget1(limit):
    return widget1(limit)


def widget2():
    client = connectMongo()
    with client:
        db = client[secret.mongodbname]
        collection = db["faculty"]
        pipeline = [
            {
                "$match": {
                    "affiliation.name": "University of illinois at Urbana Champaign"
                }
            },
            {"$unwind": "$keywords"},
            {
                "$group": {
                    "_id": {
                        "affiliation": "$affiliation",
                        "keyword_name": "$keywords.name",
                    },
                    "keyword_score": {"$sum": "$keywords.score"},
                    "keyword_count": {"$sum": 1},
                }
            },
            {"$sort": {"keyword_score": -1, "keyword_count": -1}},
            # {
            #     "$group" :
            #     {
            #         "_id" : "$_id.affiliation",
            #         "keywords" :
            #         {
            #             "$push" : {"name" : "$_id.keyword_name",
            #             "score" : "$keyword_score",
            #             "count" : "$keyword_count"}
            #         }
            #     }
            # },
            {
                "$project": {
                    "_id": 0,
                    "affiliation": "$_id.affiliation",
                    "keyword_name": "$_id.keyword_name",
                    "keyword_score": 1,
                    "keyword_count": 1
                    # "top_10_keywords" : { "$slice" : [ "$keywords", 10 ] }
                }
            },
        ]

        result = list(collection.aggregate(pipeline))
        keywordCount = len(result)
        result = result[:10]
        d = {
            "keywordName": [res["keyword_name"] for res in result],
            "mentionedTimes": [res["keyword_count"] for res in result],
            "totalScore": [res["keyword_score"] for res in result],
        }
        df = pd.DataFrame(data=d)
        fig = px.pie(df, values="totalScore", names="keywordName")
        fig.update_layout(
            title="top 10 keywords for University of illinois at Urbana Champaign"
        )
        return (
            result[0]["affiliation"]["photoUrl"],
            f"Number of Unique Keywords: {keywordCount}",
            fig,
        )


@callback(
    Output("institute-logo", "src"),
    Output("num-keyword", "children"),
    Output("widget2", "figure"),
    Input("institute-dropdown", "value"),
)
def updateWidget2(institute_name):
    client = connectMongo()
    with client:
        db = client[secret.mongodbname]
        collection = db["faculty"]
        pipeline = [
            {"$match": {"affiliation.name": "{}".format(institute_name)}},
            {"$unwind": "$keywords"},
            {
                "$group": {
                    "_id": {
                        "affiliation": "$affiliation",
                        "keyword_name": "$keywords.name",
                    },
                    "keyword_score": {"$sum": "$keywords.score"},
                    "keyword_count": {"$sum": 1},
                }
            },
            {"$sort": {"keyword_score": -1, "keyword_count": -1}},
            # {
            #     "$group" :
            #     {
            #         "_id" : "$_id.affiliation",
            #         "keywords" :
            #         {
            #             "$push" : {"name" : "$_id.keyword_name",
            #             "score" : "$keyword_score",
            #             "count" : "$keyword_count"}
            #         }
            #     }
            # },
            {
                "$project": {
                    "_id": 0,
                    "affiliation": "$_id.affiliation",
                    "keyword_name": "$_id.keyword_name",
                    "keyword_score": 1,
                    "keyword_count": 1
                    # "top_10_keywords" : { "$slice" : [ "$keywords", 10 ] }
                }
            },
        ]

        result = list(collection.aggregate(pipeline))
        keywordCount = len(result)
        result = result[:10]
        d = {
            "keywordName": [res["keyword_name"] for res in result],
            "mentionedTimes": [res["keyword_count"] for res in result],
            "totalScore": [res["keyword_score"] for res in result],
        }
        df = pd.DataFrame(data=d)
        df = pd.DataFrame(data=d)
        fig = px.pie(df, values="totalScore", names="keywordName")
        fig.update_layout(title=f"top 10 keywords for {institute_name}")
        return (
            result[0]["affiliation"]["photoUrl"],
            f"Number of Unique Keywords: {keywordCount}",
            fig,
        )


def widget3():
    client = connectMongo()
    with client:
        db = client[secret.mongodbname]
        collection = db["publications"]
        fig = go.Figure()
        fig.update_layout(
            yaxis_title="keyword score", xaxis_title="number of citations"
        )

        fig.update_layout(title="top relevant publications" + "for computer science")

        pipeline = [
            {"$unwind": "$keywords"},
            {"$match": {"keywords.name": "computer science"}},
            {"$sort": {"keywords.score": -1, "numCitations": -1}},
            {"$limit": 10},
        ]
        result = list(collection.aggregate(pipeline))
        d = {
            "numCitations": [res["numCitations"] for res in result],
            "relevance": [res["keywords"]["score"] for res in result],
            "titles": [res["title"] for res in result],
        }
        df = pd.DataFrame(data=d)

        fig = px.scatter(
            df,
            x="numCitations",
            y="relevance",
            color="titles",
        )
        fig.update_traces(
            marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )
        return fig


@callback(Output("widget3", "figure"), Input("keyword-input-3", "value"))
def updateWidget3(keyword):
    # if not keyword:
    #     return widget3()
    client = connectMongo()
    with client:
        db = client[secret.mongodbname]
        collection = db["publications"]
        fig = go.Figure()
        fig.update_layout(
            yaxis_title="keyword score", xaxis_title="number of citations"
        )

        if not keyword:
            fig.update_layout(
                title="most relevant publications and professors for given keyword"
            )
            return fig

        fig.update_layout(title="top relevant publications" + "for {}".format(keyword))

        keyword = keyword.strip()

        pipeline = [
            {"$unwind": "$keywords"},
            {"$match": {"keywords.name": "{}".format(keyword)}},
            {"$sort": {"keywords.score": -1, "numCitations": -1}},
            {"$limit": 10},
        ]
        result = list(collection.aggregate(pipeline))
        d = {
            "numCitations": [res["numCitations"] for res in result],
            "relevance": [res["keywords"]["score"] for res in result],
            "titles": [res["title"] for res in result],
        }
        df = pd.DataFrame(data=d)

        fig = px.scatter(
            df,
            x="numCitations",
            y="relevance",
            color="titles",
        )
        fig.update_traces(
            marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )
        return fig


def widget4():
    client = connectMongo()
    with client:
        db = client[secret.mongodbname]
        collection = db["faculty"]
        pipeline = [
            {"$match": {"keywords.name": "computer science"}},
            {"$unwind": "$keywords"},
            {"$match": {"keywords.name": "computer science"}},
            {"$sort": {"keywords.score": -1}},
            {"$limit": 1},
        ]
        result = collection.aggregate(pipeline)
        topProfessor = list(result)[0]
        collection = db["publications"]
        result = collection.find(
            {
                "$and": [
                    {"id": {"$in": topProfessor["publications"]}},
                    {"keywords.name": "computer science"},
                ]
            }
        ).limit(5)

        publications = list(result)
    return (
        topProfessor["name"],
        f'Keyword Score: {topProfessor["keywords"]["score"]}',
        f'{topProfessor["position"]} at {topProfessor["affiliation"]["name"]}',
        topProfessor["photoUrl"],
        [dbc.ListGroupItem(p["title"]) for p in publications],
    )


@callback(
    Output("widget-4-title", "children"),
    Output("widget-4-subtitle", "children"),
    Output("widget-4-text", "children"),
    Output("widget-4-photo", "src"),
    Output("widget-4-list", "children"),
    Input("keyword-input-4", "value"),
)
def updateWidget4(keyword):
    client = connectMongo()
    with client:
        db = client[secret.mongodbname]
        collection = db["faculty"]
        pipeline = [
            {"$match": {"keywords.name": f"{keyword}"}},
            {"$unwind": "$keywords"},
            {"$match": {"keywords.name": f"{keyword}"}},
            {"$sort": {"keywords.score": -1}},
            {"$limit": 1},
        ]
        result = collection.aggregate(pipeline)

        topProfessor = list(result)

        if not topProfessor:
            return ("Not Found", "Please try again", "", "", [])

        topProfessor = topProfessor[0]
        collection = db["publications"]
        result = collection.find(
            {
                "$and": [
                    {"id": {"$in": topProfessor["publications"]}},
                    {"keywords.name": f"{keyword}"},
                ]
            }
        ).limit(5)

        publications = list(result)
    return (
        topProfessor["name"],
        f'Keyword Score: {topProfessor["keywords"]["score"]}',
        f'{topProfessor["position"]} at {topProfessor["affiliation"]["name"]}',
        topProfessor["photoUrl"],
        [dbc.ListGroupItem(p["title"]) for p in publications],
    )


def widget5():
    # update keywords score in publication
    pass


@callback(
    Output("widget-5-output", "children"),
    Output("widget-5-output", "color"),
    Input("keyword-input-5", "value"),
)
def updateWidget5(keyword):
    if not keyword:
        return "", "primary"
    # update keywords score in faculty
    mssql_engine = connectMySQL()
    ins = inspect(mssql_engine)
    exist = False
    if ins.dialect.has_table(mssql_engine.connect(), "favorite_keyword"):
        exist = True

    with mssql_engine.connect() as connection:

        meta = MetaData()
        favoariteKeyword = Table(
            "favorite_keyword",
            meta,
            Column("id", VARCHAR(length=24), primary_key=True),
            Column("name", VARCHAR(length=255)),
        )
        if not exist:
            # create table
            meta.create_all(mssql_engine)

        keyword = keyword.strip().lower()
        keywordExistQuery = f"""
            SELECT 
                *
            FROM
                keyword
            WHERE
                name = "{keyword}"
        """
        keywordExist = pd.read_sql_query(keywordExistQuery, connection)
        if keywordExist.empty:
            return "Keyword Doesn't Exist", "danger"

        keywordDuplicateQuery = f"""
            SELECT 
                *
            FROM
                favorite_keyword
            WHERE
                name = "{keyword}"
        """
        keywordDuplicate = pd.read_sql_query(keywordDuplicateQuery, connection)

        if not keywordDuplicate.empty:
            return "Keyword Already Exist", "danger"

        _ = connection.execute(
            favoariteKeyword.insert().values(
                id=keywordExist["id:ID"][0], name=keywordExist["name"][0]
            )
        )
        # connection.commit()

        return f"{keyword} added", "success"


def widget6():
    mssql_engine = connectMySQL()
    with mssql_engine.connect() as connection:
        favoriteKeywordQuery = """
            SELECT *
            FROM favorite_keyword
        """
        favoriteKeywords = pd.read_sql_query(favoriteKeywordQuery, connection)

        return [dbc.ListGroupItem(fk, key=fk) for fk in favoriteKeywords["name"]]


@callback(
    Output("widget-6-output", "children"),
    Output("widget-7-input-options", "options"),
    Input("keyword-input-5", "value"),
    Input("widget-7-input-button", "n_clicks"),
)
def updateWidget6(keyword, n_clicks):
    mssql_engine = connectMySQL()
    with mssql_engine.connect() as connection:
        favoriteKeywordQuery = """
            SELECT *
            FROM favorite_keyword
        """
        favoriteKeywords = pd.read_sql_query(favoriteKeywordQuery, connection)
        l1, l2 = [dbc.ListGroupItem(fk, key=fk) for fk in favoriteKeywords["name"]], [
            {
                "label": "Choose a keyword to delete from",
                "value": "",
                "disabled": True,
                "selected": True,
            }
        ]
        l2.extend([{"label": fk, "value": fk} for fk in favoriteKeywords["name"]])
        return l1, l2


@callback(
    Output("widget-7-output", "children"),
    Output("widget-7-output", "color"),
    [Input("widget-7-input-button", "n_clicks")],
    [State("widget-7-input-options", "value")],
)
def updateWidget7(n_clicks, keyword):
    if not keyword:
        return "", "primary"
    mssql_engine = connectMySQL()
    with mssql_engine.connect() as connection:
        meta = MetaData()
        favoariteKeyword = Table(
            "favorite_keyword",
            meta,
            Column("id", VARCHAR(length=24), primary_key=True),
            Column("name", VARCHAR(length=255)),
        )

        _ = connection.execute(
            favoariteKeyword.delete().where(favoariteKeyword.c.name == keyword)
        )

        return f"{keyword} deleted", "success"


def widget8():
    client = connectMongo()
    with client:
        db = client[secret.mongodbname]
        collection = db["faculty"]
        favoriteKeywords = []
        mssql_engine = connectMySQL()
        with mssql_engine.connect() as connection:
            favoriteKeywordQuery = """
            SELECT *
            FROM favorite_keyword
            """
            res = pd.read_sql_query(favoriteKeywordQuery, connection)
            favoriteKeywords = [fk for fk in res["name"]]

        if not favoriteKeywords:
            return []

        pipeline = [
            {"$unwind": "$keywords"},
            {"$match": {"$or": [{"keywords.name": fk} for fk in favoriteKeywords]}},
            {
                "$group": {
                    "_id": {"id": "$id", "name": "$name"},
                    "count": {"$sum": 1},
                    "score": {"$sum": "$keywords.score"},
                }
            },
            {"$sort": {"count": -1, "score": -1}},
            {"$limit": 5},
        ]

        result = collection.aggregate(pipeline)
        matchedProfessors = [
            (res["_id"]["name"], res["count"], res["score"], res["_id"]["id"])
            for res in list(result)
        ]
        return [
            dbc.ListGroupItem(
                f"{mp[0]},match count:{mp[1]},match score:{mp[2]}", key=mp[3]
            )
            for mp in matchedProfessors
        ]


def widget9():
    client = connectMongo()
    with client:
        db = client[secret.mongodbname]
        collection = db["publications"]
        favoriteKeywords = []
        mssql_engine = connectMySQL()
        with mssql_engine.connect() as connection:
            favoriteKeywordQuery = """
            SELECT *
            FROM favorite_keyword
            """
            res = pd.read_sql_query(favoriteKeywordQuery, connection)
            favoriteKeywords = [fk for fk in res["name"]]
        if not favoriteKeywords:
            return []
        pipeline = [
            {"$unwind": "$keywords"},
            {"$match": {"$or": [{"keywords.name": fk} for fk in favoriteKeywords]}},
            {
                "$group": {
                    "_id": {"id": "$id", "title": "$title"},
                    "count": {"$sum": 1},
                    "score": {"$sum": "$keywords.score"},
                }
            },
            {"$sort": {"count": -1, "score": -1}},
            {"$limit": 5},
        ]

        result = collection.aggregate(pipeline)
        matchedPublications = [
            (res["_id"]["title"], res["count"], res["score"], res["_id"]["id"])
            for res in list(result)
        ]
        return [
            dbc.ListGroupItem(
                f"{mp[0]},match count:{mp[1]},match score:{mp[2]}", key=mp[3]
            )
            for mp in matchedPublications
        ]


@callback(
    Output("widget-8-output", "children"),
    Input("keyword-input-5", "value"),
    Input("widget-7-input-button", "n_clicks"),
)
def updateWidget8(value, n_clicks):
    return widget8()


@callback(
    Output("widget-9-output", "children"),
    Input("keyword-input-5", "value"),
    Input("widget-7-input-button", "n_clicks"),
)
def updateWidget9(value, n_clicks):
    return widget9()


(
    InstituteURL,
    numKeywords,
    f2,
) = widget2()
instituteList = getInstitutes()
f3 = widget3()
f4_1, f4_2, f4_3, f4_4, f4_5 = widget4()
f6 = widget6()
f8 = widget8()
f9 = widget9()
connectNeo()


# widget 1

app.layout = dbc.Container(
    children=[
        # title
        html.H1(
            children="Explore Keywords in Academic World", style={"textAlign": "center"}
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            children="Keyword Overview", style={"textAlign": "center"}
                        ),
                        dcc.Slider(
                            min=5, max=25, step=5, value=10, id="widget-1-input"
                        ),
                        dcc.Graph(id="widget1"),
                    ],
                    width=5,
                ),
                dbc.Col(
                    [
                        html.H2(
                            children="Top Publications with Input Keyword",
                            style={"textAlign": "center"},
                        ),
                        dbc.Input(
                            id="keyword-input-3",
                            type="text",
                            value="computer science",
                            placeholder='Input a keyword and press "ENTER" to search',
                            debounce=True,
                        ),
                        dcc.Graph(id="widget3", figure=f3),
                    ]
                ),
            ]
        ),
        # widget 1
        # dcc.Graph(id="widget1"),
        html.Br(),
        # widget 2, static, mongodb
        # keyword facts about university
        html.H2(
            children="Top Keywords in Different Universities",
            style={"textAlign": "center"},
        ),
        html.Div(
            [
                dbc.Row(
                    dcc.Dropdown(
                        instituteList,
                        "University of illinois at Urbana Champaign",
                        id="institute-dropdown",
                    )
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                children=[
                                    html.H3(
                                        numKeywords,
                                        style={"textAlign": "center"},
                                        id="num-keyword",
                                    ),
                                    html.Img(
                                        src=f"{InstituteURL}",
                                        id="institute-logo",
                                        style={
                                            "width": 400,
                                            "height": 400,
                                            "object-fit": "contain",
                                        },
                                    ),
                                ]
                            )
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="widget2",
                                figure=f2,
                                style={"display": "inline-block"},
                            )
                        ),
                    ]
                ),
            ]
        ),
        html.Br(),
        # widget 3, input, mongodb
        # given a keyword, find most relevant publications and professors
        # dcc.Graph(id="widget3", figure=f3),
        html.Br(),
        # # widget 4, input
        # # Top professor and his/her publications for given input keyword
        html.H2(
            children="Explore Top Professors with Input Keyword",
            style={"textAlign": "center"},
        ),
        dbc.Input(
            id="keyword-input-4",
            type="text",
            placeholder='Input a keyword and press "ENTER" to search',
            value="computer science",
            debounce=True,
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        children=[
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            f4_1,
                                            className="card-title",
                                            id="widget-4-title",
                                        ),
                                        html.H6(
                                            f4_2,
                                            className="card-subtitle",
                                            id="widget-4-subtitle",
                                        ),
                                        html.P(
                                            f4_3,
                                            className="card-text",
                                            id="widget-4-text",
                                        ),
                                        html.Img(
                                            src=f4_4,
                                            style={
                                                "width": 250,
                                                "height": 250,
                                                "object-fit": "contain",
                                            },
                                            id="widget-4-photo",
                                        ),
                                    ]
                                ),
                                style={"width": "18rem"},
                            ),
                        ]
                    ),
                    lg=3,
                ),
                dbc.Col(html.Div(""), lg=3),
                dbc.Col(
                    html.Div(
                        children=[
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            "Featured Publications",
                                            className="card-title",
                                        ),
                                        dbc.ListGroup(
                                            f4_5,
                                            id="widget-4-list",
                                        ),
                                    ]
                                ),
                                style={"width": "18rem"},
                            ),
                        ]
                    ),
                    lg=3,
                ),
            ],
            justify="center",
        ),
        # widget 5, input + update
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            children="Add Your Favorite Keywords",
                            style={"textAlign": "center"},
                        ),
                        dbc.Input(
                            id="keyword-input-5",
                            type="text",
                            placeholder='Input a keyword and press "ENTER" to add',
                            debounce=True,
                        ),
                        dbc.Alert("", id="widget-5-output", color="primary"),
                    ]
                ),
                dbc.Col(
                    [
                        html.H2(
                            children="Delete Your Favorite Keywords",
                            style={"textAlign": "center"},
                        ),
                        dbc.InputGroup(
                            [
                                dbc.Select(id="widget-7-input-options"),
                                dbc.Button(
                                    "Delete",
                                    id="widget-7-input-button",
                                    n_clicks=0,
                                    color="danger",
                                ),
                            ]
                        ),
                        dbc.Alert("", id="widget-7-output", color="primary"),
                    ]
                ),
            ]
        ),
        # widget 7 delete keyword
        # widget 6, input + update
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        children=[
                            html.H2(
                                children="Favorite Keywords",
                                style={"textAlign": "center"},
                            ),
                            dbc.ListGroup(f6, id="widget-6-output"),
                        ],
                    )
                ),
                dbc.Col(
                    html.Div(
                        children=[
                            html.H2(
                                children="Recomended Professors",
                                style={"textAlign": "center"},
                            ),
                            dbc.ListGroup(f8, id="widget-8-output"),
                        ]
                    )
                ),
                dbc.Col(
                    html.Div(
                        children=[
                            html.H2(
                                children="Recomended Publications",
                                style={"textAlign": "center"},
                            ),
                            dbc.ListGroup(f9, id="widget-9-output"),
                        ]
                    )
                ),
            ],
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
