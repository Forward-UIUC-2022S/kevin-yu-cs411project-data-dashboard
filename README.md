# Academic World Keyword Data Dashboard

This project is about building a dashboard that allows the user to explore different keywords in the academic world database. The target users for this project are students or researchers who are interested in finding relevant information about keywords they are interested in. And the objective is to provide relevant information such as top publications, professors, and institutes for given keywords.

## Database Connection

- MySQL
  - Follow MP3's instruction to import data.
  - Constraints:
    - Table that uses to record university/institute information is named "Institute" and has a column for "name".
    - Table that uses to record keyword information is named "keyword" and has a column for "name".
  - A new table will be created to store user's favorite keywords
- MongoDB
  - Follow MP3's instruction to import data.
- Neo4j
  - Follow MP3's instruction to import data.
- Fill in necessary information in `secret.py`

## Setup

- Start MySQL server and Neo4j server
- Make sure to have pipenv installed
- Run `pipenv shell` to activate the shell
- Run `python app.py` to start the app
- Open the dashboard in a browser using http://127.0.0.1:8050/

```
- kevin-yu-cs411project-data-dashboard
    - app.py
    - secret.py
```

- `app.py` contains codes to build up the dashboard application
- `secret.py` contains connection strings/configurations to databses

## Design/Implementation

### `connectMySQL()`

Returns a MySQL connection engine object

### `connectMySQL()`

Returns a MongoDB client object

### `connectMySQL()`

Returns a Neo4j graph object

### `getInstitutes()`

Returns a list of all institutes in academic world

### `widget()`

Returns a widget that's either a graph or html/bootstrap components

### `updateWidget()`

A callback that updates the widget

## Demo Video

https://www.youtube.com/watch?v=3PMrDrggXD8
